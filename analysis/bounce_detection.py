"""
analysis/bounce_detection.py

Finds when and where the ball touches the ground.

Between two events the ball flies freely, so in the image it follows (almost) a
parabola, and an event is a corner where two parabolas meet:
  1. the real detections (never the filled gaps, which round the corners off) are
     split into the best sequence of parabolas by dynamic programming;
  2. each junction gives the meeting time, between two frames if needed (the ball
     covers 0.5-1 m per frame at 30 fps), the meeting point and the velocity jump;
  3. physics decides what a junction can be. The ground only pushes the ball UP
     the image, and a bounce lies on the court: the homography is exact there,
     while an airborne ball projects elsewhere. A hit needs a player within reach;
  4. the rally order (hit, bounce on the other side, hit, ...) picks the labels,
     which is what tells a near bounce from the near hit that follows it.

Velocity jumps are in m/s (px/frame x fps / local px-per-metre), so the thresholds
do not depend on resolution or framing. The fit tolerances are in pixels, for
1080p footage.

Use (from project root, after the ball and player CSVs exist):
    python analysis/bounce_detection.py --video data/Input_video2.mp4
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.shot_analysis import (
    BOUNCE_FAR,
    BOUNCE_NEAR,
    HIT_FAR,
    HIT_NEAR,
    HIT_OF_PLAYER,
    _ball_near_player,
    _nearest_box,
    ball_track_for_shots,
    detect_hits,
    detect_serves,
    load_ball_track,
    load_player_boxes,
)
from utils.court_converter import CourtConverter
from utils.court_geometry import NET, L_m, W_m

BOUNCE_PARAMS = {
    # Segmentation (pixels, for 1080p footage)
    "piece_penalty": 120.0,   # cost (px^2) of one more parabola
    "residual_cap_px": 15.0,  # a wrong point costs at most this much squared
    "min_piece": 5,           # detections in a parabola
    "max_piece": 75,
    "max_piece_s": 3.0,
    # Junctions
    "max_hidden_s": 0.2,      # longest detection gap across a junction
    "max_fit_gap_px": 15.0,   # the two parabolas must really meet
    "reliable_gap_px": 4.0,   # ...and meet this closely for a reliable landing point
    # Labels (m/s)
    "min_jump_ms": 1.0,       # below this a junction is a perspective artefact
    "min_bounce_up_ms": 5.0,  # upward velocity jump of a bounce
    "max_side_ratio": 0.5,    # sideways jump / upward jump of a bounce
    "court_margin_m": 1.0,    # a bounce lies on the court, give or take this
    # Rally order
    "reward": 1.0,            # gain of labelling one more junction
    "missed_one": 0.6,        # order cost: one event skipped (a volley, a missed hit)
    "missed_two": 1.2,        # order cost: two events skipped
    "repeated": 0.9,          # order cost: same event twice (a second bounce)
    "new_point_s": 2.0,       # after this long the order starts over
    "serve_tol_s": 0.2,       # a junction this close to a serve contact is the serve
}

# The rally cycle: each event is followed by the next one, wrapping around.
RALLY_CYCLE = (HIT_NEAR, BOUNCE_FAR, HIT_FAR, BOUNCE_NEAR)


# ── loading ────────────────────────────────────────────────────────────────────

def load_real_detections(ball_csv: str) -> pd.DataFrame:
    """Frames with a real ball detection (frame, cx, cy), without the filled gaps."""
    df = pd.read_csv(ball_csv)
    if "interpolated" in df.columns:
        df = df[df["interpolated"] == 0]
    df = df.dropna(subset=["cx", "cy"])
    return df[["frame", "cx", "cy"]].sort_values("frame").reset_index(drop=True)


# ── parabola segmentation ──────────────────────────────────────────────────────

def _fit_parabola(t, pts):
    """Quadratic fit of pts(t): (centre time, 3x2 coefficients)."""
    t0 = t.mean()
    d = t - t0
    return t0, np.linalg.lstsq(np.c_[np.ones_like(d), d, d * d], pts, rcond=None)[0]


def _position(fit, t):
    t0, c = fit
    d = t - t0
    return c[0] + np.outer(d, c[1]) + np.outer(d * d, c[2])


def _velocity(fit, t):
    t0, c = fit
    return c[1] + 2.0 * c[2] * (t - t0)


def segment_trajectory(t, pts, fps, params=None):
    """Best split of the track into parabolas, as [(start, end), ...] index ranges.

    Optimal partitioning: minimises the capped squared residuals plus a penalty
    per parabola. Points that fit no parabola are left out at the cap's cost.
    """
    p = params or BOUNCE_PARAMS
    n = len(t)
    cap = p["residual_cap_px"]
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n + 1, int)
    is_piece = np.zeros(n + 1, bool)
    max_span = p["max_piece_s"] * fps
    for j in range(1, n + 1):
        best[j] = best[j - 1] + cap * cap
        prev[j] = j - 1
        for i in range(max(0, j - p["max_piece"]), j - p["min_piece"] + 1):
            if t[j - 1] - t[i] > max_span:
                continue
            fit = _fit_parabola(t[i:j], pts[i:j])
            err = np.linalg.norm(pts[i:j] - _position(fit, t[i:j]), axis=1)
            cost = best[i] + (np.minimum(err, cap) ** 2).sum() + p["piece_penalty"]
            if cost < best[j]:
                best[j], prev[j], is_piece[j] = cost, i, True
    pieces = []
    j = n
    while j > 0:
        if is_piece[j]:
            pieces.append((prev[j], j))
        j = prev[j]
    return pieces[::-1]


def _px_per_metre(conv, x_m, y_m):
    """Image scale at a ground point; it also holds for heights above that point."""
    m_to_px = np.linalg.inv(conv._H)
    ends = np.array([[[x_m - 0.5, y_m]], [[x_m + 0.5, y_m]]], dtype=np.float64)
    a, b = cv2.perspectiveTransform(ends, m_to_px)[:, 0]
    return float(np.hypot(*(b - a)))


def find_junctions(t, pts, pieces, conv, fps, params=None):
    """Where consecutive parabolas meet: one row per junction.

    jump_up_ms is the change of vertical speed (positive = pushed up the image),
    jump_side_ms the change of horizontal image speed.
    """
    p = params or BOUNCE_PARAMS
    fits = [_fit_parabola(t[i:j], pts[i:j]) for i, j in pieces]
    rows = []
    for k in range(len(pieces) - 1):
        last, first = t[pieces[k][1] - 1], t[pieces[k + 1][0]]
        if first - last > p["max_hidden_s"] * fps:
            continue
        ts = np.arange(last - 3.0, first + 3.05, 0.1)
        before, after = _position(fits[k], ts), _position(fits[k + 1], ts)
        gaps = np.linalg.norm(before - after, axis=1)
        m = int(np.argmin(gaps))
        if gaps[m] > p["max_fit_gap_px"]:
            continue
        cx, cy = (before[m] + after[m]) / 2.0
        x_m, y_m = conv.to_meters(cx, cy)
        jump = _velocity(fits[k + 1], ts[m]) - _velocity(fits[k], ts[m])
        to_ms = fps / max(_px_per_metre(conv, x_m, y_m), 1e-6)
        rows.append({"frame": float(ts[m]), "cx": cx, "cy": cy, "x_m": x_m, "y_m": y_m,
                     "jump_up_ms": -jump[1] * to_ms, "jump_side_ms": jump[0] * to_ms,
                     "fit_gap_px": float(gaps[m])})
    return pd.DataFrame(rows, columns=["frame", "cx", "cy", "x_m", "y_m", "jump_up_ms",
                                       "jump_side_ms", "fit_gap_px"])


# ── labelling ──────────────────────────────────────────────────────────────────

def _allowed_labels(junction, boxes, params):
    """The events a junction can physically be."""
    p = params
    up, side = junction.jump_up_ms, junction.jump_side_ms
    if np.hypot(up, side) < p["min_jump_ms"]:
        return []
    labels = []
    margin = p["court_margin_m"]
    on_court = (-margin <= junction.x_m <= W_m + margin
                and -margin <= junction.y_m <= L_m + margin)
    if on_court and up >= p["min_bounce_up_ms"] and abs(side) <= p["max_side_ratio"] * up:
        labels.append(BOUNCE_FAR if junction.y_m < NET else BOUNCE_NEAR)
    f = int(round(junction.frame))
    for pid, hit in HIT_OF_PLAYER.items():
        box = _nearest_box(boxes, f, pid)
        if box is not None and _ball_near_player((junction.cx, junction.cy), box):
            labels.append(hit)
    return labels


def _order_cost(prev, cur, dt_s, params):
    """Cost of `cur` following `prev`, given how far they are along the rally cycle."""
    p = params
    if dt_s > p["new_point_s"]:
        return 0.0
    step = (RALLY_CYCLE.index(cur) - RALLY_CYCLE.index(prev)) % len(RALLY_CYCLE)
    return (p["repeated"], 0.0, p["missed_one"], p["missed_two"])[step]


def label_events(junctions, boxes, fps, serves=(), params=None):
    """Label the junctions with the best rally order; returns the events found.

    A serve contact forces a hit by the server. If the track shows no junction
    there, an event is added at the contact, to anchor the order that follows.
    """
    p = params or BOUNCE_PARAMS
    tol = p["serve_tol_s"] * fps
    nodes = []                      # (row or None, frame, allowed labels)
    served = set()
    for row in junctions.itertuples():
        labels = _allowed_labels(row, boxes, p)
        for contact, pid, _ in serves:
            if abs(row.frame - contact) <= tol:
                labels = [HIT_OF_PLAYER[pid]]
                served.add(contact)
        if labels:
            nodes.append((row, row.frame, labels))
    for contact, pid, _ in serves:
        if contact not in served:
            nodes.append((None, float(contact), [HIT_OF_PLAYER[pid]]))
    nodes.sort(key=lambda node: node[1])

    # Best chain of labelled junctions: a junction may be left out for free.
    score = [{} for _ in nodes]     # label -> (score, previous node, previous label)
    for k, (_, frame, labels) in enumerate(nodes):
        for label in labels:
            top, link = 0.0, None
            for q in range(k):
                dt_s = (frame - nodes[q][1]) / fps
                for prev_label, (s, _, _) in score[q].items():
                    value = s - _order_cost(prev_label, label, dt_s, p)
                    if value > top:
                        top, link = value, (q, prev_label)
            score[k][label] = (top + p["reward"],) + (link or (None, None))

    ends = [(s[0], k, label) for k, labels in enumerate(score) for label, s in labels.items()]
    events = []
    if ends:
        _, k, label = max(ends, key=lambda e: e[0])
        while k is not None:
            events.append((nodes[k], label))
            _, k, label = score[k][label]
    rows = [{**row._asdict(), "event": label}
            for (row, _, _), label in reversed(events) if row is not None]
    return pd.DataFrame(rows, columns=[*junctions.columns, "event"])


# ── bounces ────────────────────────────────────────────────────────────────────

def detect_events(ball, boxes, conv, fps, serves=(), params=None):
    """Hits and bounces read from the track, as a DataFrame sorted by time."""
    p = params or BOUNCE_PARAMS
    t = ball["frame"].to_numpy(float)
    pts = ball[["cx", "cy"]].to_numpy(float)
    pieces = segment_trajectory(t, pts, fps, p)
    return label_events(find_junctions(t, pts, pieces, conv, fps, p), boxes, fps, serves, p)


def clip_events(ball_csv, boxes, conv, fps, params=None):
    """detect_events on a clip's ball CSV, finding its serves first."""
    hits = detect_hits(load_ball_track(ball_track_for_shots(ball_csv)), boxes, fps)
    serves = detect_serves(boxes, conv, fps, hits)
    return detect_events(load_real_detections(ball_csv), boxes, conv, fps, serves, params)


def bounces_from_events(events, params=None):
    """One row per bounce: frame (with decimals), pixel and court position."""
    p = params or BOUNCE_PARAMS
    b = events[events["event"].isin((BOUNCE_FAR, BOUNCE_NEAR))].copy()
    b["side"] = np.where(b["event"] == BOUNCE_FAR, "far", "near")
    b["in_court"] = b["x_m"].between(0, W_m) & b["y_m"].between(0, L_m)
    b["reliable"] = b["fit_gap_px"] <= p["reliable_gap_px"]
    b = b.drop(columns="event").reset_index(drop=True)
    return b.round({"frame": 1, "cx": 1, "cy": 1, "x_m": 2, "y_m": 2,
                    "jump_up_ms": 1, "jump_side_ms": 1, "fit_gap_px": 1})


def detect_bounces(ball, boxes, conv, fps, serves=(), params=None):
    """The bounces of a track, see bounces_from_events."""
    return bounces_from_events(detect_events(ball, boxes, conv, fps, serves, params), params)


def save_bounce_map(bounces: pd.DataFrame, out_dir) -> None:
    """Plot the landing points on the court, reusing player_analysis's drawing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analysis.player_analysis import _draw_court

    fig, ax = plt.subplots(figsize=(5.2, 9.0))
    fig.patch.set_facecolor("#1a1a2e")
    _draw_court(ax)
    ax.set_facecolor("#1a1a2e")
    ax.patch.set_visible(True)
    ax.scatter(bounces["x_m"], bounces["y_m"], s=70, zorder=5, edgecolors="white",
               c=np.where(bounces["reliable"], "#f6c945", "#888888"))
    for r in bounces.itertuples():
        ax.annotate(f"{r.frame:.0f}", (r.x_m, r.y_m), xytext=(6, 4), color="white",
                    textcoords="offset points", fontsize=7, zorder=6)
    ax.set_title("Bounces (grey: uncertain landing point)", color="white")
    path = Path(out_dir) / "bounce_map.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find when and where the ball touches the ground")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="source video; only its name is used, for the default CSVs")
    parser.add_argument("--ball", default=None,
                        help="ball CSV (defaults to outputs/ball_coordinates/ball_<video name>.csv)")
    parser.add_argument("--players", default=None,
                        help="player tracking CSV (defaults to "
                             "outputs/player_coordinates/players_<video name>.csv)")
    parser.add_argument("--court", default=None,
                        help="court keypoints CSV (defaults to "
                             "outputs/court_coordinates/<video name>_court.csv)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default="outputs/shot_analysis")
    parser.add_argument("--no-map", action="store_true", dest="no_map",
                        help="do not save the bounce map")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    ball_csv = args.ball or os.path.join("outputs", "ball_coordinates", f"ball_{stem}.csv")
    players_csv = args.players or os.path.join("outputs", "player_coordinates",
                                               f"players_{stem}.csv")
    court_csv = args.court or os.path.join("outputs", "court_coordinates", f"{stem}_court.csv")
    if not os.path.exists(ball_csv):
        raise SystemExit(f"Ball CSV not found: {ball_csv}")

    boxes = load_player_boxes(players_csv)
    conv = CourtConverter(court_csv)
    events = clip_events(ball_csv, boxes, conv, args.fps)
    bounces = bounces_from_events(events)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "events.csv", index=False)
    path = out_dir / "bounces.csv"
    bounces.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(bounces)} bounces, "
          f"{int(bounces['reliable'].sum())} with a reliable landing point)")
    if not args.no_map and not bounces.empty:
        save_bounce_map(bounces, out_dir)


if __name__ == "__main__":
    main()
