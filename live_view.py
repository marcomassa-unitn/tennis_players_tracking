#!/usr/bin/env python3
"""Single live-view window for the tennis tracking pipeline.

Plays the ORIGINAL clip with the player + ball bounding boxes drawn on it and,
docked on the right, a top-down minimap that moves in lockstep with the video.
Below the minimap a placeholder panel is reserved for a statistics chart to be
added later (see ``render_stats_panel``).

Nothing is saved — this is a display-only viewer. It consumes the CSVs already
produced by the rest of the pipeline (it never re-runs a tracker or the YOLO
model), so it is fast and every overlay stays frame-synced with the data the
minimap is built from:

    player boxes  <- outputs/player_coordinates/players_<stem>.csv  (playerTracking)
    ball box      <- outputs/ball_coordinates/ball_<stem>.csv        (BallTracking; OPTIONAL)
    minimap       <- players CSV projected to metres through the court CSV
    court CSV     <- outputs/court_coordinates/<stem>_court.csv

The ball CSV is the model's own output (BallTracking._write_csv), so the ball
box is redrawn on the original frame from it — the pre-burned annotated video is
never needed.

Usage (from the project root):
    python live_view.py --video data/Input_video2.mp4

All three CSV paths default to the <stem>-based names next to ``--output`` and
can be overridden individually. Press ``q`` to close the window.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Project root on sys.path so utils.* / tracking.* resolve regardless of
# the current working directory (same pattern as pipeline.py).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# We reuse player_analysis only for its px->metre conversion (_load_and_convert)
# and the ITF court constants; the minimap itself is drawn natively with OpenCV.
# Importing it pins matplotlib to the headless "Agg" backend, which is harmless
# here (no figures/windows are ever created).
from utils import player_analysis
from utils.shot_analysis import (
    load_player_boxes,
    SHOT_CATEGORY_COLORS,   # hex per category — identical palette to shot_hitmap.png
    shot_category,          # (stroke, shot_type, overhead) -> single category string
)

# ── layout constants ────────────────────────────────────────────────────────────
PANEL_W   = 380          # width (px) of the right column (minimap + stats)
MINI_FRAC = 0.62         # fraction of the right column height given to the minimap
FONT      = cv2.FONT_HERSHEY_SIMPLEX

# Shot-type legend drawn INSIDE the minimap, in the free green run-off strip to
# the LEFT of the court (the court's left sideline is ~x=142 px in a 380 px panel,
# so x<~135 is clear). One row per entry: a symbol + the shot-type label.
_LEG_FS    = 0.38            # legend font scale
_LEG_ROW_H = 16              # row pitch (px)
_LEG_X     = 6               # left inset of the legend column (px)
_LEG_TOP   = 14              # baseline y of the first row (px)
_LEG_GAP   = 6              # gap between a symbol and its label (px)
# Stable category order so the legend never reshuffles as new shots accumulate.
_LEG_ORDER = ["forehand", "backhand", "slice", "dropshot",
              "lob", "serve", "smash", "unknown"]

# Player box colours on the VIDEO, matching playerTracking._draw_players
# (P1 green, P2 red). The minimap dots use player_analysis._COLORS (P1 red,
# P2 blue); the "P1"/"P2" labels on both views carry the identity.
_BOX_COLORS = {1: (0, 255, 0), 2: (0, 0, 255)}


# ── path helpers ────────────────────────────────────────────────────────────────

def _resolve(path: str) -> str:
    """Return ``path`` if it exists, else a case-insensitive sibling match.

    The repo's on-disk court CSV is ``input_video2_court.csv`` while the
    stem-derived default is ``Input_video2_court.csv``; this keeps the viewer
    working on case-sensitive filesystems too. Falls back to the original path
    (so the caller's "not found" message still makes sense).
    """
    if os.path.exists(path):
        return path
    folder, name = os.path.split(path)
    if folder and os.path.isdir(folder):
        for entry in os.listdir(folder):
            if entry.lower() == name.lower():
                return os.path.join(folder, entry)
    return path


def derive_default_paths(video: str, output: str) -> dict:
    """Stem-based default CSV paths, one per-modality subfolder under ``output``.

    Matches the producers' defaults: players_<stem>.csv in player_coordinates/,
    ball_<stem>.csv in ball_coordinates/, <stem>_court.csv in court_coordinates/.
    """
    stem = os.path.splitext(os.path.basename(video))[0]
    return {
        "players": os.path.join(output, "player_coordinates", f"players_{stem}.csv"),
        "ball":    os.path.join(output, "ball_coordinates", f"ball_{stem}.csv"),
        "court":   os.path.join(output, "court_coordinates", f"{stem}_court.csv"),
        # Shot analysis is per-output (not per-stem) — a single shots.csv.
        "shots":   os.path.join(output, "shot_analysis", "shots.csv"),
    }


# ── data loading ────────────────────────────────────────────────────────────────

def load_ball_boxes(ball_csv: str) -> dict:
    """{frame: (x, y, w, h)} from the ball CSV (frame,x,y,w,h,cx,cy,area)."""
    import pandas as pd
    df = pd.read_csv(ball_csv)
    return {int(r.frame): (int(r.x), int(r.y), int(r.w), int(r.h))
            for r in df.itertuples()}


def load_shot_markers(shots_csv: str, minimap: "Minimap",
                      bounds: tuple) -> list:
    """Pre-resolve shots.csv into render-ready markers, sorted by frame.

    Returns ``[(frame, pid, category, (px, py)), ...]`` where (px, py) are panel
    pixels (via ``minimap._to_px`` — the SAME transform the live dots use, so
    markers and dots share one coordinate system). The shot is placed at the
    PLAYER's feet position (``player_x_m``/``player_y_m``), matching
    shot_hitmap.png. Done once so the playback loop only filters by frame.

    Edge cases handled here: NaN player position -> skip; position outside the
    walkable bounds -> skip; several shots on one frame -> all kept.
    """
    import pandas as pd
    x_lo, x_hi, y_lo, y_hi = bounds
    df = pd.read_csv(shots_csv)
    out = []
    for r in df.itertuples():
        xm = getattr(r, "player_x_m", float("nan"))
        ym = getattr(r, "player_y_m", float("nan"))
        if (xm is None or ym is None
                or (isinstance(xm, float) and np.isnan(xm))
                or (isinstance(ym, float) and np.isnan(ym))):
            continue
        xm, ym = float(xm), float(ym)
        if not (x_lo <= xm <= x_hi and y_lo <= ym <= y_hi):
            continue
        cat = shot_category(getattr(r, "stroke", None),
                            getattr(r, "shot_type", None),
                            getattr(r, "overhead", ""))
        pid = int(getattr(r, "player_id", 0))
        out.append((int(r.frame), pid, cat, minimap._to_px(xm, ym)))
    out.sort(key=lambda t: t[0])
    return out


# ── minimap ─────────────────────────────────────────────────────────────────────

# Dot colours (BGR) — match the on-video player box colours (_BOX_COLORS:
# P1 green, P2 red) so the minimap reads coherently with the boxes.
_DOT_BGR = {1: (0, 255, 0), 2: (0, 0, 255)}


# ── shot markers (live, accumulating; colours match shot_hitmap.png) ─────────────

def _hex_to_bgr(hexstr: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (B, G, R) for OpenCV. Grey fallback on a malformed value."""
    s = str(hexstr).lstrip("#")
    if len(s) != 6:
        return (127, 127, 127)
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return (b, g, r)


# Category -> BGR, converted ONCE from the shared hex palette so the live shot
# markers match shot_hitmap.png exactly (single source of colour truth).
_CAT_BGR = {cat: _hex_to_bgr(hx) for cat, hx in SHOT_CATEGORY_COLORS.items()}
_CAT_FALLBACK_BGR = (127, 127, 127)   # grey, mirrors shot_analysis' category fallback

# Shot marker geometry. The player dot is a 6 px FILLED circle; the shot marker is
# only ~1 px larger and HOLLOW, so it frames the spot without dwarfing the dot.
_SHOT_MARKER_R = 7    # hollow shot-marker "radius" (px)
_SHOT_MARKER_TH = 2   # hollow stroke thickness (px)
# sqrt(3)/2 — half-base offset of an equilateral up-triangle of circumradius r.
_TRI_H = 0.866


def _draw_shot_marker(img, center, pid, color_bgr,
                      r: int = _SHOT_MARKER_R, th: int = _SHOT_MARKER_TH) -> None:
    """Hollow, category-coloured marker; the SHAPE encodes the player
    (P1 = circle, P2 = up-triangle — same convention as shot_hitmap.png).

    A near-black ring 1 px wider is stroked first so the marker stays legible on
    the green court and over the white court lines.
    """
    px, py = int(round(center[0])), int(round(center[1]))
    outline = (20, 20, 20)
    if pid == 2:
        pts = np.array([[px,             py - r],
                        [px - _TRI_H * r, py + 0.5 * r],
                        [px + _TRI_H * r, py + 0.5 * r]], dtype=np.int32)
        cv2.polylines(img, [pts], True, outline, th + 1, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color_bgr, th, cv2.LINE_AA)
    else:   # P1 and any unknown player id -> hollow circle
        cv2.circle(img, (px, py), r + 1, outline, th, cv2.LINE_AA)
        cv2.circle(img, (px, py), r, color_bgr, th, cv2.LINE_AA)


class Minimap:
    """Fast top-down court minimap drawn natively with OpenCV.

    The court (ITF singles dimensions + walkable run-off, reusing the constants
    from player_analysis) is rasterised ONCE into a base image; each frame only
    copies that base and stamps the player dots. This is rendered live in the
    playback loop: O(1) memory and well under a millisecond per frame, so it
    scales to long clips (unlike pre-rendering every frame with matplotlib).
    """

    _MARGIN = 10   # px padding around the court inside the panel

    def __init__(self, size: tuple[int, int]):
        self.w, self.h = size
        pa = player_analysis
        # Walkable-area bounds in metres (court + run-off behind/beside).
        self._x0, self._x1 = -pa._SIDE, pa.W_m + pa._SIDE
        self._y0, self._y1 = -pa._BEHIND, pa.L_m + pa._BEHIND
        world_w = self._x1 - self._x0
        world_h = self._y1 - self._y0
        # Aspect-correct fit, then centre the court in the panel.
        usable_w, usable_h = self.w - 2 * self._MARGIN, self.h - 2 * self._MARGIN
        self._scale = min(usable_w / world_w, usable_h / world_h)
        self._ox = (self.w - world_w * self._scale) / 2.0
        self._oy = (self.h - world_h * self._scale) / 2.0
        self._base = self._draw_base()

    def _to_px(self, xm: float, ym: float) -> tuple[int, int]:
        """Metres -> panel pixels. y=0 (far baseline) maps near the top, as seen
        from the camera; y grows downward toward the near baseline."""
        px = self._ox + (xm - self._x0) * self._scale
        py = self._oy + (ym - self._y0) * self._scale
        return int(round(px)), int(round(py))

    def _line_m(self, img, p0, p1, color, thickness=1):
        cv2.line(img, self._to_px(*p0), self._to_px(*p1), color, thickness,
                 cv2.LINE_AA)

    def _draw_base(self) -> np.ndarray:
        pa = player_analysis
        img = np.full((self.h, self.w, 3), (78, 106, 45), np.uint8)  # court green (BGR)
        white = (255, 255, 255)
        # perimeter (singles court)
        self._line_m(img, (0, 0),      (pa.W_m, 0),      white)
        self._line_m(img, (0, pa.L_m), (pa.W_m, pa.L_m), white)
        self._line_m(img, (0, 0),      (0, pa.L_m),      white)
        self._line_m(img, (pa.W_m, 0), (pa.W_m, pa.L_m), white)
        # service lines + centre service line
        self._line_m(img, (0, pa.SVC_T),    (pa.W_m, pa.SVC_T), white)
        self._line_m(img, (0, pa.SVC_B),    (pa.W_m, pa.SVC_B), white)
        self._line_m(img, (pa.CL_X, pa.SVC_T), (pa.CL_X, pa.SVC_B), white)
        # net (yellow, thicker)
        self._line_m(img, (0, pa.NET), (pa.W_m, pa.NET), (60, 200, 240), 2)
        # (The shot-type + player legend is stamped per-frame in render(), inside
        #  the free green strip left of the court, so it can grow as shots occur.)
        return img

    def _draw_legend(self, img, present_cats) -> None:
        """Stamp the shot legend INSIDE the minimap, in the free green run-off
        strip left of the court. One row per shot category that has occurred so
        far (hollow colour symbol + label in that colour), in the stable
        _LEG_ORDER, then a P1=circle / P2=triangle shape key. Drawn over a faint
        dark backdrop so the coloured text reads against the green court.
        """
        rows = [c for c in _LEG_ORDER if c in present_cats]
        n = len(rows) + 2 + 1   # categories + 2 shape-key rows + 1 header row
        # Backdrop sized to the column; clipped to the panel just in case.
        x0, y0 = _LEG_X - 3, _LEG_TOP - 12
        x1 = x0 + 96
        y1 = y0 + n * _LEG_ROW_H + 4
        x1, y1 = min(x1, self.w - 1), min(y1, self.h - 1)
        ov = img[y0:y1, x0:x1].copy()
        img[y0:y1, x0:x1] = cv2.addWeighted(
            ov, 0.45, np.zeros_like(ov), 0.0, 0.0)   # darken 55%

        sym_r = 5   # legend symbol radius (compact, < the 7 px live markers)
        y = _LEG_TOP
        cv2.putText(img, "Shots", (_LEG_X, y), FONT, _LEG_FS,
                    (235, 235, 235), 1, cv2.LINE_AA)
        y += _LEG_ROW_H
        for cat in rows:
            color = _CAT_BGR.get(cat, _CAT_FALLBACK_BGR)
            cv2.circle(img, (_LEG_X + sym_r, y - 4), sym_r, color, 2, cv2.LINE_AA)
            cv2.putText(img, cat, (_LEG_X + 2 * sym_r + _LEG_GAP, y),
                        FONT, _LEG_FS, color, 1, cv2.LINE_AA)
            y += _LEG_ROW_H
        # player shape key (neutral white so only the shape reads)
        for pid, lbl in ((1, "P1"), (2, "P2")):
            _draw_shot_marker(img, (_LEG_X + sym_r, y - 4), pid,
                              (235, 235, 235), r=sym_r, th=1)
            cv2.putText(img, lbl, (_LEG_X + 2 * sym_r + _LEG_GAP, y),
                        FONT, _LEG_FS, (235, 235, 235), 1, cv2.LINE_AA)
            y += _LEG_ROW_H

    def render(self, positions: dict, shot_markers=None,
               present_cats=None) -> np.ndarray:
        """BGR minimap. ``positions`` = {pid: (xm, ym)} for the live player dots;
        ``shot_markers`` = [(pid, category, (px, py)), ...] already crossed during
        playback (accumulated); ``present_cats`` = set of categories seen so far,
        which drives the in-minimap legend (drawn in the free strip left of court).

        Shot markers are stamped FIRST (a static "history" layer) and the live
        player dots LAST (the moving "present" layer), so a dot is never hidden by
        an accumulated marker; the markers are hollow, so an overlapping dot still
        shows the coloured ring around it. The legend is stamped last so it stays
        readable over everything.
        """
        img = self._base.copy()
        if shot_markers:
            for pid, cat, center in shot_markers:
                _draw_shot_marker(img, center, pid,
                                  _CAT_BGR.get(cat, _CAT_FALLBACK_BGR))
        for pid, (xm, ym) in positions.items():
            cv2.circle(img, self._to_px(xm, ym), 6,
                       _DOT_BGR.get(pid, (255, 255, 255)), -1, cv2.LINE_AA)
        if present_cats:
            self._draw_legend(img, present_cats)
        return img


# ── stats placeholder ───────────────────────────────────────────────────────────

def render_stats_panel(frame_idx: int, size: tuple[int, int]) -> np.ndarray:
    """Placeholder for the future statistics chart (minimap sits above it).

    Receives the current ``frame_idx`` already, so a moving cursor / live
    readout can be wired in here later without touching the playback loop.
    """
    w, h = size
    panel = np.full((h, w, 3), (40, 30, 25), np.uint8)
    cv2.putText(panel, "STATS", (12, 30), FONT, 0.8, (200, 200, 200), 2)
    cv2.putText(panel, "(chart coming soon)", (12, 58), FONT, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, f"frame {frame_idx}", (12, h - 14), FONT, 0.5, (120, 120, 120), 1)
    return panel




# ── compositing ─────────────────────────────────────────────────────────────────

def compose(frame: np.ndarray, minimap: np.ndarray, stats: np.ndarray,
            disp_h: int, panel_w: int, mini_h: int) -> np.ndarray:
    """Assemble [ video | (minimap / stats) ] into one canvas of height disp_h.

    The shot legend now lives INSIDE the minimap (see Minimap._draw_legend), so
    there is no separate strip and the canvas height equals disp_h exactly.
    """
    h0, w0 = frame.shape[:2]
    vw = int(round(w0 * (disp_h / h0)))
    video = cv2.resize(frame, (vw, disp_h))

    canvas = np.zeros((disp_h, vw + panel_w, 3), np.uint8)
    canvas[:, :vw] = video
    # Resize panels to their exact sub-regions so assembly never shape-mismatches.
    canvas[0:mini_h, vw:vw + panel_w]      = cv2.resize(minimap, (panel_w, mini_h))
    canvas[mini_h:disp_h, vw:vw + panel_w] = cv2.resize(stats, (panel_w, disp_h - mini_h))
    return canvas


# ── playback ────────────────────────────────────────────────────────────────────

def run(video, players_csv, court_csv, ball_csv, min_area, anchor,
        disp_h, fps_override, shots_csv=None, max_shot_markers=0) -> None:
    # --- load per-frame overlays (indexed by absolute frame number) ---
    boxes = load_player_boxes(players_csv)
    print(f"  Player boxes: {len(boxes)} frames from {players_csv}")

    if ball_csv and os.path.exists(ball_csv):
        ball_boxes = load_ball_boxes(ball_csv)
        print(f"  Ball boxes  : {len(ball_boxes)} frames from {ball_csv}")
    else:
        ball_boxes = {}
        print(f"  Ball boxes  : none (no ball CSV at {ball_csv}); ball overlay "
              "disabled.\n               Run 'python tracking/BallTracking.py' "
              "to enable it.")

    # --- minimap: project players to metres, indexed by frame for O(1) lookup ---
    player_data = player_analysis._load_and_convert(
        players_csv, court_csv, min_area, anchor)
    indexed = {pid: sub.set_index("frame") for pid, sub in player_data.items()}
    # Walkable-area bounds, same guard as player_analysis._save_minimap.
    x_lo, x_hi = -player_analysis._SIDE - 1, player_analysis.W_m + player_analysis._SIDE + 1
    y_lo, y_hi = -player_analysis._BEHIND - 1, player_analysis.L_m + player_analysis._BEHIND + 1
    mini_h = int(round(disp_h * MINI_FRAC))
    minimap = Minimap((PANEL_W, mini_h))

    # --- shots: pre-resolve to render-ready markers (optional overlay) ---
    shot_markers = []
    if shots_csv and os.path.exists(shots_csv):
        try:
            shot_markers = load_shot_markers(
                shots_csv, minimap, (x_lo, x_hi, y_lo, y_hi))
            print(f"  Shot markers: {len(shot_markers)} shots from {shots_csv}")
        except Exception as e:   # malformed/empty CSV must never break playback
            print(f"  Shot markers: failed to load ({e}); continuing without them.")
            shot_markers = []
    else:
        print(f"  Shot markers: none (no shots CSV at {shots_csv}); shot overlay "
              "disabled.\n               Run 'python utils/shot_analysis.py' to "
              "enable it.")

    # FIXED legend: the full set of categories present in shots.csv, computed once
    # so the key is shown in full from the first frame (it does NOT build up as
    # shots occur). Empty when there are no shots -> no legend drawn.
    legend_cats = {cat for _f, _pid, cat, _c in shot_markers} or None

    def minimap_positions(frame_idx: int) -> dict:
        """{pid: (xm, ym)} for the players present (and in range) at frame_idx."""
        out = {}
        for pid, idf in indexed.items():
            if frame_idx in idf.index:
                row = idf.loc[frame_idx]
                xm, ym = float(row["x_m"]), float(row["y_m"])
                if x_lo <= xm <= x_hi and y_lo <= ym <= y_hi:
                    out[pid] = (xm, ym)
        return out

    # --- video playback ---
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_period = 0.915 / fps   # real seconds-per-frame of the source video

    win = "Tennis Live View  (q to quit)"
    print(f"  Playing at {fps:.0f} fps — press 'q' in the window to quit.")
    idx = 0
    # Shot accumulation: shot_markers is sorted by frame, so a single advancing
    # pointer reveals each marker once its frame is reached and keeps it shown
    # for the rest of the clip (O(total shots) overall, not per frame).
    active_markers = []     # [(pid, cat, center), ...] — frames already crossed
    next_shot = 0
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            # Player boxes (absolute frame lookup; missing -> nothing drawn).
            for pid, box in boxes.get(idx, {}).items():
                x, y, w, h = (int(v) for v in box)
                color = _BOX_COLORS.get(pid, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"P{pid}", (x, max(12, y - 8)),
                            FONT, 0.7, color, 2)

            # Ball box (yellow).
            ball = ball_boxes.get(idx)
            if ball is not None:
                x, y, w, h = ball
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Ball", (x, max(12, y - 8)),
                            FONT, 0.5, (0, 255, 255), 2)

            # Reveal every shot whose frame has been reached; it then persists.
            while next_shot < len(shot_markers) and shot_markers[next_shot][0] <= idx:
                _f, _pid, _cat, _center = shot_markers[next_shot]
                active_markers.append((_pid, _cat, _center))
                next_shot += 1
            # Optional clutter cap: keep only the most recent N markers (0 = all).
            shown = (active_markers[-max_shot_markers:]
                     if max_shot_markers else active_markers)

            # Legend is FIXED (full key from the first frame), not built up live.
            mini = minimap.render(minimap_positions(idx), shown, legend_cats)
            stats = render_stats_panel(idx, (PANEL_W, disp_h - mini_h))
            cv2.imshow(win, compose(frame, mini, stats, disp_h, PANEL_W, mini_h))

            # Wait only the time LEFT in this frame's period after the per-frame
            # processing, so render overhead doesn't slow playback below real time.
            wait_ms = max(1, int(round((frame_period - (time.perf_counter() - t0)) * 1000)))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
            idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live viewer: video + player/ball boxes + synced minimap "
                    "(+ reserved stats panel).")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="input video (default: data/Input_video2.mp4)")
    parser.add_argument("--output", default="outputs",
                        help="base output dir used to derive default CSV paths")
    parser.add_argument("--players", default=None,
                        help="player tracking CSV "
                             "(default: outputs/player_coordinates/players_<stem>.csv)")
    parser.add_argument("--ball", default=None,
                        help="ball CSV "
                             "(default: outputs/ball_coordinates/ball_<stem>.csv; optional)")
    parser.add_argument("--court", default=None,
                        help="court keypoints CSV "
                             "(default: outputs/court_coordinates/<stem>_court.csv)")
    parser.add_argument("--shots", default=None,
                        help="shot analysis CSV (default: "
                             "outputs/shot_analysis/shots.csv; optional — when "
                             "present, shots accumulate on the minimap)")
    parser.add_argument("--max-shot-markers", type=int, default=0,
                        dest="max_shot_markers",
                        help="cap simultaneously shown shot markers "
                             "(0 = unlimited, the default accumulate behaviour)")
    parser.add_argument("--min-area", type=int, default=500, dest="min_area",
                        help="drop player detections below this area for the "
                             "minimap (default: 500)")
    parser.add_argument("--anchor", choices=["feet", "centroid"], default="feet",
                        help="point projected to the court for the minimap "
                             "(default: feet)")
    parser.add_argument("--height", type=int, default=720,
                        help="display height in px of the composited window "
                             "(default: 720)")
    parser.add_argument("--fps", type=float, default=None,
                        help="playback fps (default: read from the video)")
    args = parser.parse_args()

    defaults = derive_default_paths(args.video, args.output)
    players_csv = _resolve(args.players or defaults["players"])
    court_csv   = _resolve(args.court   or defaults["court"])
    ball_csv    = _resolve(args.ball    or defaults["ball"])
    shots_csv   = _resolve(args.shots   or defaults["shots"])

    # Players + court are required; the ball CSV is optional (see run()).
    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")
    if not os.path.exists(players_csv):
        raise SystemExit(
            f"Players CSV not found: {players_csv}\n"
            "Run: python tracking/playerTracking.py --video "
            f"{args.video} --csv {players_csv} --no-display")
    if not os.path.exists(court_csv):
        raise SystemExit(
            f"Court CSV not found: {court_csv}\n"
            "Run: python tracking/court_tracking.py --video "
            f"{args.video} --no-display")

    print("Tennis live view")
    print(f"  video  : {args.video}")
    print(f"  players: {players_csv}")
    print(f"  court  : {court_csv}")
    run(args.video, players_csv, court_csv, ball_csv,
        args.min_area, args.anchor, args.height, args.fps,
        shots_csv, args.max_shot_markers)


if __name__ == "__main__":
    main()
