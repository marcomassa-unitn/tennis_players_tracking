"""
utils/player_analysis.py

Converts the players' pixel positions into real-world coordinates (meters)
through CourtConverter and generates:
  - minimap.gif           top-down animation of the court
  - heatmap_p1/p2.png     density map per player
  - heatmap_combined.png  both overlaid
  - speed_stats.csv       per-frame speed in km/h
  - zone_stats.csv        time spent in each zone of the court
  - zones.png             zone occupancy drawn on the court
  + statistics summary in the terminal

The point projected to the ground is by default the feet point (bottom-center
of the bounding box): the body centroid is ~1 m above the ground and the
homography would project it much farther from the net than it really is.

Quick use (from project root):
    python utils/player_analysis.py

Full use:
    python utils/player_analysis.py \\
        --players outputs/player_coordinates/players_Input_video2.csv \\
        --court   outputs/court_coordinates/Input_video2_court.csv \\
        --fps     30 \\
        --output  outputs/player_analysis \\
        --stride  3
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, does not open windows
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter

# ── ITF singles court dimensions ───────────────────────────────────────────────
_FT   = 0.3048
W_m   = 27.0 * _FT          # 8.2296 m  width (singles)
L_m   = 78.0 * _FT          # 23.7744 m length
SVC_T = 18.0 * _FT          # 5.4864 m  service line far side
SVC_B = L_m - SVC_T         # 18.288 m  service line near side
NET   = L_m / 2.0            # 11.8872 m net
CL_X  = W_m / 2.0            # 4.1148 m  center service line

# Player color convention — kept CONSISTENT across every figure (minimap,
# zones, single heatmaps, combined heatmap legend) so a reader can always map
# colour -> player reliably:  P1 = warm (red),  P2 = cool (blue).
# The solid colours below are paired with the warm/cool colormaps used for the
# heatmaps (_HEAT_CMAPS) and the matching legend patches (_HEAT_LEGEND).
_COLORS = {1: "red",       2: "steelblue"}
_LABELS = {1: "P1 (red)",  2: "P2 (blue)"}
# heatmap colormaps + legend patch colours, same warm/cool mapping as _COLORS
_HEAT_CMAPS  = {1: "hot",        2: "Blues_r"}
_HEAT_LEGEND = {1: "red",        2: "steelblue"}

# ── court zones ────────────────────────────────────────────────────────────────
# The analysis covers the whole walkable area, not just the playing rectangle:
# _BEHIND = typical run-off behind each baseline, _SIDE = lateral run-off
# (ITF recommendation ~3.66 m) beyond each sideline.
_BEHIND = 6.5
_SIDE   = 3.7
_ZONE_BANDS = [
    ("FB-out", "Behind baseline (far)",    -_BEHIND, 0.0),
    ("FB",     "Backcourt (far)",           0.0,     SVC_T),
    ("FF",     "Service zone (far)",        SVC_T,   NET),
    ("NF",     "Service zone (near)",       NET,     SVC_B),
    ("NB",     "Backcourt (near)",          SVC_B,   L_m),
    ("NB-out", "Behind baseline (near)",    L_m,     L_m + _BEHIND),
]
_SIDES = [("OL", "outer left"), ("L", "left"), ("R", "right"), ("OR", "outer right")]
_SIDE_X = {"OL": (-_SIDE, 0.0), "L": (0.0, CL_X),
           "R": (CL_X, W_m), "OR": (W_m, W_m + _SIDE)}


# ── court drawing ──────────────────────────────────────────────────────────────

def _draw_court(ax: plt.Axes) -> None:
    """Draw the ITF singles court on ax (coordinates in meters)."""
    ax.set_facecolor("#2d6a4f")
    kw = dict(color="white", linewidth=1.4, solid_capstyle="round")

    # perimeter
    ax.plot([0, W_m], [0,   0  ], **kw)
    ax.plot([0, W_m], [L_m, L_m], **kw)
    ax.plot([0, 0  ], [0,   L_m], **kw)
    ax.plot([W_m, W_m], [0, L_m], **kw)
    # service lines
    ax.plot([0, W_m], [SVC_T, SVC_T], **kw)
    ax.plot([0, W_m], [SVC_B, SVC_B], **kw)
    ax.plot([CL_X, CL_X], [SVC_T, SVC_B], **kw)
    # net
    ax.plot([0, W_m], [NET, NET], color="#f4d03f", linewidth=2.8)

    # limits extended to the whole walkable area (lateral and baseline run-off);
    # y inverted: far baseline (y=0) at top, as in the camera view
    ax.set_xlim(-_SIDE - 0.4, W_m + _SIDE + 0.4)
    ax.set_ylim(L_m + _BEHIND + 0.4, -_BEHIND - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")


# ── loading and conversion ─────────────────────────────────────────────────────

def _load_and_convert(players_csv: str, court_csv: str, min_area: int,
                      anchor: str = "feet") -> dict[int, pd.DataFrame]:
    """
    Load the players CSV, filter out detections with area < min_area and
    add the columns x_m, y_m with the real-world coordinates in meters.

    anchor = "feet"     → projects the feet point (cx, y + h)  [default]
    anchor = "centroid" → projects the blob centroid (cx, cy)
    Returns {player_id: DataFrame}.
    """
    df = pd.read_csv(players_csv)
    df = df[df["area"] >= min_area].copy()

    if anchor == "feet":
        # Sanity check (non-fatal): the feet anchor assumes "y" is the bbox TOP,
        # so the bbox vertical centre cy should sit ~ y + h/2. If a large share
        # of rows violate that, "y" is probably not the top edge and the feet
        # projection (y + h) would be wrong — warn but keep running.
        if len(df) and {"y", "h", "cy"}.issubset(df.columns):
            mismatch = np.abs((df["y"] + df["h"] / 2.0) - df["cy"]) > df["h"] * 0.25
            frac_bad = float(mismatch.mean())
            if frac_bad > 0.5:
                print(f"  Warning: cy != y + h/2 for {frac_bad:.0%} of rows; "
                      f"'y' may not be the bbox top — feet anchor (y + h) "
                      f"could be misplaced.")
        df["ax"] = df["cx"]
        df["ay"] = df["y"] + df["h"]
    elif anchor == "centroid":
        df["ax"] = df["cx"]
        df["ay"] = df["cy"]
    else:
        raise ValueError(f"unknown anchor: {anchor!r}")

    conv = CourtConverter(court_csv)
    player_data: dict[int, pd.DataFrame] = {}
    for pid in sorted(df["player_id"].unique()):
        sub = df[df["player_id"] == pid].sort_values("frame").copy()
        pts = conv.to_meters_batch(sub[["ax", "ay"]].values)
        sub["x_m"] = pts[:, 0]
        sub["y_m"] = pts[:, 1]
        player_data[int(pid)] = sub.reset_index(drop=True)
    return player_data


# ── speed statistics ───────────────────────────────────────────────────────────

def _compute_speeds(sub: pd.DataFrame, fps: float,
                    max_speed_kmh: float = 45.0) -> tuple[dict, np.ndarray]:
    """
    Compute per-frame speed (km/h) and aggregate statistics.
    Speed = NaN for non-consecutive frames (player not detected).
    Displacements above max_speed_kmh are tracking glitches (blob jumps,
    ID swaps): they are excluded from both speed and distance.
    """
    frames = sub["frame"].values
    pos    = sub[["x_m", "y_m"]].values

    dxy  = np.diff(pos, axis=0)
    dist = np.linalg.norm(dxy, axis=1)
    gaps = np.diff(frames)
    dt   = gaps / fps

    with np.errstate(divide="ignore", invalid="ignore"):
        speed_kmh = np.where(gaps == 1, dist / dt * 3.6, np.nan)
    speed_kmh = np.where(speed_kmh > max_speed_kmh, np.nan, speed_kmh)

    # series aligned with sub (first value always NaN)
    speed_full = np.concatenate([[np.nan], speed_kmh])

    valid_dist = dist[~np.isnan(speed_kmh)]
    # If there is no valid speed sample at all, every aggregate (avg/median as
    # well as max/p95) must fall back to 0.0 instead of letting the np.nan*
    # reductions emit a RuntimeWarning and return NaN.
    has_speed = not np.all(np.isnan(speed_kmh))
    stats = {
        "total_distance_m": round(float(valid_dist.sum()), 2),
        "avg_speed_kmh":    round(float(np.nanmean(speed_kmh)), 2)
                            if has_speed else 0.0,
        "median_speed_kmh": round(float(np.nanmedian(speed_kmh)), 2)
                            if has_speed else 0.0,
        "max_speed_kmh":    round(float(np.nanmax(speed_kmh)), 2)
                            if has_speed else 0.0,
        "p95_speed_kmh":    round(float(np.nanpercentile(speed_kmh, 95)), 2)
                            if has_speed else 0.0,
        "frames_detected":  len(sub),
        # Count the actual number of dropped frames, not the number of gap
        # events: a gap of g consecutive missing frames contributes (g - 1).
        "frames_missing":   int(np.sum(gaps[gaps > 1] - 1)),
    }
    return stats, speed_full


# ── court zones ────────────────────────────────────────────────────────────────

def _zone_of(x_m: float, y_m: float) -> str | None:
    """Return the zone id ("FB-L", "NB-out-OR", …) or None if out of range."""
    if x_m < -_SIDE - 1.0 or x_m > W_m + _SIDE + 1.0:
        return None
    # Boundary convention: every band is half-open [low, high) on BOTH axes, so
    # a point exactly on a dividing line always falls to the band on its higher
    # side (e.g. x == CL_X -> "R", x == W_m -> "OR"). This matches the y-band
    # test below (y0 <= y_m < y1) and removes the previous L/R vs R/OR
    # asymmetry where the centre line was exclusive but the right sideline
    # inclusive.
    if x_m < 0:
        side = "OL"
    elif x_m < CL_X:
        side = "L"
    elif x_m < W_m:
        side = "R"
    else:
        side = "OR"
    for zid, _, y0, y1 in _ZONE_BANDS:
        if y0 <= y_m < y1:
            return f"{zid}-{side}"
    return None


def _compute_zone_stats(player_data: dict, fps: float) -> pd.DataFrame:
    """
    Time spent by each player in each zone of the court.
    One row per (player_id, zone): frames, seconds and percentage.
    """
    cols = ["player_id", "zone", "description", "frames", "seconds", "percent"]
    rows = []
    band_labels = {zid: label for zid, label, _, _ in _ZONE_BANDS}
    for pid, sub in player_data.items():
        zones = [_zone_of(x, y) for x, y in sub[["x_m", "y_m"]].values]
        zones = pd.Series([z for z in zones if z is not None])
        n_tot = len(zones)
        if n_tot == 0:
            continue
        counts = zones.value_counts()
        for zone_id, n in counts.items():
            band, side = zone_id.rsplit("-", 1)
            side_label = dict(_SIDES)[side]
            rows.append({
                "player_id":  pid,
                "zone":       zone_id,
                "description": f"{band_labels[band]} {side_label}",
                "frames":     int(n),
                "seconds":    round(n / fps, 2),
                # n_tot is guaranteed > 0 here (the continue above skips
                # players with no in-range samples), so this division is safe.
                "percent":    round(100.0 * n / n_tot, 2),
            })
    if not rows:
        # No in-range samples for any player: return an empty frame that still
        # carries the expected columns so downstream .sort_values / filtering /
        # to_csv keep working instead of raising on a column-less frame.
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(
        ["player_id", "percent"], ascending=[True, False]
    ).reset_index(drop=True)


def _save_zone_outputs(player_data: dict, zone_df: pd.DataFrame,
                       out_dir: Path) -> None:
    """Save zone_stats.csv and the figure zones.png (percentages on the court)."""
    csv_path = out_dir / "zone_stats.csv"
    zone_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    pids = sorted(player_data.keys())
    fig, axes = plt.subplots(1, len(pids), figsize=(4.2 * len(pids), 9))
    fig.patch.set_facecolor("#1a1a2e")
    if len(pids) == 1:
        axes = [axes]

    for ax, pid in zip(axes, pids):
        _draw_court(ax)
        ax.set_ylim(L_m + _BEHIND + 0.4, -_BEHIND - 0.4)
        sub_df = zone_df[zone_df["player_id"] == pid]
        pct = dict(zip(sub_df["zone"], sub_df["percent"]))
        max_pct = max(pct.values()) if pct else 1.0

        for zid, _, y0, y1 in _ZONE_BANDS:
            for side, _ in _SIDES:
                p = pct.get(f"{zid}-{side}", 0.0)
                x0, x1 = _SIDE_X[side]
                alpha = 0.75 * p / max_pct
                outside = zid.endswith("-out") or side in ("OL", "OR")
                ax.add_patch(plt.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=_COLORS[pid], alpha=alpha,
                    edgecolor="white", linewidth=0.4,
                    linestyle=":" if outside else "-",
                ))
                if p > 0.05:
                    ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{p:.1f}%",
                            ha="center", va="center", color="white",
                            fontsize=7, fontweight="bold")
        ax.set_title(f"Zones – Player {pid}", color="white",
                     fontsize=11, pad=4)

    path = out_dir / "zones.png"
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ── heatmap ────────────────────────────────────────────────────────────────────

# heatmap extent: the whole walkable area
_HEAT_X = (-_SIDE, W_m + _SIDE)
_HEAT_Y = (-_BEHIND, L_m + _BEHIND)
_HEAT_EXTENT = [*_HEAT_X, *_HEAT_Y]


def _make_heat_array(x_m: np.ndarray, y_m: np.ndarray,
                     bins=(70, 165), sigma=2.5) -> np.ndarray:
    xc = np.clip(x_m, *_HEAT_X)
    yc = np.clip(y_m, *_HEAT_Y)
    H, _, _ = np.histogram2d(xc, yc, bins=bins,
                             range=[_HEAT_X, _HEAT_Y])
    H = gaussian_filter(H.T.astype(float), sigma=sigma)
    H /= H.max() + 1e-10
    return H


def _save_heatmaps(player_data: dict, out_dir: Path) -> None:
    # same warm/cool convention as _COLORS (P1 warm, P2 cool)
    cmaps = _HEAT_CMAPS

    for pid, sub in player_data.items():
        fig, ax = plt.subplots(figsize=(4, 9))
        fig.patch.set_facecolor("#1a1a2e")
        _draw_court(ax)
        H = _make_heat_array(sub["x_m"].values, sub["y_m"].values)
        ax.imshow(H, origin="lower", extent=_HEAT_EXTENT,
                  cmap=cmaps.get(pid, "hot"), alpha=0.70, aspect="auto")
        ax.set_title(f"Heatmap – Player {pid}", color="white",
                     fontsize=11, pad=4)
        path = out_dir / f"heatmap_p{pid}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: {path}")

    # combined heatmap
    fig, ax = plt.subplots(figsize=(4, 9))
    fig.patch.set_facecolor("#1a1a2e")
    _draw_court(ax)
    cmap_alpha = {1: (_HEAT_CMAPS[1], 0.55), 2: (_HEAT_CMAPS[2], 0.55)}
    for pid, (cmap, alpha) in cmap_alpha.items():
        if pid not in player_data:
            continue
        H = _make_heat_array(player_data[pid]["x_m"].values,
                             player_data[pid]["y_m"].values)
        ax.imshow(H, origin="lower", extent=_HEAT_EXTENT,
                  cmap=cmap, alpha=alpha, aspect="auto")
    # legend colours match the warm/cool player convention (_HEAT_LEGEND)
    patches = [mpatches.Patch(color=_HEAT_LEGEND[1], label="Player 1"),
               mpatches.Patch(color=_HEAT_LEGEND[2], label="Player 2")]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              framealpha=0.5, facecolor="#333", edgecolor="none",
              labelcolor="white")
    ax.set_title("Combined heatmap", color="white", fontsize=11, pad=4)
    path = out_dir / "heatmap_combined.png"
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ── animated minimap ───────────────────────────────────────────────────────────

def _save_minimap(player_data: dict, all_frames: np.ndarray,
                  fps: float, stride: int, out_dir: Path) -> None:
    # fast per-frame indexing
    indexed = {pid: sub.set_index("frame") for pid, sub in player_data.items()}

    sampled  = all_frames[::stride]
    gif_fps  = max(1, int(fps // stride))

    fig, ax = plt.subplots(figsize=(3.5, 8))
    fig.patch.set_facecolor("#1a1a2e")
    _draw_court(ax)

    dots = {
        pid: ax.plot([], [], "o", color=_COLORS[pid], ms=10,
                     label=_LABELS[pid], zorder=5)[0]
        for pid in player_data
    }
    title = ax.text(
        W_m / 2, -_BEHIND - 0.8, "",
        ha="center", va="bottom", color="white", fontsize=7,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.45,
              facecolor="#222", edgecolor="none", labelcolor="white")

    def _update(fid):
        title.set_text(f"frame {fid}")
        for pid, dot in dots.items():
            if fid in indexed[pid].index:
                row = indexed[pid].loc[fid]
                xm, ym = float(row["x_m"]), float(row["y_m"])
                # also show positions outside the court (walkable area)
                if (-_SIDE - 1 <= xm <= W_m + _SIDE + 1
                        and -_BEHIND - 1 <= ym <= L_m + _BEHIND + 1):
                    dot.set_data([xm], [ym])
                    continue
            dot.set_data([], [])
        return list(dots.values()) + [title]

    ani = FuncAnimation(fig, _update, frames=sampled, blit=True,
                        interval=1000 // gif_fps)
    path = out_dir / "minimap.gif"
    print(f"  Generating GIF ({len(sampled)} frames, {gif_fps} fps) …")
    ani.save(str(path), writer=PillowWriter(fps=gif_fps), dpi=90)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── speed CSV ──────────────────────────────────────────────────────────────────

def _save_speed_csv(player_data: dict, speed_map: dict, out_dir: Path) -> None:
    parts = []
    for pid, sub in player_data.items():
        tmp = sub[["frame", "x_m", "y_m"]].copy()
        tmp["player_id"] = pid
        tmp["speed_kmh"] = speed_map[pid]
        parts.append(tmp)
    out = pd.concat(parts).sort_values(["frame", "player_id"])
    path = out_dir / "speed_stats.csv"
    out.to_csv(path, index=False, float_format="%.4f")
    print(f"  Saved: {path}")


# ── terminal summary ───────────────────────────────────────────────────────────

def _print_summary(stats_map: dict, zone_df: pd.DataFrame,
                   total_frames: int) -> None:
    print("\n" + "=" * 52)
    print("  PLAYER ANALYSIS  –  SUMMARY")
    print("=" * 52)
    for pid, s in stats_map.items():
        print(f"\nPlayer {pid} ({_COLORS[pid]}):")
        print(f"  Total distance   : {s['total_distance_m']:.1f} m")
        print(f"  Average speed    : {s['avg_speed_kmh']:.1f} km/h")
        print(f"  Median speed     : {s['median_speed_kmh']:.1f} km/h")
        print(f"  p95 speed        : {s['p95_speed_kmh']:.1f} km/h")
        print(f"  Max speed        : {s['max_speed_kmh']:.1f} km/h")
        print(f"  Frames detected  : {s['frames_detected']} / {total_frames}"
              f"  (missing: {s['frames_missing']})")
        top = zone_df[zone_df["player_id"] == pid].head(3)
        if not top.empty:
            print("  Main zones       :")
            for _, r in top.iterrows():
                print(f"    {r['description']:<28s} "
                      f"{r['seconds']:6.1f} s  ({r['percent']:.1f}%)")
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tennis player movement analysis: minimap, heatmap, speed"
    )
    parser.add_argument(
        "--video", default="data/Input_video2.mp4",
        help="Source video; used to derive --players and --court defaults "
             "when those are not given explicitly",
    )
    parser.add_argument(
        "--players", default=None,
        help="CSV with player tracking (defaults to "
             "outputs/player_coordinates/players_<video name>.csv)",
    )
    parser.add_argument(
        "--court", default=None,
        help="CSV with court coordinates FROM THE SAME video as the players "
             "(defaults to outputs/court_coordinates/<video name>_court.csv)",
    )
    parser.add_argument("--fps",      type=float, default=30.0,
                        help="Frames per second of the source video (default: 30)")
    parser.add_argument("--output",   default="outputs/player_analysis",
                        help="Output directory (default: outputs/player_analysis)")
    parser.add_argument("--min-area", type=int, default=500, dest="min_area",
                        help="Filter out detections with area < N pixels (default: 500)")
    parser.add_argument("--anchor", choices=["feet", "centroid"], default="feet",
                        help="Point projected to the ground: feet (bottom-center "
                             "of the bbox, default) or blob centroid")
    parser.add_argument("--max-speed", type=float, default=45.0,
                        dest="max_speed",
                        help="Speed (km/h) above which a displacement is "
                             "considered a tracking glitch (default: 45)")
    parser.add_argument("--stride",   type=int, default=3,
                        help="Sample 1 frame every N for the GIF (default: 3 → ~10 fps)")
    parser.add_argument("--no-animation", action="store_true",
                        help="Skip GIF generation (faster)")
    args = parser.parse_args()

    # Derive the input CSV paths from the video name when not given explicitly,
    # so analysing a different --video reads that video's tracking output
    # instead of always loading the clip2 / Input_video2 files. Matches the
    # naming used by playerTracking.py (players_<stem>.csv) and
    # court_tracking.py (<stem>_court.csv).
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    if args.players is None:
        args.players = os.path.join("outputs", "player_coordinates",
                                    f"players_{video_stem}.csv")
    if args.court is None:
        args.court = os.path.join("outputs", "court_coordinates",
                                  f"{video_stem}_court.csv")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.players}  (anchor: {args.anchor})")
    player_data = _load_and_convert(args.players, args.court,
                                    args.min_area, args.anchor)

    all_frames_set: set = set()
    for sub in player_data.values():
        all_frames_set.update(sub["frame"].tolist())
    all_frames  = np.array(sorted(all_frames_set))
    total_frames = len(all_frames)
    print(f"  {total_frames} total frames, {len(player_data)} players")

    # speed
    stats_map: dict = {}
    speed_map: dict = {}
    for pid, sub in player_data.items():
        stats, speed_series = _compute_speeds(sub, args.fps, args.max_speed)
        stats_map[pid] = stats
        speed_map[pid] = speed_series

    zone_df = _compute_zone_stats(player_data, args.fps)

    print("\nGenerating output …")
    _save_heatmaps(player_data, out_dir)
    _save_speed_csv(player_data, speed_map, out_dir)
    _save_zone_outputs(player_data, zone_df, out_dir)

    if not args.no_animation:
        _save_minimap(player_data, all_frames, args.fps, args.stride, out_dir)

    _print_summary(stats_map, zone_df, total_frames)
    print(f"Output in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
