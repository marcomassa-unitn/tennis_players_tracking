"""
Project player pixel positions to court meters (via CourtConverter) and emit
movement analytics.

Outputs: minimap.gif, heatmap_p1/p2.png, heatmap_combined.png, speed_stats.csv,
zone_stats.csv, zones.png, plus a terminal summary.

Ground anchor defaults to the feet point (bbox bottom-center): projecting the
body centroid (~1 m off the ground) through the homography lands it far past the
net.

    python utils/player_analysis.py
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter

# ── ITF singles court dimensions ───────────────────────────────────────────────
# Shared via utils/court_geometry; imported (not redefined) so live_view.py can
# read them as player_analysis.<name> module attributes.
from utils.court_geometry import W_m, L_m, SVC_T, SVC_B, NET, CL_X

# Player colour convention, kept identical across every figure so colour -> player
# is unambiguous: P1 = warm (red), P2 = cool (blue). Paired with the heatmap
# colormaps (_HEAT_CMAPS) and legend patches (_HEAT_LEGEND).
_COLORS = {1: "red",       2: "deepskyblue"}
_LABELS = {1: "P1 (red)",  2: "P2 (blue)"}

# Heatmap colormaps. Alpha ramps from 0 (empty -> transparent, background stays
# clean) to 1 (dense -> opaque) over two well-separated hues (P1 warm red→yellow,
# P2 cool cyan→blue). The old `hot`+`Blues_r` pair fogged into mud where players
# overlapped and sent P2's densest bins to white; transparent tails make overlap
# read as "both were here".
def _alpha_ramp_cmap(name, rgb_lo, rgb_hi):
    """Build a colormap that goes transparent at low density, saturated-opaque at high."""
    r0, g0, b0 = rgb_lo
    r1, g1, b1 = rgb_hi
    cdict = {
        "red":   [(0.0, r0, r0), (1.0, r1, r1)],
        "green": [(0.0, g0, g0), (1.0, g1, g1)],
        "blue":  [(0.0, b0, b0), (1.0, b1, b1)],
        # Empty bins invisible (alpha 0), then climb fast so faint traffic hits a
        # readable opacity early (0.20 -> 0.55, 0.40 -> 0.90).
        "alpha": [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.20, 0.55, 0.55),
                  (0.40, 0.90, 0.90), (1.0, 1.0, 1.0)],
    }
    return LinearSegmentedColormap(name, cdict, N=256)

# Low end is deliberately luminous (not dark red/blue) so the faintest traffic
# still separates from the near-black background.
_HEAT_CMAPS  = {
    1: _alpha_ramp_cmap("p1_warm", (1.0, 0.45, 0.10), (1.0, 0.95, 0.25)),
    2: _alpha_ramp_cmap("p2_cool", (0.20, 0.65, 1.0), (0.35, 0.97, 1.0)),
}
# Legend patches: a vivid mid-ramp colour of each map (what a busy zone looks like).
_HEAT_LEGEND = {1: "#ff5a1f",     2: "#19c3ff"}

# ── court zones ────────────────────────────────────────────────────────────────
# Analysis covers the whole walkable area, not just the playing rectangle.
# _BEHIND = run-off behind each baseline (m), _SIDE = lateral run-off beyond each
# sideline (m; ITF recommends ~3.66 m).
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

# Shared y-limits for every court figure; y inverted (far baseline y=0 at top) to
# match the camera view.
_COURT_YLIM = (L_m + _BEHIND + 0.4, -_BEHIND - 0.4)


# ── court drawing ──────────────────────────────────────────────────────────────

def _draw_court(ax: plt.Axes) -> None:
    """Draw the ITF singles court (meters) onto ax."""
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

    # extend to the whole walkable area; y inverted (far baseline y=0 at top)
    ax.set_xlim(-_SIDE - 0.4, W_m + _SIDE + 0.4)
    ax.set_ylim(*_COURT_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")


def _save_fig(fig, path) -> None:
    """Save the figure PNG keeping its dark facecolor, then close and log.

    Shared by the zone and heatmap exports; the animated minimap uses its own
    writer and bypasses this.
    """
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ── loading and conversion ─────────────────────────────────────────────────────

def _load_and_convert(players_csv: str, court_csv: str, min_area: int,
                      anchor: str = "feet") -> dict[int, pd.DataFrame]:
    """
    Load the players CSV, drop detections with area < min_area, and append
    x_m/y_m court coordinates per player.

    anchor "feet" projects (cx, y + h); "centroid" projects (cx, cy).
    Returns {player_id: DataFrame}. Raises ValueError on an unknown anchor.
    """
    df = pd.read_csv(players_csv)
    df = df[df["area"] >= min_area].copy()

    if anchor == "feet":
        # Non-fatal sanity check: feet anchor assumes "y" is the bbox top, so
        # cy ≈ y + h/2. If most rows violate that, "y" isn't the top edge and the
        # y + h projection is misplaced — warn but continue.
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
    Per-frame speed (km/h) and aggregate stats.

    Speed is NaN across non-consecutive frames (missed detection). Displacements
    over max_speed_kmh are treated as tracking glitches (blob jumps, ID swaps)
    and excluded from both speed and total distance.
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

    # align with sub (leading NaN)
    speed_full = np.concatenate([[np.nan], speed_kmh])

    valid_dist = dist[~np.isnan(speed_kmh)]
    # With no valid sample, fall back to 0.0 so the np.nan* reductions don't warn
    # and return NaN.
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
        # count dropped frames, not gap events: a gap of g contributes (g - 1)
        "frames_missing":   int(np.sum(gaps[gaps > 1] - 1)),
    }
    return stats, speed_full


# ── court zones ────────────────────────────────────────────────────────────────

def _zone_of(x_m: float, y_m: float) -> str | None:
    """Map a court point to its zone id ("FB-L", "NB-out-OR", …), or None if out of range."""
    if x_m < -_SIDE - 1.0 or x_m > W_m + _SIDE + 1.0:
        return None
    # Both axes are half-open [low, high), so a point on a dividing line falls to
    # the higher band (x == CL_X -> "R", x == W_m -> "OR"), consistent with the
    # y-band test below.
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
    Time spent per zone, one row per (player_id, zone): frames, seconds, percent.
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
                # n_tot > 0 here (the earlier continue skips empty players)
                "percent":    round(100.0 * n / n_tot, 2),
            })
    if not rows:
        # Keep the expected columns even when empty, so downstream sort/filter/
        # to_csv don't raise on a column-less frame.
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(
        ["player_id", "percent"], ascending=[True, False]
    ).reset_index(drop=True)


def _save_zone_outputs(player_data: dict, zone_df: pd.DataFrame,
                       out_dir: Path) -> None:
    """Write zone_stats.csv and zones.png (per-zone percentages drawn on the court)."""
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
        ax.set_ylim(*_COURT_YLIM)
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
    _save_fig(fig, path)


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
    # Mask bins below a small floor (-> transparent) so only real traffic paints
    # and faint tails don't fog the figure.
    return np.ma.masked_less(H, 0.04)


# Slightly darker than the other figures for maximum contrast against the
# alpha-ramped heat colours.
_HEAT_BG = "#10101c"


def _save_heatmaps(player_data: dict, out_dir: Path) -> None:
    # warm/cool per _COLORS (P1 warm, P2 cool)
    for pid, sub in player_data.items():
        fig, ax = plt.subplots(figsize=(4, 9))
        fig.patch.set_facecolor(_HEAT_BG)
        _draw_court(ax)
        H = _make_heat_array(sub["x_m"].values, sub["y_m"].values)
        # alpha is baked into the colormap, so no flat alpha= here
        ax.imshow(H, origin="lower", extent=_HEAT_EXTENT, vmin=0.0, vmax=1.0,
                  cmap=_HEAT_CMAPS.get(pid, "hot"), aspect="auto",
                  interpolation="bilinear")
        ax.set_title(f"Heatmap – Player {pid}", color="white",
                     fontsize=11, pad=4)
        path = out_dir / f"heatmap_p{pid}.png"
        _save_fig(fig, path)

    # combined: reuse the same per-player maps; transparent empty bins let the two
    # layers coexist (red-only P1, cyan-only P2, blend only on real overlap)
    fig, ax = plt.subplots(figsize=(4, 9))
    fig.patch.set_facecolor(_HEAT_BG)
    _draw_court(ax)
    for pid in (1, 2):
        if pid not in player_data:
            continue
        H = _make_heat_array(player_data[pid]["x_m"].values,
                             player_data[pid]["y_m"].values)
        ax.imshow(H, origin="lower", extent=_HEAT_EXTENT, vmin=0.0, vmax=1.0,
                  cmap=_HEAT_CMAPS[pid], aspect="auto", interpolation="bilinear")
    # legend colours match the warm/cool player convention (_HEAT_LEGEND)
    patches = [mpatches.Patch(color=_HEAT_LEGEND[1], label="Player 1"),
               mpatches.Patch(color=_HEAT_LEGEND[2], label="Player 2")]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              framealpha=0.6, facecolor="#222", edgecolor="none",
              labelcolor="white")
    ax.set_title("Combined heatmap", color="white", fontsize=11, pad=4)
    path = out_dir / "heatmap_combined.png"
    _save_fig(fig, path)


# ── animated minimap ───────────────────────────────────────────────────────────

def _save_minimap(player_data: dict, all_frames: np.ndarray,
                  fps: float, stride: int, out_dir: Path) -> None:
    # index by frame for O(1) per-frame lookup
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
                # keep positions out to the walkable run-off, not just the court
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

    # Derive CSV paths from the video stem when not given, matching the naming of
    # playerTracking.py (players_<stem>.csv) and court_tracking.py
    # (<stem>_court.csv), so a different --video reads its own tracking output.
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

    # per-player speed + aggregates
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
