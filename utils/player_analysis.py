"""
utils/player_analysis.py

Converte le posizioni pixel dei giocatori in coordinate reali (metri) tramite
CourtConverter e genera:
  - minimap.gif           animazione top-down del campo
  - heatmap_p1/p2.png     mappa di densità per giocatore
  - heatmap_combined.png  entrambi sovrapposti
  - speed_stats.csv       velocità per frame in km/h
  - zone_stats.csv        tempo trascorso in ogni zona del campo
  - zones.png             occupazione delle zone disegnata sul campo
  + riepilogo statistiche a terminale

Il punto proiettato a terra è per default il punto-piedi (centro-basso del
bounding box): il centroide del corpo sta ~1 m sopra il suolo e l'omografia
lo proietterebbe molto più lontano dalla rete di quanto sia davvero.

Uso rapido (da radice progetto):
    python utils/player_analysis.py

Uso completo:
    python utils/player_analysis.py \\
        --players outputs/players_clip2.csv \\
        --court   outputs/court_coordinates/input_video_court.csv \\
        --fps     30 \\
        --output  outputs/player_analysis \\
        --stride  3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend headless, non apre finestre
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter

# ── dimensioni campo ITF singles ───────────────────────────────────────────────
_FT   = 0.3048
W_m   = 27.0 * _FT          # 8.2296 m  larghezza (singles)
L_m   = 78.0 * _FT          # 23.7744 m lunghezza
SVC_T = 18.0 * _FT          # 5.4864 m  linea servizio lato far
SVC_B = L_m - SVC_T         # 18.288 m  linea servizio lato near
NET   = L_m / 2.0            # 11.8872 m rete
CL_X  = W_m / 2.0            # 4.1148 m  linea centrale servizio

# colori per i due giocatori
_COLORS = {1: "lime",   2: "tomato"}
_LABELS = {1: "P1 (lime)", 2: "P2 (rosso)"}

# ── zone del campo ─────────────────────────────────────────────────────────────
# L'analisi copre tutta l'area calpestabile, non solo il rettangolo di gioco:
# _BEHIND = run-off tipico dietro ciascuna baseline, _SIDE = run-off laterale
# (raccomandazione ITF ~3.66 m) oltre ciascuna sideline.
_BEHIND = 6.5
_SIDE   = 3.7
_ZONE_BANDS = [
    ("FB-out", "Dietro baseline (far)",   -_BEHIND, 0.0),
    ("FB",     "Fondocampo (far)",         0.0,     SVC_T),
    ("FF",     "Zona servizio (far)",      SVC_T,   NET),
    ("NF",     "Zona servizio (near)",     NET,     SVC_B),
    ("NB",     "Fondocampo (near)",        SVC_B,   L_m),
    ("NB-out", "Dietro baseline (near)",   L_m,     L_m + _BEHIND),
]
_SIDES = [("OL", "esterno sx"), ("L", "sx"), ("R", "dx"), ("OR", "esterno dx")]
_SIDE_X = {"OL": (-_SIDE, 0.0), "L": (0.0, CL_X),
           "R": (CL_X, W_m), "OR": (W_m, W_m + _SIDE)}


# ── disegno campo ──────────────────────────────────────────────────────────────

def _draw_court(ax: plt.Axes) -> None:
    """Disegna campo ITF singles su ax (coordinate in metri)."""
    ax.set_facecolor("#2d6a4f")
    kw = dict(color="white", linewidth=1.4, solid_capstyle="round")

    # perimetro
    ax.plot([0, W_m], [0,   0  ], **kw)
    ax.plot([0, W_m], [L_m, L_m], **kw)
    ax.plot([0, 0  ], [0,   L_m], **kw)
    ax.plot([W_m, W_m], [0, L_m], **kw)
    # linee di servizio
    ax.plot([0, W_m], [SVC_T, SVC_T], **kw)
    ax.plot([0, W_m], [SVC_B, SVC_B], **kw)
    ax.plot([CL_X, CL_X], [SVC_T, SVC_B], **kw)
    # rete
    ax.plot([0, W_m], [NET, NET], color="#f4d03f", linewidth=2.8)

    # limiti estesi a tutta l'area calpestabile (run-off laterali e di fondo);
    # y invertito: baseline lontana (y=0) in alto, come nella vista camera
    ax.set_xlim(-_SIDE - 0.4, W_m + _SIDE + 0.4)
    ax.set_ylim(L_m + _BEHIND + 0.4, -_BEHIND - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")


# ── caricamento e conversione ──────────────────────────────────────────────────

def _load_and_convert(players_csv: str, court_csv: str, min_area: int,
                      anchor: str = "feet") -> dict[int, pd.DataFrame]:
    """
    Carica il CSV dei giocatori, filtra le detection con area < min_area e
    aggiunge le colonne x_m, y_m con le coordinate reali in metri.

    anchor = "feet"     → proietta il punto-piedi (cx, y + h)  [default]
    anchor = "centroid" → proietta il centroide del blob (cx, cy)
    Ritorna {player_id: DataFrame}.
    """
    df = pd.read_csv(players_csv)
    df = df[df["area"] >= min_area].copy()

    if anchor == "feet":
        df["ax"] = df["cx"]
        df["ay"] = df["y"] + df["h"]
    elif anchor == "centroid":
        df["ax"] = df["cx"]
        df["ay"] = df["cy"]
    else:
        raise ValueError(f"anchor sconosciuto: {anchor!r}")

    conv = CourtConverter(court_csv)
    player_data: dict[int, pd.DataFrame] = {}
    for pid in sorted(df["player_id"].unique()):
        sub = df[df["player_id"] == pid].sort_values("frame").copy()
        pts = conv.to_meters_batch(sub[["ax", "ay"]].values)
        sub["x_m"] = pts[:, 0]
        sub["y_m"] = pts[:, 1]
        player_data[int(pid)] = sub.reset_index(drop=True)
    return player_data


# ── statistiche velocità ───────────────────────────────────────────────────────

def _compute_speeds(sub: pd.DataFrame, fps: float,
                    max_speed_kmh: float = 45.0) -> tuple[dict, np.ndarray]:
    """
    Calcola velocità per frame (km/h) e statistiche aggregate.
    Velocità = NaN per frame non consecutivi (giocatore non rilevato).
    Spostamenti sopra max_speed_kmh sono glitch di tracking (salti di blob,
    scambi di ID): vengono esclusi sia dalla velocità sia dalla distanza.
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

    # serie allineata con sub (primo valore sempre NaN)
    speed_full = np.concatenate([[np.nan], speed_kmh])

    valid_dist = dist[~np.isnan(speed_kmh)]
    stats = {
        "total_distance_m": round(float(valid_dist.sum()), 2),
        "avg_speed_kmh":    round(float(np.nanmean(speed_kmh)), 2),
        "median_speed_kmh": round(float(np.nanmedian(speed_kmh)), 2),
        "max_speed_kmh":    round(float(np.nanmax(speed_kmh)), 2)
                            if not np.all(np.isnan(speed_kmh)) else 0.0,
        "p95_speed_kmh":    round(float(np.nanpercentile(speed_kmh, 95)), 2)
                            if not np.all(np.isnan(speed_kmh)) else 0.0,
        "frames_detected":  len(sub),
        "frames_missing":   int(np.sum(gaps > 1)),
    }
    return stats, speed_full


# ── zone del campo ─────────────────────────────────────────────────────────────

def _zone_of(x_m: float, y_m: float) -> str | None:
    """Ritorna l'id zona ("FB-L", "NB-out-OR", …) oppure None se fuori range."""
    if x_m < -_SIDE - 1.0 or x_m > W_m + _SIDE + 1.0:
        return None
    if x_m < 0:
        side = "OL"
    elif x_m < CL_X:
        side = "L"
    elif x_m <= W_m:
        side = "R"
    else:
        side = "OR"
    for zid, _, y0, y1 in _ZONE_BANDS:
        if y0 <= y_m < y1:
            return f"{zid}-{side}"
    return None


def _compute_zone_stats(player_data: dict, fps: float) -> pd.DataFrame:
    """
    Tempo trascorso da ogni giocatore in ciascuna zona del campo.
    Una riga per (player_id, zona): frame, secondi e percentuale.
    """
    rows = []
    band_labels = {zid: label for zid, label, _, _ in _ZONE_BANDS}
    for pid, sub in player_data.items():
        zones = [_zone_of(x, y) for x, y in sub[["x_m", "y_m"]].values]
        zones = pd.Series([z for z in zones if z is not None])
        n_tot = len(zones)
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
                "percent":    round(100.0 * n / n_tot, 2),
            })
    return pd.DataFrame(rows).sort_values(
        ["player_id", "percent"], ascending=[True, False]
    ).reset_index(drop=True)


def _save_zone_outputs(player_data: dict, zone_df: pd.DataFrame,
                       out_dir: Path) -> None:
    """Salva zone_stats.csv e la figura zones.png (percentuali sul campo)."""
    csv_path = out_dir / "zone_stats.csv"
    zone_df.to_csv(csv_path, index=False)
    print(f"  Salvato: {csv_path}")

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
        ax.set_title(f"Zone – Giocatore {pid}", color="white",
                     fontsize=11, pad=4)

    path = out_dir / "zones.png"
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Salvato: {path}")


# ── heatmap ────────────────────────────────────────────────────────────────────

# estensione della heatmap: tutta l'area calpestabile
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
    cmaps = {1: "hot", 2: "Blues_r"}

    for pid, sub in player_data.items():
        fig, ax = plt.subplots(figsize=(4, 9))
        fig.patch.set_facecolor("#1a1a2e")
        _draw_court(ax)
        H = _make_heat_array(sub["x_m"].values, sub["y_m"].values)
        ax.imshow(H, origin="lower", extent=_HEAT_EXTENT,
                  cmap=cmaps.get(pid, "hot"), alpha=0.70, aspect="auto")
        ax.set_title(f"Heatmap – Giocatore {pid}", color="white",
                     fontsize=11, pad=4)
        path = out_dir / f"heatmap_p{pid}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Salvato: {path}")

    # heatmap combinata
    fig, ax = plt.subplots(figsize=(4, 9))
    fig.patch.set_facecolor("#1a1a2e")
    _draw_court(ax)
    cmap_alpha = {1: ("hot", 0.55), 2: ("Blues_r", 0.55)}
    for pid, (cmap, alpha) in cmap_alpha.items():
        if pid not in player_data:
            continue
        H = _make_heat_array(player_data[pid]["x_m"].values,
                             player_data[pid]["y_m"].values)
        ax.imshow(H, origin="lower", extent=_HEAT_EXTENT,
                  cmap=cmap, alpha=alpha, aspect="auto")
    patches = [mpatches.Patch(color="red",       label="Giocatore 1"),
               mpatches.Patch(color="steelblue", label="Giocatore 2")]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              framealpha=0.5, facecolor="#333", edgecolor="none",
              labelcolor="white")
    ax.set_title("Heatmap combinata", color="white", fontsize=11, pad=4)
    path = out_dir / "heatmap_combined.png"
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Salvato: {path}")


# ── minimap animata ────────────────────────────────────────────────────────────

def _save_minimap(player_data: dict, all_frames: np.ndarray,
                  fps: float, stride: int, out_dir: Path) -> None:
    # indicizzazione rapida per frame
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
                # mostra anche le posizioni fuori dal campo (area calpestabile)
                if (-_SIDE - 1 <= xm <= W_m + _SIDE + 1
                        and -_BEHIND - 1 <= ym <= L_m + _BEHIND + 1):
                    dot.set_data([xm], [ym])
                    continue
            dot.set_data([], [])
        return list(dots.values()) + [title]

    ani = FuncAnimation(fig, _update, frames=sampled, blit=True,
                        interval=1000 // gif_fps)
    path = out_dir / "minimap.gif"
    print(f"  Generazione GIF ({len(sampled)} frame, {gif_fps} fps) …")
    ani.save(str(path), writer=PillowWriter(fps=gif_fps), dpi=90)
    plt.close(fig)
    print(f"  Salvato: {path}")


# ── CSV velocità ───────────────────────────────────────────────────────────────

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
    print(f"  Salvato: {path}")


# ── riepilogo terminale ────────────────────────────────────────────────────────

def _print_summary(stats_map: dict, zone_df: pd.DataFrame,
                   total_frames: int) -> None:
    print("\n" + "=" * 52)
    print("  PLAYER ANALYSIS  –  RIEPILOGO")
    print("=" * 52)
    for pid, s in stats_map.items():
        print(f"\nGiocatore {pid} ({_COLORS[pid]}):")
        print(f"  Distanza totale  : {s['total_distance_m']:.1f} m")
        print(f"  Velocità media   : {s['avg_speed_kmh']:.1f} km/h")
        print(f"  Velocità mediana : {s['median_speed_kmh']:.1f} km/h")
        print(f"  Velocità p95     : {s['p95_speed_kmh']:.1f} km/h")
        print(f"  Velocità max     : {s['max_speed_kmh']:.1f} km/h")
        print(f"  Frame rilevati   : {s['frames_detected']} / {total_frames}"
              f"  (mancanti: {s['frames_missing']})")
        top = zone_df[zone_df["player_id"] == pid].head(3)
        if not top.empty:
            print("  Zone principali  :")
            for _, r in top.iterrows():
                print(f"    {r['description']:<28s} "
                      f"{r['seconds']:6.1f} s  ({r['percent']:.1f}%)")
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analisi movimenti giocatori tennis: minimap, heatmap, velocità"
    )
    parser.add_argument(
        "--players", default="outputs/players_clip2.csv",
        help="CSV con tracking giocatori (default: outputs/players_clip2.csv)",
    )
    parser.add_argument(
        "--court", default="outputs/court_coordinates/Input_video2_court.csv",
        help="CSV con coordinate campo DELLO STESSO video dei giocatori "
             "(default: outputs/court_coordinates/Input_video2_court.csv)",
    )
    parser.add_argument("--fps",      type=float, default=30.0,
                        help="Frame per secondo del video sorgente (default: 30)")
    parser.add_argument("--output",   default="outputs/player_analysis",
                        help="Directory di output (default: outputs/player_analysis)")
    parser.add_argument("--min-area", type=int, default=500, dest="min_area",
                        help="Filtra detection con area < N pixel (default: 500)")
    parser.add_argument("--anchor", choices=["feet", "centroid"], default="feet",
                        help="Punto proiettato a terra: piedi (centro-basso del "
                             "bbox, default) o centroide del blob")
    parser.add_argument("--max-speed", type=float, default=45.0,
                        dest="max_speed",
                        help="Velocità (km/h) oltre cui uno spostamento è "
                             "considerato glitch di tracking (default: 45)")
    parser.add_argument("--stride",   type=int, default=3,
                        help="Campiona 1 frame ogni N per la GIF (default: 3 → ~10 fps)")
    parser.add_argument("--no-animation", action="store_true",
                        help="Salta la generazione della GIF (più veloce)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caricamento: {args.players}  (anchor: {args.anchor})")
    player_data = _load_and_convert(args.players, args.court,
                                    args.min_area, args.anchor)

    all_frames_set: set = set()
    for sub in player_data.values():
        all_frames_set.update(sub["frame"].tolist())
    all_frames  = np.array(sorted(all_frames_set))
    total_frames = len(all_frames)
    print(f"  {total_frames} frame totali, {len(player_data)} giocatori")

    # velocità
    stats_map: dict = {}
    speed_map: dict = {}
    for pid, sub in player_data.items():
        stats, speed_series = _compute_speeds(sub, args.fps, args.max_speed)
        stats_map[pid] = stats
        speed_map[pid] = speed_series

    zone_df = _compute_zone_stats(player_data, args.fps)

    print("\nGenerazione output …")
    _save_heatmaps(player_data, out_dir)
    _save_speed_csv(player_data, speed_map, out_dir)
    _save_zone_outputs(player_data, zone_df, out_dir)

    if not args.no_animation:
        _save_minimap(player_data, all_frames, args.fps, args.stride, out_dir)

    _print_summary(stats_map, zone_df, total_frames)
    print(f"Output in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
