"""
utils/shot_analysis.py

Rileva il momento dei colpi e classifica dritto/rovescio a partire dal CSV
della palla prodotto da ballTracking/BallTracking.py (frame,x,y,w,h,cx,cy,area)
e dal CSV dei giocatori.

Rilevazione del colpo (sul tracciato palla smussato con Savitzky-Golay):
  1. candidati = inversione persistente del segno della velocità verticale vy
     (in immagine: palla che viaggia verso il giocatore far ha vy<0, verso il
     giocatore near vy>0; un colpo inverte la direzione) UNITI ai picchi del
     modulo dell'accelerazione |dv| (cambio brusco di velocità);
  2. un candidato è un COLPO solo se la palla è vicina al bounding box di un
     giocatore (i rimbalzi invertono vy ma avvengono lontano dai giocatori);
  3. gap minimo tra colpi consecutivi (default 0.5 s), si tiene il candidato
     con accelerazione maggiore.

Dritto / rovescio al frame del colpo:
  - lato campo del giocatore (near/far) dal suo punto-piedi proiettato in
    metri: near = lo vediamo di spalle, la sua destra è la destra immagine;
    far = lo vediamo frontale, la sua destra è la SINISTRA immagine;
  - dritto se la palla è dal lato della mano dominante, rovescio altrimenti;
    per i mancini (--p1-hand/--p2-hand left) il ragionamento si inverte;
  - se la palla è quasi sull'asse del corpo il colpo è marcato "unknown".

Uso (da radice progetto, dopo aver generato il ball CSV con BallTracking):
    python utils/shot_analysis.py --ball outputs/ball_clip2.csv
    python utils/shot_analysis.py --ball outputs/ball_clip2.csv \\
        --p1-hand right --p2-hand left

Validazione della logica senza modello YOLO (traiettoria sintetica):
    python utils/shot_analysis.py --self-test
"""

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter

_FT = 0.3048
W_m = 27.0 * _FT
L_m = 78.0 * _FT
NET = L_m / 2.0


# ── caricamento ────────────────────────────────────────────────────────────────

def load_ball_track(ball_csv: str) -> pd.DataFrame:
    """
    Carica il CSV palla, lo riallinea su tutti i frame e smussa cx/cy.
    Ritorna un DataFrame indicizzato per frame con colonne cx, cy (smussate),
    NaN dove la palla non è mai stata vista.
    """
    df = pd.read_csv(ball_csv)
    if df.empty:
        raise ValueError(f"Ball CSV vuoto: {ball_csv}")
    full = df.set_index("frame")[["cx", "cy"]].reindex(
        range(int(df["frame"].min()), int(df["frame"].max()) + 1))
    full = full.interpolate(limit=8, limit_area="inside")

    win = 9
    for col in ("cx", "cy"):
        vals = full[col].values.astype(float)
        ok = ~np.isnan(vals)
        if ok.sum() > win:
            sm = vals.copy()
            sm[ok] = savgol_filter(vals[ok], win, 2)
            full[col] = sm
    return full


def load_player_boxes(players_csv: str) -> dict:
    """{frame: {pid: (x, y, w, h)}} dal CSV del tracker giocatori."""
    df = pd.read_csv(players_csv)
    boxes = defaultdict(dict)
    for r in df.itertuples():
        boxes[int(r.frame)][int(r.player_id)] = (
            float(r.x), float(r.y), float(r.w), float(r.h))
    return boxes


# ── rilevazione colpi ──────────────────────────────────────────────────────────

def _nearest_box(boxes, frame, pid, radius=3):
    """Box del giocatore pid nel frame più vicino entro ±radius, o None."""
    for d in range(radius + 1):
        for f in (frame - d, frame + d):
            if f in boxes and pid in boxes[f]:
                return boxes[f][pid]
    return None


def _ball_near_player(ball_xy, box, expand_w=0.9, expand_h=0.55):
    """True se la palla è dentro il box espanso del giocatore."""
    x, y, w, h = box
    bx, by = ball_xy
    return (x - expand_w * w <= bx <= x + w + expand_w * w
            and y - expand_h * h <= by <= y + h + expand_h * h)


def detect_hits(track: pd.DataFrame, boxes: dict, fps: float,
                min_gap_s: float = 0.5, vy_min: float = 0.5,
                acc_thr: float = 1.5) -> list:
    """
    Ritorna [(frame, player_id, acc_strength), ...] dei colpi rilevati.

    vy_min  : modulo minimo (px/frame) della velocità media prima/dopo perché
              un'inversione di segno conti come candidato
    acc_thr : soglia (px/frame^2) sui picchi del modulo dell'accelerazione
    """
    frames = track.index.values
    cx = track["cx"].values
    cy = track["cy"].values
    valid = ~(np.isnan(cx) | np.isnan(cy))

    vx = np.gradient(np.where(valid, cx, np.nan))
    vy = np.gradient(np.where(valid, cy, np.nan))
    acc = np.hypot(np.gradient(vx), np.gradient(vy))

    win = 4
    candidates = {}   # idx -> strength
    for i in range(win, len(frames) - win):
        if not valid[i - win:i + win + 1].all():
            continue
        before = np.nanmean(vy[i - win:i])
        after = np.nanmean(vy[i + 1:i + 1 + win])
        flip = (before * after < 0
                and min(abs(before), abs(after)) >= vy_min)
        acc_peak = (acc[i] >= acc_thr
                    and acc[i] == np.nanmax(acc[i - win:i + win + 1]))
        if flip or acc_peak:
            candidates[i] = max(candidates.get(i, 0.0), float(acc[i]))

    # filtro prossimità giocatore
    near_player = {}
    for i, strength in candidates.items():
        f = int(frames[i])
        ball_xy = (cx[i], cy[i])
        best = None
        for pid in (1, 2):
            box = _nearest_box(boxes, f, pid)
            if box is not None and _ball_near_player(ball_xy, box):
                bx_c = box[0] + box[2] / 2.0
                by_c = box[1] + box[3] / 2.0
                d = np.hypot(ball_xy[0] - bx_c, ball_xy[1] - by_c)
                if best is None or d < best[1]:
                    best = (pid, d)
        if best is not None:
            near_player[i] = (best[0], strength)

    # gap minimo: tieni il candidato più forte in ogni finestra
    min_gap = int(min_gap_s * fps)
    hits = []
    for i in sorted(near_player):
        pid, strength = near_player[i]
        f = int(frames[i])
        if hits and f - hits[-1][0] < min_gap:
            if strength > hits[-1][2]:
                hits[-1] = (f, pid, strength)
            continue
        hits.append((f, pid, strength))
    return hits


# ── classificazione dritto / rovescio ──────────────────────────────────────────

def classify_stroke(ball_cx, player_box, player_side, hand,
                    deadband_frac=0.12):
    """
    player_side : "near" | "far" (rispetto alla camera)
    hand        : "right" | "left"
    Ritorna "forehand", "backhand" o "unknown" (palla sull'asse del corpo).
    """
    x, y, w, h = player_box
    player_cx = x + w / 2.0
    db = ball_cx - player_cx
    if abs(db) < deadband_frac * w:
        return "unknown"
    # la mano dominante è verso destra-immagine se (near e destro) o (far e mancino)
    dominant_is_image_right = (player_side == "near") == (hand == "right")
    return "forehand" if (db > 0) == dominant_is_image_right else "backhand"


def analyze_shots(track, boxes, conv, fps, hands, min_gap_s=0.5):
    """Pipeline completa: rileva colpi e li classifica. Ritorna DataFrame."""
    hits = detect_hits(track, boxes, fps, min_gap_s=min_gap_s)
    rows = []
    for f, pid, strength in hits:
        box = _nearest_box(boxes, f, pid)
        if box is None:
            continue
        # mediana della posizione palla su ±2 frame (riduce il rumore)
        sel = track.loc[max(track.index.min(), f - 2): f + 2]
        ball_cx = float(np.nanmedian(sel["cx"]))
        ball_cy = float(np.nanmedian(sel["cy"]))

        feet = (box[0] + box[2] / 2.0, box[1] + box[3])
        y_m = conv.to_meters(*feet)[1]
        side = "near" if y_m > NET else "far"
        stroke = classify_stroke(ball_cx, box, side, hands[pid])

        bx_m, by_m = conv.to_meters(ball_cx, ball_cy)
        rows.append({
            "frame": f, "time_s": round(f / fps, 2), "player_id": pid,
            "side": side, "hand": hands[pid], "stroke": stroke,
            "ball_cx": round(ball_cx, 1), "ball_cy": round(ball_cy, 1),
            "player_cx": round(box[0] + box[2] / 2.0, 1),
            "ball_x_m": round(bx_m, 2), "ball_y_m": round(by_m, 2),
        })
    return pd.DataFrame(rows)


# ── output ─────────────────────────────────────────────────────────────────────

def save_shot_frames(shots: pd.DataFrame, video_path: str, boxes: dict,
                     out_dir: Path) -> None:
    """PNG di verifica: frame del colpo con box giocatore e palla evidenziati."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  (video non disponibile, salto i PNG: {video_path})")
        return
    for r in shots.itertuples():
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame)
        ok, fr = cap.read()
        if not ok:
            continue
        box = _nearest_box(boxes, r.frame, r.player_id)
        if box:
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(fr, (int(r.ball_cx), int(r.ball_cy)), 9, (0, 255, 255), 2)
        label = f"P{r.player_id} {r.stroke} (frame {r.frame})"
        cv2.putText(fr, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 3)
        path = out_dir / f"shot_f{r.frame:05d}_P{r.player_id}_{r.stroke}.png"
        cv2.imwrite(str(path), fr)
    cap.release()
    print(f"  PNG dei colpi salvati in {out_dir}")


def print_summary(shots: pd.DataFrame, hands: dict) -> None:
    print("\n" + "=" * 56)
    print("  SHOT ANALYSIS  -  RIEPILOGO")
    print("=" * 56)
    if shots.empty:
        print("  Nessun colpo rilevato.")
        return
    for r in shots.itertuples():
        print(f"  frame {r.frame:5d} ({r.time_s:6.2f}s)  "
              f"P{r.player_id} ({r.side:4s}, {r.hand:5s})  ->  {r.stroke}")
    print()
    for pid, sub in shots.groupby("player_id"):
        n_fh = (sub["stroke"] == "forehand").sum()
        n_bh = (sub["stroke"] == "backhand").sum()
        n_uk = (sub["stroke"] == "unknown").sum()
        print(f"  P{pid} ({hands[pid]:5s}): {len(sub)} colpi  "
              f"-  dritti {n_fh}, rovesci {n_bh}, incerti {n_uk}")
    print()


# ── self-test con traiettoria sintetica ────────────────────────────────────────

def _synthetic_court_csv(path: str) -> None:
    """Campo sintetico: omografia da 4 corner px arbitrari + proporzioni ITF."""
    corners_m = np.array([[0, 0], [W_m, 0], [0, L_m], [W_m, L_m]],
                         dtype=np.float32)                 # TL TR BL BR
    corners_px = np.array([[700, 300], [1200, 300], [400, 860], [1500, 860]],
                          dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners_m, corners_px)

    def proj(xm, ym):
        p = H @ np.array([xm, ym, 1.0])
        return p[0] / p[2], p[1] / p[2]

    svc_t, svc_b = 18.0 * _FT, L_m - 18.0 * _FT
    pts = {"TL": (0, 0), "TR": (W_m, 0), "BL": (0, L_m), "BR": (W_m, L_m),
           "STL": (0, svc_t), "STR": (W_m, svc_t),
           "SBL": (0, svc_b), "SBR": (W_m, svc_b)}
    with open(path, "w", newline="") as f:
        f.write("label,x,y\n")
        for lab, (xm, ym) in pts.items():
            x, y = proj(xm, ym)
            f.write(f"{lab},{x:.1f},{y:.1f}\n")


def self_test() -> None:
    """
    Rally sintetico: 4 colpi con lato palla noto + rimbalzi a metà campo che
    NON devono essere rilevati come colpi. Verifica frame, giocatore e
    classificazione, anche nel caso mancino.
    """
    rng = np.random.default_rng(3)
    p1 = (900, 700, 100, 190)     # near, feet y=890, cx=950
    p2 = (920, 200, 60, 120)      # far,  feet y=320, cx=950

    # (frame_colpo, giocatore, offset palla rispetto al centro, atteso destro)
    plan = [(20, 1, +70, "forehand"),
            (60, 2, -45, "forehand"),
            (100, 1, -70, "backhand"),
            (140, 2, +45, "backhand")]

    contact = {}
    for f, pid, dx, _ in plan:
        box = p1 if pid == 1 else p2
        cx_c = box[0] + box[2] / 2.0 + dx
        cy_c = box[1] + box[3] * (0.45 if pid == 1 else 0.5)
        contact[f] = (cx_c, cy_c)

    # traiettoria: tratti lineari tra i contatti + rimbalzo (cuspide) a meta'
    frames = np.arange(0, 171)
    cx = np.full(len(frames), np.nan)
    cy = np.full(len(frames), np.nan)
    keys = sorted(contact)
    segs = [(0, (950.0, 520.0), keys[0], contact[keys[0]])]
    segs += [(keys[i], contact[keys[i]], keys[i + 1], contact[keys[i + 1]])
             for i in range(len(keys) - 1)]
    segs += [(keys[-1], contact[keys[-1]], 170, (950.0, 520.0))]
    for f0, (x0, y0), f1, (x1, y1) in segs:
        for f in range(f0, f1 + 1):
            s = (f - f0) / max(1, f1 - f0)
            # cuspide di rimbalzo a s=0.65 (deviazione verso il basso immagine)
            bounce = 35.0 * max(0.0, 1.0 - abs(s - 0.65) / 0.18)
            cx[f] = x0 + s * (x1 - x0)
            cy[f] = y0 + s * (y1 - y0) + bounce
    cx += rng.normal(0, 0.8, len(frames))
    cy += rng.normal(0, 0.8, len(frames))

    with tempfile.TemporaryDirectory() as tmp:
        ball_csv = os.path.join(tmp, "ball.csv")
        pd.DataFrame({"frame": frames,
                      "x": cx - 5, "y": cy - 5, "w": 10, "h": 10,
                      "cx": cx, "cy": cy, "area": 100}).to_csv(
            ball_csv, index=False)
        players_csv = os.path.join(tmp, "players.csv")
        rows = []
        for f in frames:
            for pid, box in ((1, p1), (2, p2)):
                x, y, w, h = box
                rows.append([f, pid, x, y, w, h, x + w / 2, y + h / 2, w * h])
        pd.DataFrame(rows, columns=["frame", "player_id", "x", "y", "w", "h",
                                    "cx", "cy", "area"]).to_csv(
            players_csv, index=False)
        court_csv = os.path.join(tmp, "court.csv")
        _synthetic_court_csv(court_csv)

        track = load_ball_track(ball_csv)
        boxes = load_player_boxes(players_csv)
        conv = CourtConverter(court_csv)

        failures = []
        for hands, expect_fn in (
            ({1: "right", 2: "right"}, lambda _, exp: exp),
            ({1: "right", 2: "left"},
             lambda pid, exp: exp if pid == 1 else
             ("backhand" if exp == "forehand" else "forehand")),
        ):
            shots = analyze_shots(track, boxes, conv, fps=30.0, hands=hands)
            tag = f"hands={hands}"
            if len(shots) != len(plan):
                failures.append(f"{tag}: attesi {len(plan)} colpi, "
                                f"rilevati {len(shots)}")
                continue
            for (f_exp, pid_exp, _, stroke_exp), r in zip(
                    plan, shots.itertuples()):
                stroke_exp = expect_fn(pid_exp, stroke_exp)
                if abs(r.frame - f_exp) > 4:
                    failures.append(f"{tag}: colpo a frame {r.frame}, "
                                    f"atteso ~{f_exp}")
                if r.player_id != pid_exp:
                    failures.append(f"{tag}: frame {r.frame} assegnato a "
                                    f"P{r.player_id}, atteso P{pid_exp}")
                if r.stroke != stroke_exp:
                    failures.append(f"{tag}: frame {r.frame} P{pid_exp} "
                                    f"{r.stroke}, atteso {stroke_exp}")

    print("SELF-TEST shot_analysis")
    print(f"  colpi pianificati : {[(f, p, s) for f, p, _, s in plan]}")
    if failures:
        print("  FAIL:")
        for msg in failures:
            print("   -", msg)
        raise SystemExit(1)
    print("  PASS: 4/4 colpi rilevati e classificati correttamente "
          "(anche nel caso mancino); rimbalzi a meta' campo ignorati.")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rilevazione colpi e classificazione dritto/rovescio "
                    "dal tracking della palla")
    parser.add_argument("--ball", default="outputs/ball_clip2.csv",
                        help="CSV palla generato da BallTracking.py")
    parser.add_argument("--players", default="outputs/players_clip2.csv")
    parser.add_argument("--court",
                        default="outputs/court_coordinates/Input_video2_court.csv")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="video sorgente per i PNG di verifica")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--p1-hand", choices=["right", "left"],
                        default="right", dest="p1_hand",
                        help="mano dominante del giocatore 1 (default right)")
    parser.add_argument("--p2-hand", choices=["right", "left"],
                        default="right", dest="p2_hand",
                        help="mano dominante del giocatore 2 (default right)")
    parser.add_argument("--min-gap", type=float, default=0.5, dest="min_gap",
                        help="distanza minima in secondi tra due colpi")
    parser.add_argument("--output", default="outputs/shot_analysis")
    parser.add_argument("--no-frames", action="store_true",
                        help="non salvare i PNG dei colpi")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="valida la logica su una traiettoria sintetica")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not os.path.exists(args.ball):
        raise SystemExit(
            f"Ball CSV non trovato: {args.ball}\n"
            "Generalo prima con il ball tracker (serve il modello YOLO):\n"
            "    python ballTracking/BallTracking.py")

    hands = {1: args.p1_hand, 2: args.p2_hand}
    track = load_ball_track(args.ball)
    boxes = load_player_boxes(args.players)
    conv = CourtConverter(args.court)

    shots = analyze_shots(track, boxes, conv, args.fps, hands,
                          min_gap_s=args.min_gap)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "shots.csv"
    shots.to_csv(csv_path, index=False)
    print(f"Salvato: {csv_path}  ({len(shots)} colpi)")

    if not args.no_frames and not shots.empty:
        save_shot_frames(shots, args.video, boxes, out_dir)

    print_summary(shots, hands)


if __name__ == "__main__":
    main()
