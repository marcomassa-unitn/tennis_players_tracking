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
# RENDER vs WINDOW size are DECOUPLED. The whole canvas is composed at RENDER_WIDTH
# (a fixed, generous resolution) so all text — on-video labels AND the panel — is
# rasterised large and stays sharp. Only at the very end is the finished canvas
# area-downscaled ONCE to WINDOW_WIDTH for display. Making the window smaller then
# never costs text quality (it shrinks crisp pixels with INTER_AREA); it used to,
# because the old single --max-width drove both the render resolution and the
# window, so a smaller window literally rendered fewer pixels of text.
RENDER_WIDTH = 1600      # px width the canvas is COMPOSED at (quality; not the window)
WINDOW_WIDTH = 1100      # px width the window is SHOWN at (display size; <= RENDER_WIDTH)
PANEL_W   = 380          # width (px) of the right column (minimap + stats)
MINI_FRAC = 0.50         # fraction of the right column height given to the minimap
                         # (stats panel gets the other half; it needs the room for
                         #  8 category rows + totals + footer without crowding)
FONT      = cv2.FONT_HERSHEY_SIMPLEX
# Heavier (double-stroke) font for the on-VIDEO player labels: they sit over a busy
# court at a small on-screen size, where SIMPLEX reads thin. The panel/minimap keep
# SIMPLEX (denser text on a controlled dark background).
FONT_VIDEO = cv2.FONT_HERSHEY_DUPLEX

# Supersample factor for the side panels (stats + minimap). They are rendered at
# SS× their final size and downscaled with INTER_AREA in compose(), so the thin
# Hershey strokes come out smoothly anti-aliased instead of spindly. 4× the pixel
# work on a ~380-px column is sub-millisecond, so it never affects playback.
SS = 2

# On-video label style (drawn at DISPLAY resolution, after the frame is resized,
# so strokes are crisp). Each label gets a dark semi-transparent plate behind it
# so the coloured text stays legible over the court.
_LBL_SCALE_PLAYER = 0.6   # font scale for the "P1"/"P2" tags
_LBL_TH           = 2     # stroke thickness (true 2 px at display resolution)
_LBL_PAD          = 4     # padding (px) inside the plate around the glyphs
_LBL_BG           = (0, 0, 0)   # plate colour (BGR)
_LBL_BG_ALPHA     = 0.55  # plate opacity (0=invisible, 1=solid)

# Panel text tints (brighter than the pure on-video box colours, which have low
# luminance on the dark warm panel background). Used for panel/legend TEXT only;
# the on-video boxes keep the pure _BOX_COLORS.
_TINT = {1: (90, 255, 90), 2: (90, 90, 255)}   # P1 light-green, P2 light-red (BGR)
_PANEL_BG   = (40, 30, 25)     # stats panel background (BGR)
_RULE_BGR   = (70, 60, 55)     # subtle separator-line colour (BGR)
_PANEL_FG   = (230, 230, 230)  # default near-white panel text

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
        # Supersample scale: the minimap is built at SS× its on-screen width, so
        # marker/dot radii (and the legend) scale by this to keep their on-screen
        # size constant after the INTER_AREA downscale in compose().
        self._s = self.w / PANEL_W
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
        far (hollow colour swatch + near-WHITE label, so dark/grey hues stay
        legible), in the stable _LEG_ORDER, then a P1=circle / P2=triangle shape
        key. Drawn over a dark backdrop so the text reads against the green court.

        The minimap is rendered at SS× its on-screen size, so all geometry/fonts
        scale by ``s`` (derived from the panel width) and the glyphs come out
        smooth after the INTER_AREA downscale in compose().
        """
        s = self.w / PANEL_W
        def S(v): return int(round(v * s))
        leg_fs = _LEG_FS * 1.2 * s        # a touch larger than before, then *SS
        row_h = S(_LEG_ROW_H + 3)
        leg_x, leg_top, leg_gap = S(_LEG_X), S(_LEG_TOP), S(_LEG_GAP)
        th = max(2, S(1.5))               # heavier strokes (thickness-1 looked thin)

        rows = [c for c in _LEG_ORDER if c in present_cats]
        n = len(rows) + 2 + 1   # categories + 2 shape-key rows + 1 header row
        # Backdrop sized to the column; clipped to the panel just in case.
        x0, y0 = leg_x - S(3), leg_top - S(12)
        x1 = x0 + S(112)
        y1 = y0 + n * row_h + S(4)
        x1, y1 = min(x1, self.w - 1), min(y1, self.h - 1)
        ov = img[y0:y1, x0:x1].copy()
        img[y0:y1, x0:x1] = cv2.addWeighted(
            ov, 0.35, np.zeros_like(ov), 0.0, 0.0)   # darken 65% for contrast

        sym_r = S(6)   # legend swatch radius
        lbl_x = leg_x + 2 * sym_r + leg_gap
        y = leg_top
        cv2.putText(img, "Shots", (leg_x, y), FONT, leg_fs,
                    _PANEL_FG, th, cv2.LINE_AA)
        y += row_h
        for cat in rows:
            color = _CAT_BGR.get(cat, _CAT_FALLBACK_BGR)
            cv2.circle(img, (leg_x + sym_r, y - S(4)), sym_r, color, th, cv2.LINE_AA)
            cv2.putText(img, cat, (lbl_x, y), FONT, leg_fs, _PANEL_FG, th,
                        cv2.LINE_AA)
            y += row_h
        # player shape key (neutral white so only the shape reads)
        for pid, lbl in ((1, "P1"), (2, "P2")):
            _draw_shot_marker(img, (leg_x + sym_r, y - S(4)), pid,
                              _PANEL_FG, r=sym_r, th=max(1, S(1)))
            cv2.putText(img, lbl, (lbl_x, y), FONT, leg_fs, _PANEL_FG, th,
                        cv2.LINE_AA)
            y += row_h

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
        # Scale marker/dot geometry to the SS× canvas so they keep their on-screen
        # size after the downscale in compose().
        s = self._s
        mk_r, mk_th = int(round(_SHOT_MARKER_R * s)), max(1, int(round(_SHOT_MARKER_TH * s)))
        dot_r = int(round(6 * s))
        if shot_markers:
            for pid, cat, center in shot_markers:
                _draw_shot_marker(img, center, pid,
                                  _CAT_BGR.get(cat, _CAT_FALLBACK_BGR),
                                  r=mk_r, th=mk_th)
        for pid, (xm, ym) in positions.items():
            cv2.circle(img, self._to_px(xm, ym), dot_r,
                       _DOT_BGR.get(pid, (255, 255, 255)), -1, cv2.LINE_AA)
        if present_cats:
            self._draw_legend(img, present_cats)
        return img


# ── stats placeholder ───────────────────────────────────────────────────────────

def render_stats_panel(size: tuple[int, int], counts: dict,
                       cats: list, ended: bool = False) -> np.ndarray:
    """Live per-player shot tally, two columns (P1 | P2).

    ``counts`` = {pid: {category: n}} updated as shots are registered; ``cats``
    is the fixed ordered category list (so rows never reshuffle). Each column is
    headed by the player in their on-video box colour (P1 green, P2 red) and
    lists one counter per shot category, then the per-player total. The footer
    shows the rally length = the TOTAL shots played by both players; when
    ``ended`` is True it also marks the clip as finished (the playback loop
    freezes on the last frame).
    """
    w, h = size
    # The panel is rendered at SS× its on-screen size (then area-downscaled in
    # compose), so derive a scale from the incoming width and express all sizes in
    # NATIVE units * s. This keeps the layout readable in native terms and makes
    # the panel resolution-agnostic. Category labels are drawn in near-white with
    # the colour carried ONLY in the swatch, so every row clears a contrast floor
    # regardless of hue; body strokes are thickness 2 (spindly thickness-1 was a
    # big part of the "low quality" look).
    panel = np.full((h, w, 3), _PANEL_BG, np.uint8)
    s = w / PANEL_W                                   # native -> render scale
    def S(v): return int(round(v * s))                # scale a length
    def fs(v): return v * s                           # scale a font size
    th_body = max(1, S(1))                            # thickness ~2 at SS=2
    th_bold = max(2, S(1.5))

    # Title band + a separator rule beneath it.
    title = "FINAL STATS" if ended else "SHOT STATS"
    cv2.putText(panel, title, (S(16), S(28)), FONT, fs(0.7), _PANEL_FG,
                th_bold, cv2.LINE_AA)
    cv2.line(panel, (S(12), S(40)), (w - S(12), S(40)), _RULE_BGR, max(1, S(1)),
             cv2.LINE_AA)

    # Two equal columns, headed by each player in their (brightened) tint, with a
    # vertical rule between them.
    col_x = {1: S(16), 2: w // 2 + S(10)}
    head_y = S(62)
    for pid in (1, 2):
        cv2.putText(panel, f"P{pid}", (col_x[pid], head_y), FONT, fs(0.66),
                    _TINT.get(pid, _PANEL_FG), th_bold, cv2.LINE_AA)

    # One counter row per category + a total row. The pitch is FIT-DRIVEN: it
    # divides the space between the header and the footer band by the row count so
    # the rows + total ALWAYS sit above the footer (no overlap), clamped to a
    # readable max. The row font tracks the pitch (so a short panel shrinks text a
    # touch rather than colliding) with a legible floor.
    top = S(86)                                  # first counter baseline
    foot = S(48)                                 # footer band reserved at bottom
    n_rows = len(cats) + 1                        # categories + the "total" row
    # Distribute the available height across all rows INCLUDING the total, so the
    # total baseline always lands above the footer rule.
    avail = (h - foot) - top
    pitch = int(np.clip(avail / max(1, n_rows), S(13), S(24)))
    row_fs = float(np.clip(pitch / s / 24.0 * 0.5, 0.40, 0.52))  # native font, tracks pitch
    cv2.line(panel, (w // 2, S(70)), (w // 2, h - foot), _RULE_BGR,
             max(1, S(1)), cv2.LINE_AA)

    y = top
    for cat in cats:
        color = _CAT_BGR.get(cat, _CAT_FALLBACK_BGR)
        for pid in (1, 2):
            n = counts.get(pid, {}).get(cat, 0)
            cv2.circle(panel, (col_x[pid] + S(6), y - S(5)), max(2, S(5)), color,
                       th_body, cv2.LINE_AA)
            cv2.putText(panel, f"{cat}: {n}", (col_x[pid] + S(18), y), FONT,
                        fs(row_fs), _PANEL_FG, th_body, cv2.LINE_AA)
        y += pitch

    # per-player total flows directly after the last category (the pitch reserved a
    # slot for it via n_rows), then a thin rule separates it from the footer band.
    cv2.line(panel, (S(12), y - pitch + S(4)), (w - S(12), y - pitch + S(4)),
             _RULE_BGR, max(1, S(1)), cv2.LINE_AA)
    for pid in (1, 2):
        total = sum(counts.get(pid, {}).values())
        cv2.putText(panel, f"total: {total}", (col_x[pid], y), FONT,
                    fs(row_fs + 0.02), _TINT.get(pid, _PANEL_FG), th_bold,
                    cv2.LINE_AA)

    # footer: rally length = total shots played by BOTH players
    cv2.line(panel, (S(12), h - foot + S(4)), (w - S(12), h - foot + S(4)),
             _RULE_BGR, max(1, S(1)), cv2.LINE_AA)
    rally_total = sum(sum(counts.get(pid, {}).values()) for pid in (1, 2))
    cv2.putText(panel, f"Rally length: {rally_total} shots",
                (S(16), h - S(22)), FONT, fs(0.52), (200, 220, 255), th_body,
                cv2.LINE_AA)
    if ended:
        cv2.putText(panel, "clip ended - press q to quit",
                    (S(16), h - S(6)), FONT, fs(0.46), (160, 200, 160),
                    th_body, cv2.LINE_AA)
    return panel




# ── compositing ─────────────────────────────────────────────────────────────────

def draw_label(img: np.ndarray, text: str, anchor: tuple[int, int],
               color: tuple, scale: float = _LBL_SCALE_PLAYER,
               thickness: int = _LBL_TH, pad: int = _LBL_PAD,
               above: bool = True) -> None:
    """Draw ``text`` with a dark semi-transparent plate behind it for contrast.

    Crisp because it is called on the ALREADY-RESIZED display-resolution image,
    so the glyphs are rasterised once at the final pixel size (no later
    downscale to soften them). ``anchor`` is a point on the bbox edge: with
    ``above=True`` the plate sits just above it (so it frames a box top-left).
    The plate is clamped to the image so a label near an edge never indexes out
    of range. Drawn in place.
    """
    (tw, th), base = cv2.getTextSize(text, FONT_VIDEO, scale, thickness)
    pw, ph = tw + 2 * pad, th + base + 2 * pad
    ax, ay = anchor
    x0 = int(np.clip(ax, 0, max(0, img.shape[1] - pw)))
    if above:
        y0 = int(np.clip(ay - ph - 2, 0, max(0, img.shape[0] - ph)))
    else:
        y0 = int(np.clip(ay, 0, max(0, img.shape[0] - ph)))
    x1, y1 = x0 + pw, y0 + ph
    roi = img[y0:y1, x0:x1]
    if roi.size:   # blend a dark plate (same addWeighted idiom as the legend)
        plate = np.full_like(roi, _LBL_BG)
        img[y0:y1, x0:x1] = cv2.addWeighted(
            roi, 1.0 - _LBL_BG_ALPHA, plate, _LBL_BG_ALPHA, 0.0)
    cv2.putText(img, text, (x0 + pad, y1 - pad - base), FONT_VIDEO, scale,
                color, thickness, cv2.LINE_AA)


def compose(frame: np.ndarray, minimap: np.ndarray, stats: np.ndarray,
            disp_h: int, panel_w: int, mini_h: int,
            player_overlays=(), ball_overlay=None) -> np.ndarray:
    """Assemble [ video | (minimap / stats) ] into one canvas of height disp_h.

    The player/ball overlays are drawn HERE, on the resized display-resolution
    video — NOT on the source frame — so the player-label text strokes are crisp
    at the final size instead of being softened by the source→display downscale.
    ``player_overlays`` = [(pid, x, y, w, h, color), ...] (box + "P#" label) and
    ``ball_overlay`` = (x, y, w, h) (box only, no label) in SOURCE-pixel coords;
    they are scaled by the same factor the frame is resized with.

    The shot legend lives INSIDE the minimap (see Minimap._draw_legend), so there
    is no separate strip and the canvas height equals disp_h exactly. The minimap
    and stats panels arrive at SS× their final size and are area-downscaled here
    (smooth anti-aliasing for their thin text).
    """
    h0, w0 = frame.shape[:2]
    vw = int(round(w0 * (disp_h / h0)))
    # INTER_AREA is the correct, alias-free filter for this downscale (the frame
    # shrinks ~0.5×); INTER_LINEAR softened the court and box edges.
    video = cv2.resize(frame, (vw, disp_h), interpolation=cv2.INTER_AREA)

    # Overlays at DISPLAY resolution: scale source-pixel coords to the resized
    # video, then draw crisp boxes + plated labels.
    sx, sy = vw / w0, disp_h / h0
    for pid, x, y, w, h, color in player_overlays:
        X, Y, W, H = (int(round(x * sx)), int(round(y * sy)),
                      int(round(w * sx)), int(round(h * sy)))
        cv2.rectangle(video, (X, Y), (X + W, Y + H), color, 2, cv2.LINE_AA)
        draw_label(video, f"P{pid}", (X, Y), color, _LBL_SCALE_PLAYER)
    if ball_overlay is not None:
        x, y, w, h = ball_overlay
        X, Y, W, H = (int(round(x * sx)), int(round(y * sy)),
                      int(round(w * sx)), int(round(h * sy)))
        cv2.rectangle(video, (X, Y), (X + W, Y + H), (0, 255, 255), 2, cv2.LINE_AA)

    canvas = np.zeros((disp_h, vw + panel_w, 3), np.uint8)
    canvas[:, :vw] = video
    # Area-downscale the SS× panels to their exact sub-regions (also guarantees
    # assembly never shape-mismatches).
    canvas[0:mini_h, vw:vw + panel_w]      = cv2.resize(
        minimap, (panel_w, mini_h), interpolation=cv2.INTER_AREA)
    canvas[mini_h:disp_h, vw:vw + panel_w] = cv2.resize(
        stats, (panel_w, disp_h - mini_h), interpolation=cv2.INTER_AREA)
    return canvas


def _fit_to_screen(canvas: np.ndarray, window_width: int) -> np.ndarray:
    """Downscale the WHOLE finished canvas (preserving aspect) to the on-screen
    ``window_width``. This is the ONE place render resolution becomes window size.

    The canvas is composed at RENDER_WIDTH (high, for sharp text); this single
    INTER_AREA step shrinks those already-crisp pixels to the smaller window, so a
    smaller window costs no text quality. A no-op when the canvas is already <=
    window_width (e.g. window_width >= render_width); ``window_width<=0`` disables
    scaling entirely (show at full render resolution).
    """
    if window_width and window_width > 0 and canvas.shape[1] > window_width:
        scale = window_width / canvas.shape[1]
        new_h = int(round(canvas.shape[0] * scale))
        return cv2.resize(canvas, (window_width, new_h),
                          interpolation=cv2.INTER_AREA)
    return canvas


def _screen_size(default=(1920, 1080)) -> tuple[int, int]:
    """(width, height) of the primary screen in px; safe fallback if unknown."""
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        w, h = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        pass
    return default


def _center_window(win_name: str, w: int, h: int) -> None:
    """Move an OpenCV window so its w×h box is centred on the primary screen."""
    sw, sh = _screen_size()
    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 2)
    cv2.moveWindow(win_name, x, y)


# ── on-launch analysis generation ────────────────────────────────────────────────

def _generate_analysis_figures(players_csv, court_csv, ball_csv, output_dir,
                               fps, min_area, anchor):
    """Regenerate the three end-of-clip summary figures in-process, headless:
    heatmap_combined.png + zones.png (player_analysis) and shot_hitmap.png
    (shot_analysis). Only the FAST figures are produced — the slow minimap.gif and
    the per-shot frame PNGs are intentionally skipped.

    Both analysis modules use matplotlib's Agg backend, so nothing pops up. Raises
    on any failure so the caller can abort (the figures are required for the
    end-of-clip summary).
    """
    from pathlib import Path
    from utils import player_analysis as pa
    from utils import shot_analysis as sa
    from utils.court_converter import CourtConverter

    fps = fps or 30.0
    hand_map = {1: "right", 2: "right"}

    # --- player analysis: combined heatmap + court zones (skip the slow gif) ---
    pa_dir = Path(output_dir) / "player_analysis"
    pa_dir.mkdir(parents=True, exist_ok=True)
    player_data = pa._load_and_convert(players_csv, court_csv, min_area, anchor)
    if not player_data:
        raise RuntimeError("player analysis produced no player tracks.")
    pa._save_heatmaps(player_data, pa_dir)                 # -> heatmap_combined.png
    zone_df = pa._compute_zone_stats(player_data, fps)
    pa._save_zone_outputs(player_data, zone_df, pa_dir)    # -> zones.png

    # --- shot analysis: shot hitmap (needs the ball CSV) ---
    if not (ball_csv and os.path.exists(ball_csv)):
        raise RuntimeError(
            f"ball CSV not found ({ball_csv}); the shot hitmap can't be built.\n"
            "Run 'python tracking/BallTracking.py' first, or launch without the "
            "summary by removing this requirement.")
    sa_dir = Path(output_dir) / "shot_analysis"
    sa_dir.mkdir(parents=True, exist_ok=True)
    track = sa.load_ball_track(ball_csv)
    boxes = sa.load_player_boxes(players_csv)
    conv = CourtConverter(court_csv)
    shots = sa.analyze_shots(track, boxes, conv, fps, hand_map)
    shots.to_csv(sa_dir / "shots.csv", index=False)
    sa.save_shot_hitmap(shots, sa_dir)                     # -> shot_hitmap.png


# ── end-of-clip summary window ───────────────────────────────────────────────────

# The three analysis figures shown together when the clip ends, with their fixed
# paths relative to the --output base dir (see player_analysis.py / shot_analysis.py).
_SUMMARY_FIGS = [
    ("Combined heatmap", os.path.join("player_analysis", "heatmap_combined.png")),
    ("Court zones",      os.path.join("player_analysis", "zones.png")),
    ("Shot hitmap",      os.path.join("shot_analysis",   "shot_hitmap.png")),
]
_SUMMARY_BG = (26, 26, 46)   # dark indigo, matches the analysis figures' bg (BGR)


def _summary_panel(title: str, img, panel_h: int, bar_h: int = 34) -> np.ndarray:
    """One labelled column of the summary window: a title bar over the figure
    (or a 'not found' placeholder), scaled to ``panel_h`` total height."""
    fig_h = panel_h - bar_h
    if img is None:                      # missing/unreadable -> placeholder column
        w = int(round(fig_h * 0.45))
        col = np.full((panel_h, w, 3), _SUMMARY_BG, np.uint8)
        cv2.putText(col, "not found", (10, panel_h // 2), FONT, 0.6,
                    (120, 120, 160), 1, cv2.LINE_AA)
    else:
        h0, w0 = img.shape[:2]
        w = max(1, int(round(w0 * fig_h / h0)))
        fig = cv2.resize(img, (w, fig_h), interpolation=cv2.INTER_AREA)
        col = np.full((panel_h, w, 3), _SUMMARY_BG, np.uint8)
        col[bar_h:bar_h + fig_h, :] = fig
    # title bar
    cv2.putText(col, title, (10, 23), FONT, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
    return col


def _show_summary_window(output_dir: str, window_name: str,
                         max_width: int = 1360, panel_h: int = 660) -> bool:
    """Open a SEPARATE, screen-centred window tiling the combined heatmap, the
    zones map and the shot hitmap side by side, so the whole-clip stats can be
    studied at the end.

    Returns True if at least one figure was found/shown. Missing figures become a
    labelled placeholder rather than an error, so a partial pipeline still works.
    """
    cols, found = [], False
    for title, rel in _SUMMARY_FIGS:
        path = os.path.join(output_dir, rel)
        img = cv2.imread(path) if os.path.exists(path) else None
        if img is not None:
            found = True
        else:
            print(f"  Summary: '{title}' not found at {path} (skipped).")
        cols.append(_summary_panel(title, img, panel_h))

    gap = 12
    total_w = sum(c.shape[1] for c in cols) + gap * (len(cols) + 1)
    board = np.full((panel_h + 2 * gap, total_w, 3), _SUMMARY_BG, np.uint8)
    x = gap
    for c in cols:
        board[gap:gap + panel_h, x:x + c.shape[1]] = c
        x += c.shape[1] + gap

    if max_width and board.shape[1] > max_width:     # fit to a sane on-screen width
        scale = max_width / board.shape[1]
        board = cv2.resize(board, (max_width, int(round(board.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, board.shape[1], board.shape[0])
    cv2.imshow(window_name, board)
    _center_window(window_name, board.shape[1], board.shape[0])
    return found


# ── playback ────────────────────────────────────────────────────────────────────

def run(video, players_csv, court_csv, ball_csv, min_area, anchor,
        disp_h, fps_override, shots_csv=None, max_shot_markers=0,
        render_width=RENDER_WIDTH, window_width=WINDOW_WIDTH,
        output_dir="outputs") -> None:
    # Solve the RENDER height up front so the composed canvas width (video + the
    # PANEL_W stats column) equals render_width — a fixed, generous resolution that
    # keeps all text sharp. The window the user sees is a SEPARATE, smaller size:
    # the finished canvas is area-downscaled to window_width ONCE at display time,
    # so shrinking the window never re-rasterises (and thus never softens) the text.
    # The video keeps its true aspect ratio: vw + PANEL_W = render_width,
    # vw = src_w*disp_h/src_h  =>  disp_h = (render_width - PANEL_W) * src_h / src_w.
    probe = cv2.VideoCapture(video)
    src_w = probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
    src_h = probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0
    probe.release()
    if render_width and render_width > PANEL_W + 80 and src_w > 0:
        disp_h = int(round((render_width - PANEL_W) * src_h / src_w))

    # --- regenerate the end-of-clip summary figures up front, headless, so they
    # always reflect the CURRENT CSVs (no manual run of the analysis scripts). The
    # slow minimap.gif / per-shot PNGs are skipped; only the 3 fast figures are
    # built. Abort on failure — the summary requires them. ---
    print("  Generating analysis figures (heatmap, zones, shot hitmap)...")
    try:
        _generate_analysis_figures(players_csv, court_csv, ball_csv, output_dir,
                                   fps_override or 30.0, min_area, anchor)
    except Exception as e:
        raise SystemExit(
            f"Analysis generation failed: {e}\n"
            "Fix the inputs above and relaunch (the end-of-clip summary needs "
            "the combined heatmap, zones and shot hitmap).")
    print("  Analysis figures ready.")

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
    # The minimap is built at SS× its final size so its court lines + legend text
    # come out smooth after the INTER_AREA downscale in compose(). _to_px (and the
    # shot markers pre-resolved through it) inherit the SS× scale automatically, so
    # everything stays self-consistent.
    minimap = Minimap((PANEL_W * SS, mini_h * SS))

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
    # Ordered category list for the stats columns (stable, only types that occur).
    stat_cats = [c for c in _LEG_ORDER if legend_cats and c in legend_cats]

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
    # Resizable window opened at the chosen (small) window_width; the frames pushed
    # to it are already downscaled to that width, so what's shown is 1:1 crisp.
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if window_width and window_width > 0:
        # Open the window at window_width, preserving the canvas aspect ratio
        # (the canvas is vw + PANEL_W wide by disp_h tall), centred on the screen.
        canvas_w = int(round(src_w * disp_h / src_h)) + PANEL_W
        win_h = int(round(disp_h * window_width / canvas_w))
        cv2.resizeWindow(win, window_width, win_h)
        _center_window(win, window_width, win_h)
    print(f"  Render @ {render_width}px wide → window @ {window_width}px "
          f"(text rendered high, shown small).")
    print(f"  Playing at {fps:.0f} fps — press 'q' in the window to quit.")
    idx = 0
    # Shot accumulation: shot_markers is sorted by frame, so a single advancing
    # pointer reveals each marker once its frame is reached and keeps it shown
    # for the rest of the clip (O(total shots) overall, not per frame).
    active_markers = []     # [(pid, cat, center), ...] — frames already crossed
    next_shot = 0
    # Live per-player tally, incremented exactly when a shot is revealed so the
    # stats panel counter for each category stays in step with the markers.
    counts = {1: {}, 2: {}}
    last_canvas = None      # keep the final composite to freeze on after the clip
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            # Collect overlays in SOURCE-pixel coords; compose() draws them on the
            # resized display-resolution video so the text comes out crisp.
            player_overlays = []
            for pid, box in boxes.get(idx, {}).items():
                x, y, w, h = (int(v) for v in box)
                color = _BOX_COLORS.get(pid, (0, 255, 0))
                player_overlays.append((pid, x, y, w, h, color))

            # Ball box (yellow), drawn without a label.
            ball = ball_boxes.get(idx)
            ball_overlay = ball[:4] if ball is not None else None

            # Reveal every shot whose frame has been reached; it then persists.
            # Each reveal bumps that player's per-category counter so the stats
            # panel updates exactly when its marker appears.
            while next_shot < len(shot_markers) and shot_markers[next_shot][0] <= idx:
                _f, _pid, _cat, _center = shot_markers[next_shot]
                active_markers.append((_pid, _cat, _center))
                pc = counts.setdefault(_pid, {})
                pc[_cat] = pc.get(_cat, 0) + 1
                next_shot += 1
            # Optional clutter cap: keep only the most recent N markers (0 = all).
            shown = (active_markers[-max_shot_markers:]
                     if max_shot_markers else active_markers)

            # Legend is FIXED (full key from the first frame), not built up live.
            # Panels are rendered at SS× and area-downscaled in compose() for
            # smoothly anti-aliased text.
            mini = minimap.render(minimap_positions(idx), shown, legend_cats)
            stats = render_stats_panel((PANEL_W * SS, (disp_h - mini_h) * SS),
                                       counts, stat_cats)
            last_canvas = compose(frame, mini, stats, disp_h, PANEL_W, mini_h,
                                  player_overlays, ball_overlay)
            # Downscale the high-res canvas to the (smaller) window size ONCE here.
            cv2.imshow(win, _fit_to_screen(last_canvas, window_width))

            # Wait only the time LEFT in this frame's period after the per-frame
            # processing, so render overhead doesn't slow playback below real time.
            wait_ms = max(1, int(round((frame_period - (time.perf_counter() - t0)) * 1000)))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
            idx += 1

        # --- clip finished: FREEZE on the last frame so the final statistics can
        # be studied. Recompose the last frame with the full final tally (total
        # shots + per-player totals) and block until the user presses 'q'. ---
        if last_canvas is not None:
            mini = minimap.render(minimap_positions(max(idx - 1, 0)),
                                  (active_markers[-max_shot_markers:]
                                   if max_shot_markers else active_markers),
                                  legend_cats)
            stats = render_stats_panel((PANEL_W * SS, (disp_h - mini_h) * SS),
                                       counts, stat_cats, ended=True)
            # reuse the last decoded video frame held in `last_canvas`'s left part;
            # area-downscale the SS× panels (same as compose) for crisp text.
            final = last_canvas.copy()
            final[0:mini_h, -PANEL_W:] = cv2.resize(
                mini, (PANEL_W, mini_h), interpolation=cv2.INTER_AREA)
            final[mini_h:disp_h, -PANEL_W:] = cv2.resize(
                stats, (PANEL_W, disp_h - mini_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, _fit_to_screen(final, window_width))
            # Separate window with the whole-clip analysis figures (combined
            # heatmap + court zones + shot hitmap) so the totals can be studied.
            summary_win = "Clip summary  (combined heatmap | zones | shot hitmap)"
            if _show_summary_window(output_dir, summary_win):
                print(f"  Summary window: combined heatmap + zones + shot hitmap "
                      f"(from {output_dir}).")
            print("  Clip finished — frozen on last frame; press 'q' to quit.")
            while True:
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
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
    parser.add_argument("--window-width", type=int, default=WINDOW_WIDTH,
                        dest="window_width",
                        help="on-screen WIDTH (px) of the window. Smaller = smaller "
                             "window; text stays sharp because it is rendered at "
                             "--render-width and downscaled once for display "
                             f"(0 = show at full render width, default: {WINDOW_WIDTH})")
    parser.add_argument("--render-width", "--max-width", type=int,
                        default=RENDER_WIDTH, dest="render_width",
                        help="internal RENDER width (px) the canvas is composed at "
                             "— the text-quality knob; higher = crisper text, more "
                             "CPU. Independent of the window size "
                             f"(default: {RENDER_WIDTH}). --max-width is a legacy alias.")
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
        shots_csv, args.max_shot_markers, args.render_width, args.window_width,
        args.output)


if __name__ == "__main__":
    main()
