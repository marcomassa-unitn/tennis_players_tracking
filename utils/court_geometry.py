"""
utils/court_geometry.py

Single source of truth for the ITF singles-court dimensions in METERS, shared by
the analysis utilities (court_converter, player_analysis, shot_analysis) that all
previously redefined these same constants byte-for-byte.

Origin convention: TL = top-left corner of the FAR baseline (matches the labelled
court CSV produced by tracking/court_tracking.py and consumed by CourtConverter).
  x : 0 (left sideline)  ->  W_m (right sideline)
  y : 0 (far baseline)   ->  L_m (near baseline)

NOTE: tracking/court_tracking.py keeps its OWN ``REAL_FT`` table in FEET with a
different (bottom-left) origin/orientation; that is a separate convention used
only for the detection-time homography and is intentionally NOT shared here.

The arithmetic expression order below is preserved exactly as it was at each
former definition site so the resulting float values stay bit-identical.
"""

_FT = 0.3048

W_m = 27.0 * _FT          # 8.2296 m  width (singles)
L_m = 78.0 * _FT          # 23.7744 m length
SVC_T = 18.0 * _FT        # 5.4864 m  service line far side
SVC_B = L_m - SVC_T       # 18.288 m  service line near side
NET = L_m / 2.0           # 11.8872 m net
CL_X = W_m / 2.0          # 4.1148 m  center service line

# Real-world ITF positions in meters for each labelled corner (TL origin).
# Values are bit-identical to the former court_converter._REAL_WORLD literals.
_REAL_WORLD = {
    "TL":  (0.0,  0.0),
    "TR":  (W_m,  0.0),
    "BL":  (0.0,  L_m),
    "BR":  (W_m,  L_m),
    "STL": (0.0,  SVC_T),
    "STR": (W_m,  SVC_T),
    "SBL": (0.0,  SVC_B),
    "SBR": (W_m,  SVC_B),
}
