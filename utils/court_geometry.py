"""ITF singles-court dimensions in METERS, shared by the analysis utilities.

Single source of truth for court_converter, player_analysis and shot_analysis,
which previously each redefined these constants byte-for-byte.

Origin: TL = top-left corner of the FAR baseline (matches the labelled court CSV
from tracking/court_tracking.py, consumed by CourtConverter).
  x : 0 (left sideline)  ->  W_m (right sideline)
  y : 0 (far baseline)   ->  L_m (near baseline)

tracking/court_tracking.py keeps its own ``REAL_FT`` table in FEET with a
bottom-left origin -- a separate detection-time-homography convention, not shared.

Expression order is preserved verbatim from each former definition site so the
float values stay bit-identical.
"""

_FT = 0.3048

W_m = 27.0 * _FT          # 8.2296 m  width (singles)
L_m = 78.0 * _FT          # 23.7744 m length
SVC_T = 18.0 * _FT        # 5.4864 m  service line far side
SVC_B = L_m - SVC_T       # 18.288 m  service line near side
NET = L_m / 2.0           # 11.8872 m net
CL_X = W_m / 2.0          # 4.1148 m  center service line

# Labelled-corner positions in meters, TL origin (baseline + service-box corners).
# Bit-identical to the former court_converter._REAL_WORLD literals.
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
