import pytest

from utils import court_geometry as g


def test_itf_singles_dimensions():
    assert g.W_m == pytest.approx(8.23, abs=0.01)
    assert g.L_m == pytest.approx(23.77, abs=0.01)
    assert g.SVC_T == pytest.approx(5.49, abs=0.01)


def test_derived_lines_are_symmetric():
    assert g.NET == pytest.approx(g.L_m / 2)
    assert g.CL_X == pytest.approx(g.W_m / 2)
    # Both service lines sit 6.40 m from the net.
    assert g.NET - g.SVC_T == pytest.approx(g.SVC_B - g.NET)


def test_labelled_corners():
    corners = g._REAL_WORLD
    assert set(corners) == {"TL", "TR", "BL", "BR", "STL", "STR", "SBL", "SBR"}
    assert corners["TL"] == (0.0, 0.0)
    assert corners["BR"] == pytest.approx((g.W_m, g.L_m))
    # Left-side labels share x = 0, right-side labels share x = W.
    for label in ("TL", "BL", "STL", "SBL"):
        assert corners[label][0] == 0.0
    for label in ("TR", "BR", "STR", "SBR"):
        assert corners[label][0] == pytest.approx(g.W_m)
