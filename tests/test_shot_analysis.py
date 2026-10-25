"""Run the synthetic self-tests shipped with the shot classifier.

Each one raises SystemExit on failure, so a clean return means it passed.
"""

from analysis import shot_analysis


def test_stroke_self_test():
    shot_analysis.self_test()


def test_overhead_self_test():
    shot_analysis.overhead_self_test()
