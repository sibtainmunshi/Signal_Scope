"""Calibration-error definition checks on hand-computed examples."""
import pytest

from signalscope.metrics import expected_calibration_error


def test_ece_is_zero_when_confidence_matches_accuracy():
    labels = [1] * 8 + [0] * 2
    assert expected_calibration_error(labels, [0.8] * 10) == pytest.approx(0, abs=1e-12)


def test_ece_measures_overconfidence_for_both_classes():
    assert expected_calibration_error([1, 0, 1, 0], [0.99] * 4) == pytest.approx(0.49)
    assert expected_calibration_error([1, 0, 1, 0], [0.01] * 4) == pytest.approx(0.49)
