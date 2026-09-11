import numpy as np
import pytest
from signalscope.metrics import binary_metrics, select_threshold


def test_auc_uses_ranked_scores_and_fpr_uses_real_denominator():
    result = binary_metrics([0, 0, 0, 1], [.1, .2, .7, .8], threshold=.5)
    assert result["roc_auc"] == 1
    assert result["accuracy"] == .75
    assert result["false_positive_rate"] == pytest.approx(1/3)
    assert result["confusion_matrix"] == [[2, 1], [0, 1]]


def test_single_class_does_not_invent_auc():
    assert binary_metrics([0, 0], [.2, .3])["roc_auc"] is None


def test_threshold_respects_real_false_positive_constraint():
    labels = np.array([0]*10+[1]*4)
    scores = np.array([.05,.1,.12,.2,.21,.3,.4,.45,.5,.9,.6,.7,.8,.95])
    threshold = select_threshold(labels, scores, max_fpr=.1)
    result = binary_metrics(labels, scores, threshold)
    assert result["false_positive_rate"] <= .1
    assert result["true_positive_rate"] == 1


def test_bad_scores_fail_instead_of_silently_reporting():
    with pytest.raises(ValueError):
        binary_metrics([0,1], [.1,float("nan")])
