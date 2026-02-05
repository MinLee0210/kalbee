import numpy as np
import pytest
from kalbee.modules.filters.auto_filter import AutoFilter
from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.ekf_filter import ExtendedKalmanFilter
from kalbee.modules.filters.abg_filter import AlphaBetaGammaFilter


def test_auto_filter_creation():
    state = np.array([[0.0]])
    cov = np.eye(1)

    # Test KF creation
    kf = AutoFilter.from_filter(
        state, cov, np.eye(1), np.eye(1), np.eye(1), np.eye(1), mode="kf"
    )
    assert isinstance(kf, KalmanFilter)

    # Test EKF creation
    ekf = AutoFilter.from_filter(state, cov, np.eye(1), np.eye(1), mode="ekf")
    assert isinstance(ekf, ExtendedKalmanFilter)

    # Test ABG creation
    abg = AutoFilter.from_filter(np.array([[0], [0], [0]]), 0.1, 0.1, 0.05, mode="abg")
    assert isinstance(abg, AlphaBetaGammaFilter)


def test_auto_filter_invalid_mode():
    with pytest.raises(ValueError):
        AutoFilter.from_filter(mode="unknown_filter")
