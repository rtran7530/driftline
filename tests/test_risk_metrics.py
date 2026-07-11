import numpy as np
from simulator import GBMParams, compute_risk


def test_max_drawdown_hand_traced():
    portfolio_values = np.array([[100.0, 120.0, 90.0, 80.0, 95.0]])
    expected_max_dd = 40.0

    running_peak = 100.0
    max_dd = 0.0
    for v in portfolio_values[0]:
        if v > running_peak:
            running_peak = v
        dd = running_peak - v
        if dd > max_dd:
            max_dd = dd

    assert max_dd == expected_max_dd


def test_max_drawdown_matches_compute_risk():
    portfolio_values = np.array([[100.0, 120.0, 90.0, 80.0, 95.0]])
    params = GBMParams(
        S0=np.array([100.0]),
        mu=np.array([0.05]),
        sigma=np.array([0.2]),
        weights=np.array([1.0]),
        rho=np.array([[1.0]]),
        T=1.0,
        n_steps=4,
        n_paths=1,
    )
    risk = compute_risk(portfolio_values, params)
    assert risk.mdd_95 == 40.0
