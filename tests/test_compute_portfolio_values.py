import numpy as np
import pytest

from simulator import GBMParams, _validate_params, compute_portfolio_values


def _two_asset_params(**overrides) -> GBMParams:
    defaults = dict(
        S0=np.array([100.0, 50.0]),
        mu=np.array([0.05, 0.03]),
        sigma=np.array([0.2, 0.15]),
        weights=np.array([0.6, 0.4]),
        rho=np.array([[1.0, 0.3], [0.3, 1.0]]),
        T=1.0,
        n_steps=4,
        n_paths=1,
        rebal_steps=None,
    )
    defaults.update(overrides)
    return GBMParams(**defaults)


def _bare_params(**overrides):
    """Build a GBMParams shell without running __post_init__ validation."""
    p = object.__new__(GBMParams)
    p.S0 = np.array([100.0, 50.0])
    p.mu = np.array([0.05, 0.03])
    p.sigma = np.array([0.2, 0.15])
    p.weights = np.array([0.6, 0.4])
    p.rho = np.array([[1.0, 0.3], [0.3, 1.0]])
    p.n_steps = 10
    p.rebal_steps = None
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def test_no_rebalancing_terminal_value_hand_computed():
    """
    Two-asset, fixed-share case: V(t) = V0 * sum(w_i * P_i(t) / S0_i).

    S0 = [100, 50], w = [0.6, 0.4], V0 = 150
    shares = [0.9, 1.2]
    """
    params = _two_asset_params(rebal_steps=None, n_steps=2)

    asset_paths = np.array(
        [
            [
                [100.0, 110.0, 120.0],
                [50.0, 55.0, 60.0],
            ]
        ],
        dtype=np.float64,
    )

    expected = np.array(
        [
            [
                150.0,  # 0.9*100 + 1.2*50
                165.0,  # 0.9*110 + 1.2*55
                180.0,  # 0.9*120 + 1.2*60
            ]
        ]
    )

    result = compute_portfolio_values(asset_paths, params)
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


def test_rebalancing_conserves_value_at_boundaries():
    """
    Within each rebalancing segment, holdings are fixed.  The value at the
    segment end must therefore equal the segment-start value scaled by relative
    price moves under the target-weight allocation:

        pv[:, b] == sum_i w_i * pv[:, a] * P_i(b) / P_i(a)

    At internal rebalance boundaries, this endpoint is also the post-rebalance
    start stored at the same index, so matching it verifies that no cash is
    created or destroyed.
    """
    params = _two_asset_params(rebal_steps=2, n_steps=4)

    asset_paths = np.array(
        [
            [
                [100.0, 105.0, 110.0, 115.0, 120.0],
                [50.0, 52.0, 54.0, 56.0, 58.0],
            ],
            [
                [100.0, 95.0, 90.0, 85.0, 80.0],
                [50.0, 55.0, 60.0, 65.0, 70.0],
            ],
        ],
        dtype=np.float64,
    )

    pv = compute_portfolio_values(asset_paths, params)

    n_timepoints = asset_paths.shape[2]
    boundaries = list(range(0, n_timepoints, params.rebal_steps))
    if boundaries[-1] != n_timepoints - 1:
        boundaries.append(n_timepoints - 1)

    weights = params.weights[np.newaxis, :]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        prices_a = asset_paths[:, :, a]
        prices_b = asset_paths[:, :, b]
        expected_at_b = (
            weights * pv[:, a, np.newaxis] * (prices_b / prices_a)
        ).sum(axis=1)
        np.testing.assert_allclose(
            pv[:, b].astype(np.float64), expected_at_b, rtol=0, atol=1e-5
        )


def test_validate_params_rejects_non_symmetric_rho():
    rho = np.array([[1.0, 0.5], [0.3, 1.0]])
    params = _bare_params(rho=rho)

    with pytest.raises(ValueError, match="perfectly symmetric"):
        _validate_params(params)


def test_validate_params_rejects_non_psd_rho():
    rho = np.array([[1.0, 1.05], [1.05, 1.0]])
    params = _bare_params(rho=rho)

    with pytest.raises(ValueError, match="positive semi-definite"):
        _validate_params(params)


def test_rebalancing_hand_traced_every_timestep():
    """
    Fully hand-traced 2-asset example: 3 timepoints (n_steps=2), one rebalance
    at index 1 (rebal_steps=1).  Expected pv is built from explicit share
    counts times prices at each step — no weighted price-ratio formula.

    Setup
    -----
    S0 = [100, 50],  w = [0.6, 0.4],  V0 = 150
    Initial shares: 0.9 of asset 0, 1.2 of asset 1

    Prices (asset 0, asset 1):
      t=0:  100,  50
      t=1:  100,  60
      t=2:  110,  55

    Segment 1 — fixed shares [0.9, 1.2]
      t=0:  0.9 × 100 + 1.2 × 50  =  90 + 60   = 150.00
      t=1:  0.9 × 100 + 1.2 × 60  =  90 + 72   = 162.00

    Rebalance at t=1 (V = 162):
      shares_0 = 0.6 × 162 / 100 = 0.972
      shares_1 = 0.4 × 162 / 60   = 1.08

    Segment 2 — fixed shares [0.972, 1.08]
      t=2:  0.972 × 110 + 1.08 × 55  =  106.92 + 59.4  = 166.32
    """
    params = _two_asset_params(rebal_steps=1, n_steps=2)

    asset_paths = np.array(
        [
            [
                [100.0, 100.0, 110.0],
                [50.0, 60.0, 55.0],
            ]
        ],
        dtype=np.float64,
    )

    expected = np.array([[150.0, 162.0, 166.32]])

    result = compute_portfolio_values(asset_paths, params)
    np.testing.assert_allclose(
        result.astype(np.float64), expected, rtol=0, atol=1e-5
    )
