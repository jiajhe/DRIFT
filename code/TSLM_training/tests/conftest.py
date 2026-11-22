"""Support both root pytest and the bundle's run.sh Python entry point."""
from pathlib import Path
import sys
import pytest

MODULES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(MODULES), str(MODULES.parent)]


@pytest.fixture(scope="session")
def observation():
    import pandas as pd
    from TSLM_training.export_drift_features import CELL, load_model, prepare_panels, replay

    model, config, checkpoint = load_model(CELL)
    values, prices, days, lookback = prepare_panels(config, pd.Timestamp("2022-12-28"))
    shapes = []
    hook = model.temporal_encoder.register_forward_pre_hook(
        lambda module, args: shapes.append(tuple(args[0].shape))
    )
    try:
        arrays, errors = replay(model, values, prices, days, lookback, "cpu")
    finally:
        hook.remove()
    return checkpoint, prices, shapes, arrays, errors
