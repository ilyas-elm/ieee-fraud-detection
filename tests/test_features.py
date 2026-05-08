import sys
sys.path.insert(0, "/Users/ilyas/Desktop/ieee-fraud-detection")

import numpy as np
import pandas as pd
import pytest
from src.features.build_features import CardAggFeatures


def make_train_data():
    """Small synthetic training set — two cards, known amounts."""
    return pd.DataFrame({
        "card1": [1, 1, 1, 2, 2],
        "TransactionAmt": [100.0, 200.0, 300.0, 50.0, 150.0],
    })


def make_val_data():
    """Validation set — different amounts, same cards plus one unseen card."""
    return pd.DataFrame({
        "card1": [1, 2, 3],
        "TransactionAmt": [999.0, 888.0, 777.0],
    })


def test_card_agg_no_refit_on_val():
    """
    Fit CardAggFeatures on train.
    Record learned card1_mean_ statistics.
    Transform val.
    Assert statistics did not change — val data never influenced the fit.
    """
    X_train = make_train_data()
    X_val = make_val_data()

    transformer = CardAggFeatures()
    transformer.fit(X_train)

    # Record statistics learned from training data only
    mean_after_fit = transformer.card1_mean_.copy()

    # Transform validation set — must not change learned statistics
    transformer.transform(X_val)

    assert transformer.card1_mean_.equals(mean_after_fit), (
        "card1_mean_ changed after transforming val — leakage detected."
    )


def test_card_agg_unseen_card_gets_global_mean():
    """
    Card3 is not in training data.
    It should receive the global mean, not NaN.
    """
    X_train = make_train_data()
    X_val = make_val_data()

    transformer = CardAggFeatures()
    transformer.fit(X_train)

    result = transformer.transform(X_val)

    global_mean = transformer.global_mean_
    unseen_card_mean = result.loc[result.index[2], "card1_amt_mean"]

    assert unseen_card_mean == pytest.approx(global_mean), (
        f"Unseen card got {unseen_card_mean}, expected global mean {global_mean}."
    )