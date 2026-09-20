"""
Deterministic, domain-driven feature engineering for the DKV Mobility case.

All features are calculated row-wise from information already available for
the customer. No statistics are learned from the dataset, so these
transformations can be applied identically to training and inference data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PAY_STATUS_COLUMNS = [
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
]

BILL_COLUMNS = [
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
]

PAYMENT_COLUMNS = [
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
]

ENGINEERED_FEATURES = [
    "max_delinquency",
    "mean_delinquency",
    "delinquent_months",
    "severe_delinquent_months",
    "mean_bill_amount",
    "bill_amount_std",
    "bill_amount_change",
    "current_utilization",
    "mean_utilization",
    "mean_payment_amount",
    "payment_amount_std",
    "zero_payment_months",
    "payment_to_positive_bill_ratio",
]


def add_domain_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add deterministic credit-behaviour features.

    The original columns are retained. New features summarize:
    - delinquency frequency/severity,
    - balance level/volatility/trend,
    - utilization relative to the credit limit,
    - payment level/volatility/frequency,
    - aggregate payment coverage.

    No target information or population-level statistics are used.
    """

    required_columns = (
        PAY_STATUS_COLUMNS
        + BILL_COLUMNS
        + PAYMENT_COLUMNS
        + ["limit_bal"]
    )

    missing = sorted(
        set(required_columns) - set(df.columns)
    )
    if missing:
        raise ValueError(
            f"Cannot engineer features; missing columns: {missing}"
        )

    out = df.copy()

    pay_status = out[PAY_STATUS_COLUMNS]
    bills = out[BILL_COLUMNS]
    payments = out[PAYMENT_COLUMNS]

    # Delinquency behaviour across the six observed months.
    out["max_delinquency"] = pay_status.max(axis=1)
    out["mean_delinquency"] = pay_status.mean(axis=1)
    out["delinquent_months"] = (
        pay_status.gt(0).sum(axis=1)
    )
    out["severe_delinquent_months"] = (
        pay_status.ge(2).sum(axis=1)
    )

    # Balance level, variation, and most-recent-vs-oldest movement.
    out["mean_bill_amount"] = bills.mean(axis=1)
    out["bill_amount_std"] = bills.std(
        axis=1,
        ddof=0,
    )
    out["bill_amount_change"] = (
        out["bill_amt1"] - out["bill_amt6"]
    )

    # Utilization proxies. LIMIT_BAL is positive in this dataset, but the
    # denominator is guarded for robustness.
    safe_limit = out["limit_bal"].clip(lower=1)
    out["current_utilization"] = (
        out["bill_amt1"] / safe_limit
    )
    out["mean_utilization"] = (
        out["mean_bill_amount"] / safe_limit
    )

    # Payment behaviour across the six observed months.
    out["mean_payment_amount"] = payments.mean(axis=1)
    out["payment_amount_std"] = payments.std(
        axis=1,
        ddof=0,
    )
    out["zero_payment_months"] = (
        payments.eq(0).sum(axis=1)
    )

    # Rough aggregate payment-coverage proxy. Only positive billed amounts
    # enter the denominator so credit balances do not create sign artefacts.
    total_payments = payments.sum(axis=1)
    total_positive_bills = (
        bills.clip(lower=0).sum(axis=1).clip(lower=1)
    )
    out["payment_to_positive_bill_ratio"] = (
        total_payments / total_positive_bills
    )

    # Defensive cleanup for possible numerical edge cases.
    out[ENGINEERED_FEATURES] = (
        out[ENGINEERED_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    return out
