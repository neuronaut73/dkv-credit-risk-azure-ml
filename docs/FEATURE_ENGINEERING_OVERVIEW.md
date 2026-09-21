# Feature engineering overview

The implementation is in `src/features.py`. All 13 features are deterministic, row-wise transformations; they do not use the target or fitted population statistics.

| Feature | Exact construction |
|---|---|
| `max_delinquency` | max of `pay_0, pay_2, pay_3, pay_4, pay_5, pay_6` |
| `mean_delinquency` | mean of the six PAY status fields |
| `delinquent_months` | count of PAY status values `> 0` |
| `severe_delinquent_months` | count of PAY status values `>= 2` |
| `mean_bill_amount` | mean of `bill_amt1 ... bill_amt6` |
| `bill_amount_std` | population SD (`ddof=0`) of `bill_amt1 ... bill_amt6` |
| `bill_amount_change` | `bill_amt1 - bill_amt6` |
| `current_utilization` | `bill_amt1 / max(limit_bal, 1)` |
| `mean_utilization` | `mean_bill_amount / max(limit_bal, 1)` |
| `mean_payment_amount` | mean of `pay_amt1 ... pay_amt6` |
| `payment_amount_std` | population SD (`ddof=0`) of `pay_amt1 ... pay_amt6` |
| `zero_payment_months` | count of `pay_amt1 ... pay_amt6 == 0` |
| `payment_to_positive_bill_ratio` | sum payments / max(sum positive bill amounts, 1) |

## Final compact 10-feature model

The frozen feature set is stored in `docs/results/candidate_final_feature_sets.json` under `expanded.top_10`:

- `severe_delinquent_months`
- `education`
- `mean_payment_amount`
- `payment_amount_std`
- `delinquent_months`
- `max_delinquency`
- `pay_0`
- `limit_bal`
- `pay_3`
- `bill_amt2`
