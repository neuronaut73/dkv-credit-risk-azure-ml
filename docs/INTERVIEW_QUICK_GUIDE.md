# Interview quick guide

## Where to show the final 10 features
- `docs/results/candidate_final_feature_sets.json` -> `expanded.top_10`
- `src/train_compact_azure.py` -> `load_selected_features()` and `model_metadata`
- Azure ML registered model -> compact model details/tags

## Where feature formulas live
- `src/features.py`

## Where nested feature selection lives
- `analysis/budget_feature_selection_experiment.py`

## Where Azure orchestration lives
- `pipeline/run_research_pipeline.py`

## Where final compact training/evaluation lives
- `src/train_compact_azure.py`

## Where explainability lives
- `analysis/grouped_permutation_importance.py`
- `analysis/shap_compact_10.py`

## Key result artifacts
- `docs/results/budget_model_summary.csv`
- `docs/results/holdout_metrics_compact10.json`
- `docs/results/grouped_permutation_importance.png`
- `docs/results/shap_beeswarm.png`
- `docs/results/shap_local_waterfall.png`
