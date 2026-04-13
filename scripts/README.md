# Scripts

Current-paper entry points:

- `agent12_full_experiment.py`: main unconstrained baseline training/evaluation for Majority, KNN, ML, CC, ML-kNN, and Encoder.
- `evaluate_weighted_war_methods.py`: current constrained ML/Encoder + WAR evaluation.
- `export_agent12_predictions.py`: exports per-prompt prediction CSVs for the main routers across one or more seeds.
- `simulate_agent12_execution.py`: deployment-style coverage/cost/utility metrics from prediction CSVs.
- `measure_latency_agent12.py`: latency measurement for the main routers across one or more seeds.
- `eval_llm_router.py`: optional zero-shot LLM baseline.
- `analyze_inter_prompt_consistency.py`: inter-prompt consistency analysis.
- `generate_agent12_dataset_figs.py`: dataset composition figures.
- `generate_agent12_result_figs_from_aggregates.py`: result and WAR figures from aggregate CSVs.

Implementation helpers used by these scripts are in `../src/`.
