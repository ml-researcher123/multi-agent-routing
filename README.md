# Agent Routing as Set-Valued Prediction

This repository contains the code and processed benchmark artifacts for the RecSys paper:

**Multi-Agent Routing as Set-Valued Prediction: A WildChat Benchmark and Cost-Aware Evaluation**

The benchmark treats fixed-catalog agent routing as multi-label set prediction. Given a natural-language prompt, a router predicts a compact set of agents from a 12-agent catalog. The repository includes the processed WildChat-derived splits, baseline training/evaluation scripts, constrained Weighted Agent Routing (WAR) evaluation, inter-prompt consistency analysis, and figure generation scripts.

## Repository Layout

```text
agent-routing-recsys-artifact/
  config/                 # Fixed agent inventory and WAR cost tiers
  data/
    processed/            # Train/dev/test benchmark splits
    results/              # Paper-level aggregate CSV/MD outputs
  figures/                # Generated paper figures
  scripts/                # Reproducibility scripts
  src/                    # Local compatibility modules used by scripts
  requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The encoder-based experiments use `sentence-transformers` / PyTorch. GPU is optional but recommended for full encoder reruns.

## Data

The included processed splits are:

- `data/processed/wildchat_agent12_balanced_3000_equalized_train.csv` (`2400` rows)
- `data/processed/wildchat_agent12_balanced_3000_equalized_dev.csv` (`300` rows)
- `data/processed/wildchat_agent12_balanced_3000_equalized_test.csv` (`300` rows)

Each row contains a `prompt` and `gold_agents` field. Agent labels are AI-assisted heuristic reference labels under the fixed 12-agent catalog in `config/agent_inventory_12.json`.

## Core Experiments

Run the main set-evaluation experiment:

```powershell
python .\scripts\agent12_full_experiment.py `
  --train_csv data/processed/wildchat_agent12_balanced_3000_equalized_train.csv `
  --dev_csv data/processed/wildchat_agent12_balanced_3000_equalized_dev.csv `
  --test_csv data/processed/wildchat_agent12_balanced_3000_equalized_test.csv `
  --seeds 42,43,44 `
  --thresholds 0.4,0.5,0.6,0.7,0.8,0.9 `
  --encoder `
  --encoder_epochs 3 `
  --encoder_batch_size 8 `
  --output_prefix outputs/agent12_full_experiment
```

Run the current constrained WAR study:

```powershell
python .\scripts\evaluate_weighted_war_methods.py `
  --train_csv data/processed/wildchat_agent12_balanced_3000_equalized_train.csv `
  --dev_csv data/processed/wildchat_agent12_balanced_3000_equalized_dev.csv `
  --test_csv data/processed/wildchat_agent12_balanced_3000_equalized_test.csv `
  --methods ml,encoder `
  --seeds 42,43,44 `
  --thresholds 0.4,0.5,0.6,0.7,0.8,0.9 `
  --war_lambdas 0,0.02,0.05,0.1,0.15 `
  --war_cost_tiers config/agent_cost_tiers_12.json `
  --output_prefix outputs/weighted_war_methods
```

Export per-prompt predictions for the deployment-style utility simulation:

```powershell
python .\scripts\export_agent12_predictions.py `
  --train_csv data/processed/wildchat_agent12_balanced_3000_equalized_train.csv `
  --test_csv data/processed/wildchat_agent12_balanced_3000_equalized_test.csv `
  --seeds 42,43,44 `
  --threshold 0.60 `
  --methods majority,knn,ml,cc,mlknn,encoder `
  --encoder_epochs 3 `
  --encoder_batch_size 8 `
  --output_dir outputs/predictions_agent12
```

Run the utility/cost simulation from those exported predictions:

```powershell
python .\scripts\simulate_agent12_execution.py `
  --input Majority=outputs/predictions_agent12/majority_predictions.csv `
  --input KNN=outputs/predictions_agent12/knn_predictions.csv `
  --input ML=outputs/predictions_agent12/ml_predictions.csv `
  --input CC=outputs/predictions_agent12/cc_predictions.csv `
  --input MLkNN=outputs/predictions_agent12/mlknn_predictions.csv `
  --input Encoder=outputs/predictions_agent12/encoder_predictions.csv `
  --cost_tiers_json config/agent_cost_tiers_12.json `
  --output_csv outputs/simulated_execution_summary.csv `
  --output_md outputs/simulated_execution_summary.md
```

Measure routing latency across the same seed set:

```powershell
python .\scripts\measure_latency_agent12.py `
  --train_csv data/processed/wildchat_agent12_balanced_3000_equalized_train.csv `
  --test_csv data/processed/wildchat_agent12_balanced_3000_equalized_test.csv `
  --seeds 42,43,44 `
  --threshold 0.60 `
  --methods majority,knn,ml,cc,mlknn,encoder `
  --encoder_epochs 3 `
  --encoder_batch_size 8 `
  --output_csv outputs/latency_agent12_3seed.csv
```

Run the inter-prompt consistency analysis:

```powershell
python .\scripts\analyze_inter_prompt_consistency.py `
  --input_csvs data/processed/wildchat_agent12_balanced_3000_equalized_train.csv,data/processed/wildchat_agent12_balanced_3000_equalized_dev.csv,data/processed/wildchat_agent12_balanced_3000_equalized_test.csv `
  --output_prefix outputs/wildchat_agent12_interprompt_consistency `
  --backend tfidf `
  --top_k 5
```

## Zero-Shot LLM Baseline

This requires an OpenAI API key and is optional for local reproducibility.

```powershell
$env:OPENAI_API_KEY="your_key_here"
python .\scripts\eval_llm_router.py `
  --dataset data/processed/wildchat_agent12_balanced_3000_equalized_test.csv `
  --inventory_json config/agent_inventory_12.json `
  --model gpt-4o `
  --output_csv outputs/llm_router_wildchat_agent12_test_predictions.csv
```

## Figure Generation

Dataset figures:

```powershell
python .\scripts\generate_agent12_dataset_figs.py `
  --input_csvs data/processed/wildchat_agent12_balanced_3000_equalized_train.csv,data/processed/wildchat_agent12_balanced_3000_equalized_dev.csv,data/processed/wildchat_agent12_balanced_3000_equalized_test.csv `
  --out_setsize_png figures/fig_gold_setsize_donut.png `
  --out_agent_png figures/fig_agent_pct_lollipop.png
```

Result figures from included aggregate files:

```powershell
python .\scripts\generate_agent12_result_figs_from_aggregates.py `
  --threshold_csv data/results/manual_aggregate_threshold_3seed.csv `
  --war_dev_csv data/results/weighted_war_methods_dev_sweep.csv `
  --war_test_csv data/results/weighted_war_methods_test_summary.csv `
  --selected_threshold 0.60 `
  --out_threshold_png figures/fig_threshold_sweep.png `
  --out_pr_png figures/fig_pr_scatter.png `
  --out_setsize_png figures/fig_avg_set_size.png `
  --out_war_png figures/fig_war_tradeoff.png
```

## Notes on Reproducibility

- Deterministic baselines can still show minor numerical differences across environments due to library versions.
- Encoder results depend on PyTorch, sentence-transformer version, device, and random seed.
- The included `data/results/` files are the aggregate files used to generate the current paper figures and WAR table.
- The LLM baseline is single-run and API-dependent; it is included as a reference baseline, not as a required artifact step.

## Anonymous Submission Checklist

Before linking this repository in a double-blind submission:

- remove any personal GitHub account metadata or identifying commit history;
- avoid adding author names, institutions, or acknowledgments;
- use an anonymous hosting mechanism if required by the venue.


