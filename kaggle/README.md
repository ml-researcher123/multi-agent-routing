# Kaggle semantic-group robustness run

This notebook runs the one additional robustness experiment recommended for the REALM submission. It does not perform or claim human label validation.

## Kaggle setup

1. Upload `agent-routing-recsys-artifact-kaggle.zip` as a private Kaggle dataset.
2. Create a notebook with a GPU accelerator (a T4 is sufficient).
3. Enable Internet for the first run so Hugging Face models can be downloaded.
4. Add the uploaded dataset to the notebook.
5. Import and run `run_semantic_group_robustness.ipynb` from top to bottom.

The notebook:

- reconstructs the complete 3,000-prompt pool from the committed random splits;
- creates exact cosine-similarity components with a frozen multilingual sentence encoder;
- forces every prompt pair with cosine similarity at or above `0.82` into the same partition;
- searches for an approximately 80/10/10 component assignment that preserves label and route-cardinality distributions;
- trains Linear SVM and fine-tuned encoder routers with seeds 42, 43, and 44;
- selects ordinary thresholds by development F1 and development utility;
- selects WAR threshold/penalty pairs by development utility;
- evaluates the grouped test split once and exports per-seed and aggregate tables.

Download `semantic_group_robustness_outputs.zip` from the notebook output after completion.
