# Semantic-group robustness summary

Near-neighbor semantic components are kept intact across train, development, and test.
All thresholds and WAR parameters are selected on the grouped development split.

- split sizes: {'train': 2400, 'dev': 301, 'test': 299}
- seeds: 42,43,44
- thresholds: 0.3,0.4,0.5,0.6,0.7,0.8,0.9
- WAR lambdas: 0,0.02,0.05,0.1,0.15
- encoder model: `sentence-transformers/all-mpnet-base-v2`

| Method | Policy | F1 | Exact | Coverage | Cost | Utility |
|---|---|---:|---:|---:|---:|---:|
| Encoder | f1_selected | 0.880+-0.009 | 0.692+-0.015 | 0.847+-0.001 | 1.999+-0.054 | 0.645+-0.006 |
| Encoder | utility_selected | 0.880+-0.009 | 0.692+-0.015 | 0.847+-0.001 | 1.999+-0.054 | 0.645+-0.006 |
| Encoder | war | 0.881+-0.025 | 0.681+-0.037 | 0.915+-0.062 | 2.520+-0.435 | 0.651+-0.016 |
| ML | f1_selected | 0.618+-0.000 | 0.452+-0.000 | 0.624+-0.000 | 2.338+-0.000 | 0.366+-0.000 |
| ML | utility_selected | 0.618+-0.000 | 0.452+-0.000 | 0.624+-0.000 | 2.338+-0.000 | 0.366+-0.000 |
| ML | war | 0.628+-0.000 | 0.475+-0.000 | 0.612+-0.000 | 2.033+-0.000 | 0.390+-0.000 |
| Majority | reference | 0.071+-0.000 | 0.040+-0.000 | 0.063+-0.000 | 1.000+-0.000 | -0.082+-0.000 |
