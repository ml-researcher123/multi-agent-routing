# Weighted WAR Methods Summary

- methods: ['ML', 'Encoder']
- utility_cost_lambda: 0.1
- utility_extra_lambda: 0.05
- war_cost_tiers: .\agent_cost_tiers_12.json

## Selected dev-optimal settings
- seed 42 ML: base threshold=0.60 (dev utility=0.436), WAR threshold=0.40, WAR lambda=0.10 (dev utility=0.447)
- seed 42 Encoder: base threshold=0.60 (dev utility=0.658), WAR threshold=0.50, WAR lambda=0.02 (dev utility=0.680)
- seed 43 ML: base threshold=0.60 (dev utility=0.436), WAR threshold=0.40, WAR lambda=0.10 (dev utility=0.447)
- seed 43 Encoder: base threshold=0.60 (dev utility=0.649), WAR threshold=0.50, WAR lambda=0.02 (dev utility=0.674)
- seed 44 ML: base threshold=0.60 (dev utility=0.436), WAR threshold=0.40, WAR lambda=0.10 (dev utility=0.447)
- seed 44 Encoder: base threshold=0.60 (dev utility=0.666), WAR threshold=0.50, WAR lambda=0.02 (dev utility=0.678)

## Test summary (mean+-std over seeds)

### ML

#### base
- f1: 0.7179 +- 0.0000
- jacc: 0.6636 +- 0.0000
- exact: 0.4967 +- 0.0000
- success: 0.6667 +- 0.0000
- coverage: 0.7533 +- 0.0000
- avg_cost: 2.6700 +- 0.0000
- avg_extra: 0.4267 +- 0.0000
- avg_utility: 0.4650 +- 0.0000
- avg_p: 1.5033 +- 0.0000

#### war
- f1: 0.6972 +- 0.0000
- jacc: 0.6342 +- 0.0000
- exact: 0.4533 +- 0.0000
- success: 0.6667 +- 0.0000
- coverage: 0.7539 +- 0.0000
- avg_cost: 2.4800 +- 0.0000
- avg_extra: 0.5267 +- 0.0000
- avg_utility: 0.4796 +- 0.0000
- avg_p: 1.6000 +- 0.0000

### Encoder

#### base
- f1: 0.8964 +- 0.0062
- jacc: 0.8560 +- 0.0083
- exact: 0.7356 +- 0.0158
- success: 0.7433 +- 0.0120
- coverage: 0.8607 +- 0.0068
- avg_cost: 2.0567 +- 0.0186
- avg_extra: 0.0289 +- 0.0038
- avg_utility: 0.6536 +- 0.0063
- avg_p: 1.2044 +- 0.0084

#### war
- f1: 0.9136 +- 0.0082
- jacc: 0.8725 +- 0.0108
- exact: 0.7333 +- 0.0153
- success: 0.9211 +- 0.0069
- coverage: 0.9639 +- 0.0034
- avg_cost: 2.8189 +- 0.0302
- avg_extra: 0.2422 +- 0.0334
- avg_utility: 0.6699 +- 0.0054
- avg_p: 1.6611 +- 0.0353
