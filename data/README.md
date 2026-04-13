# Data Notes

`processed/` contains the fixed WildChat-derived benchmark splits used by the paper.

Schema highlights:

- `prompt_id`: stable row identifier
- `prompt`: natural-language prompt
- `gold_agents`: pipe-separated reference agent set
- `gold_agent_count`: number of agents in `gold_agents`
- metadata fields from the source prompt pool and curation process

`results/` contains aggregate outputs used for paper tables and figures. These are included so figures can be regenerated without rerunning every model.
