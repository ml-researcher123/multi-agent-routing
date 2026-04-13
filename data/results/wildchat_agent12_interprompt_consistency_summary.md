# Inter-Prompt Consistency Analysis

- prompts: `3000`
- top-k neighbors: `5`
- similarity backend: `tfidf`
- neighbor pairs analyzed: `15000`
- random baseline pairs: `15000`

## Main results

- rank-1 nearest neighbor mean cosine: `0.4536`
- rank-1 nearest neighbor mean Jaccard: `0.6014`
- rank-1 share-any-label rate: `0.7320`
- rank-1 exact label-set match rate: `0.4903`

- top-k mean Jaccard: `0.4940`
- top-k share-any-label rate: `0.6539`
- top-k exact label-set match rate: `0.3669`

- random-pair mean Jaccard: `0.0912`
- random-pair share-any-label rate: `0.1789`
- random-pair exact label-set match rate: `0.0343`

## Similarity gradient

- top 10% cosine threshold: `0.8647`
- top 10% cosine mean Jaccard: `0.9501`
- top 10% cosine share-any-label rate: `1.0000`

- bottom 10% cosine threshold: `0.1161`
- bottom 10% cosine mean Jaccard: `0.2560`
- bottom 10% cosine share-any-label rate: `0.3927`

- Pearson correlation between cosine similarity and label-set Jaccard: `0.5166`

## Interpretation

Higher label overlap among nearest prompts under the chosen text representation than among random pairs suggests that
the routing labels are systematic rather than arbitrary. This is an internal consistency signal: it supports learnable
structure under the benchmark protocol, but it does not by itself establish unique human-level correctness for each prompt.
