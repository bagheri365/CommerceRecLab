# CommerceRecLab v0.6 — Candidate Generation / Retrieval

v0.5 showed that learned task-specific reranking remained constrained by candidate recall. v0.6 therefore moves the intervention upstream and asks whether broader, still leakage-safe retrieval can recover substantially more future targets before ranking.

## Primary estimand

For each frozen v0.2 evaluation query, candidate recall is the fraction of the task-specific future target set that appears anywhere in the retrieval candidate union before top-K ranking. View, add-to-cart, and transaction are reported separately.

## Retrieval variants

- `base_union`: the v0.5-style category, co-visitation, recency, and visitor-history union at depth 100.
- `deep_category`: increases same-category retrieval to depth 300 while other sources remain at 100.
- `deep_behavioral`: increases co-visitation, recency, and visitor-history retrieval to 300 while category remains at 100.
- `parent_category`: adds popular items from sibling categories that share the current item's parent in `category_tree.csv`.
- `expanded_union`: combines deep category, deep behavioral, and parent/sibling-category expansion.

Category state is frozen using only metadata at or before the training cutoff. The category tree supplies hierarchy structure only; no future item-property state is used.

## Cost and ranking diagnostics

Candidate-set mean, median, and p95 are reported alongside candidate recall. Fixed RRF top-K Recall/NDCG are secondary diagnostics only; v0.6 does not tune a learned ranker.

## Retain rule

Retain `expanded_union` only if mean validation candidate recall across view, add-to-cart, and transaction improves over `base_union` by at least 0.03 absolute and no task-specific validation candidate recall decreases. Test is confirmation only.

## Claim discipline

Candidate recall is an offline retrieval metric under the frozen session prediction task. Missing events remain unobserved rather than observed dislikes or recommendation impressions, and cart/transaction results are conditional on queries that contain corresponding future targets.
