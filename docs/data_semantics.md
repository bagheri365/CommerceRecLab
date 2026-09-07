# Data Semantics

CommerceRecLab uses the Retailrocket e-commerce dataset as its primary empirical source.

## Raw empirical objects

The release contains four canonical files:

```text
events.csv
item_properties_part1.csv
item_properties_part2.csv
category_tree.csv
```

### Event log

`events.csv` contains timestamped visitor-item actions:

```text
timestamp, visitorid, event, itemid, transactionid
```

The observed event types are expected to be:

```text
view
addtocart
transaction
```

An event means that the corresponding action was logged. It does **not** imply that every item without an event was shown to the visitor and rejected.

Therefore:

```text
no event for (visitor, item) != observed negative preference
```

The dataset should not be described as a complete recommendation-impression log unless an additional source establishes that fact.

### Transaction IDs

`transactionid` identifies observed transaction outcomes. It is not an input feature available before the purchase and must not leak into ranking or candidate-generation features intended to predict a future transaction.

## Item properties

The two `item_properties_part*.csv` files contain timestamped item state:

```text
timestamp, itemid, property, value
```

Item properties are time-varying. A feature for an event at time `t` may use only item state observed at or before `t`.

Correct conceptual join:

```text
item state = latest known state with property_timestamp <= event_timestamp
```

Incorrect conceptual join:

```text
item state = latest state anywhere in the full dataset
```

The latter can leak future information.

Many property/value identifiers are opaque or hashed. Do not assign business semantics to them unless the dataset documentation explicitly provides those semantics.

## Category hierarchy

`category_tree.csv` contains:

```text
categoryid, parentid
```

It defines the category hierarchy used by item `categoryid` property records. Root categories have no parent.

## Valid empirical claims

The data can support experiments about:

- observed visitor-item behavioral sequences;
- view/add-to-cart/transaction outcomes;
- temporal recommendation and session intent;
- item metadata/state available at a given time;
- category-aware retrieval/ranking;
- catalog coverage and popularity concentration;
- candidate-generation and serving experiments built on the observed corpus.

## Claims the raw data do not establish

Without additional evidence, do not claim the dataset directly records:

- every recommendation impression;
- every product a visitor considered and rejected;
- causal effects of recommendations;
- complete inventory state outside the provided `available` property history;
- semantic meaning for opaque property hashes.
