from pathlib import Path

import pandas as pd

from commercereclab.evaluation.observation import (
    audit_observation_semantics,
    render_markdown,
    write_outputs,
)


def _fixture(path: Path) -> Path:
    frame = pd.DataFrame(
        [
            [1, 10, "view", 100, None],
            [2, 10, "addtocart", 100, None],
            [3, 10, "transaction", 100, 900],
            [3, 10, "transaction", 100, 900],  # exact duplicate
            [1, 11, "addtocart", 101, None],
            [2, 11, "transaction", 101, 901],
            [3, 12, "transaction", 102, 902],
            [1, 13, "view", 103, None],
            [3, 13, "transaction", 103, 903],
            [4, 14, "addtocart", 104, None],
            [5, 14, "view", 104, None],  # view occurs after cart
        ],
        columns=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    )
    events = path / "events.csv"
    frame.to_csv(events, index=False)
    return events


def test_observation_audit_counts_patterns_and_order(tmp_path: Path) -> None:
    audit = audit_observation_semantics(_fixture(tmp_path))

    assert audit.raw_rows == 11
    assert audit.exact_duplicate_rows == 1
    assert audit.deduplicated_rows == 10
    assert audit.unique_visitor_item_pairs == 5
    assert audit.pair_pattern_counts == {
        "addtocart+transaction": 1,
        "transaction": 1,
        "view+addtocart": 1,
        "view+addtocart+transaction": 1,
        "view+transaction": 1,
    }
    assert audit.cart_pairs == 3
    assert audit.cart_pairs_with_prior_view == 1
    assert audit.transaction_pairs == 4
    assert audit.transaction_pairs_with_prior_view == 2
    assert audit.transaction_pairs_with_prior_cart == 2
    assert audit.transaction_pairs_with_prior_view_and_cart == 1
    assert audit.transaction_pairs_with_ordered_view_cart == 1


def test_observation_outputs_are_written(tmp_path: Path) -> None:
    audit = audit_observation_semantics(_fixture(tmp_path))
    json_path, md_path = write_outputs(audit, tmp_path / "out")

    assert json_path.exists()
    assert md_path.exists()
    markdown = render_markdown(audit)
    assert "unobserved" not in markdown.lower()  # report focuses on measured path coverage
    assert "released log" in markdown
