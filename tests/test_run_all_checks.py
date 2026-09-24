import json

from run_all_checks import build_comparison_table, count_flags, discover_dataset_names


def test_count_flags_counts_nested_true_flags_only():
    data = {
        "channels": {
            "u_x": {"flagged": True, "mean": 1.0},
            "u_y": {"flagged": False, "mean": 2.0},
        },
        "energy": {"total": {"flagged": True}},
        "other": "flagged",  # a string value, not a "flagged" key -- must not count
    }
    assert count_flags(data) == 2


def test_count_flags_zero_for_no_flags():
    assert count_flags({"a": {"flagged": False}, "b": [1, 2, {"flagged": False}]}) == 0


def test_discover_dataset_names_reads_summary_filenames(tmp_path):
    (tmp_path / "re16k_t400_0__check_split.json").write_text("{}")
    (tmp_path / "re16k_t400_0__check_divergence.json").write_text("{}")
    (tmp_path / "re16k_t400_1__check_split.json").write_text("{}")
    (tmp_path / "comparison.csv").write_text("dataset\n")  # must be ignored, not a summary

    assert discover_dataset_names(tmp_path) == ["re16k_t400_0", "re16k_t400_1"]


def test_discover_dataset_names_empty_dir(tmp_path):
    assert discover_dataset_names(tmp_path) == []


def test_build_comparison_table_survives_a_partial_run(tmp_path):
    """A subset run refreshing one dataset must not lose other datasets'
    already-written summaries when the table is rebuilt from discovery.
    """
    full = {
        "n_steps": 100,
        "train_range": [0, 80],
        "test_range": [80, 100],
        "channels": {"u_x": {"flagged": False}},
    }
    (tmp_path / "ds_a__check_split.json").write_text(json.dumps(full))
    (tmp_path / "ds_b__check_split.json").write_text(json.dumps(full))

    # Simulate a partial run touching only ds_a, the way run_all_checks.py's
    # main() rediscovers all known datasets rather than trusting its own args.
    rows = build_comparison_table(discover_dataset_names(tmp_path), tmp_path)

    assert {row["dataset"] for row in rows} == {"ds_a", "ds_b"}
