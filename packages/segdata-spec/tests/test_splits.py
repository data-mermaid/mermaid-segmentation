"""Split-file reading and the group-coverage check."""

from __future__ import annotations

from pathlib import Path

import pytest

from segdata_spec.errors import SegdataError, SplitCoverageError
from segdata_spec.splits import check_coverage, coverage_checker, read_split_file


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "coralnet.csv"
    path.write_text(text)
    return path


VALID = "group,split,split_group\ns1,train,1\ns2,val,2\ns3,test,3\n"


def test_valid_file_reads_groups_as_strings(tmp_path: Path) -> None:
    table = read_split_file(_write(tmp_path, VALID))
    assert table.column("group").to_pylist() == ["s1", "s2", "s3"]
    assert table.column("split_group").to_pylist() == ["1", "2", "3"]


@pytest.mark.parametrize(
    ("text", "column"),
    [("split\ntrain\n", "group"), ("group\ns1\n", "split")],
)
def test_missing_required_column_names_it(tmp_path: Path, text: str, column: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ValueError, match=column) as exc:
        read_split_file(path)
    assert path.name in str(exc.value)


def test_unknown_split_value_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation"):
        read_split_file(_write(tmp_path, "group,split\ns1,validation\n"))


def test_excluded_split_value_is_read_and_kept(tmp_path: Path) -> None:
    table = read_split_file(_write(tmp_path, "group,split\ns1,excluded\ns2,train\n"))
    assert table.column("split").to_pylist() == ["excluded", "train"]


def test_duplicate_group_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="s1"):
        read_split_file(_write(tmp_path, "group,split\ns1,train\ns1,val\n"))


def test_split_group_across_two_splits_is_rejected(tmp_path: Path) -> None:
    text = "group,split,split_group\ns1,train,7\ns2,test,7\n"
    with pytest.raises(ValueError, match="7"):
        read_split_file(_write(tmp_path, text))


def test_blank_split_group_means_none_and_may_span_splits(tmp_path: Path) -> None:
    text = "group,source_id,split_group,split\ns1,1,5,train\ns2,2,,excluded\ns3,3,,train\n"
    assert read_split_file(_write(tmp_path, text)).num_rows == 3


def test_split_group_repeated_within_one_split_is_accepted(tmp_path: Path) -> None:
    text = "group,split,split_group\ns1,train,7\ns2,train,7\n"
    assert read_split_file(_write(tmp_path, text)).num_rows == 2


def test_coverage_allows_groups_the_version_lacks(tmp_path: Path) -> None:
    path = _write(tmp_path, "group,split\ns1,train\ns2,val\ns9,test\n")
    check_coverage(path, "coralnet", {"s1", "s2"})


def test_coverage_names_the_dataset_and_sorted_unlisted_groups(tmp_path: Path) -> None:
    path = _write(tmp_path, "group,split\ns1,train\ns2,val\ns9,test\n")
    with pytest.raises(SplitCoverageError, match="coralnet") as exc:
        check_coverage(path, "coralnet", {"s1", "s8", "s7"})
    assert "['s7', 's8']" in str(exc.value)
    assert isinstance(exc.value, SegdataError)


def test_coverage_checker_checks_only_mapped_datasets(tmp_path: Path) -> None:
    check = coverage_checker({"coralnet": _write(tmp_path, "group,split\ns1,train\n")})
    check("pacific", {"anything"})
    with pytest.raises(SplitCoverageError):
        check("coralnet", {"s99"})
