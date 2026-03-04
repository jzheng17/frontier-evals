from __future__ import annotations

from typing import get_args

from paperbench.nano.eval import PaperBench
from paperbench.utils import get_experiments_dir


def test_pilot_split_is_declared() -> None:
    paper_split_type = PaperBench.__annotations__["paper_split"]
    assert "pilot" in get_args(paper_split_type)


def test_pilot_split_contains_single_paper() -> None:
    split_path = get_experiments_dir() / "splits" / "pilot.txt"
    lines = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
