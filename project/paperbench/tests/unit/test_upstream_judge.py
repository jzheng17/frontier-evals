from pathlib import Path
from tempfile import TemporaryDirectory

from paperbench.judge.create_judge import create_judge
from paperbench.judge.simple import SimpleJudge
from paperbench.judge.upstream import UpstreamJudge
from paperbench.rubric.tasks import TaskNode


class DummyCompleter:
    encoding_name = "cl100k_base"
    n_ctx = 8192


class DummyCompleterConfig:
    def build(self) -> DummyCompleter:
        return DummyCompleter()


def _minimal_rubric() -> TaskNode:
    return TaskNode.from_dict(
        {
            "name": "root",
            "description": "root",
            "valid_score": True,
            "score": 1,
            "children": [],
        }
    )


def test_upstream_aliases_simple() -> None:
    completer_config = DummyCompleterConfig()
    judge_kwargs = {"completer_config": completer_config}

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        paper_path = tmp / "paper.md"
        submission_dir = tmp / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        paper_path.write_text("placeholder")
        paper_md = paper_path
        judge = create_judge(
            "upstream",
            judge_kwargs,
            paper_path=paper_path,
            rubric=_minimal_rubric(),
            addendum=None,
            judge_addendum=None,
            submission_dir=submission_dir,
            paper_md=paper_md,
            log_path=None,
            float_completer_config=DummyCompleterConfig(),
            int_completer_config=DummyCompleterConfig(),
        )

    assert isinstance(judge, UpstreamJudge)
    assert isinstance(judge, SimpleJudge)
