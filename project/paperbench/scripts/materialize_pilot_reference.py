#!/usr/bin/env python3
"""Materialize a pilot reference submission for semantic-self-consistency."""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path


PAPER_SLUG = "semantic-self-consistency"

ALLOWED_FILENAMES = {
    "Dockerfile",
    "dockerfile",
    "reproduce.sh",
    "reproduce.py",
    "reproduce.log",
    "reproduce.log.creation_time",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "conda.yml",
    "Makefile",
    "README.md",
    "README.txt",
    "LICENSE",
}

ALLOWED_EXTENSIONS = {
    ".py",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".ipynb",
    ".sh",
    ".bash",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".npy",
    ".npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./_harbor_export", help="Export root path")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_extract_tar(tar_path: Path, dest: Path, strip_prefix: str | None = None) -> None:
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.isdev() or member.issym() or member.islnk():
                continue
            name = member.name
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix) :].lstrip("/")
            if not name:
                continue
            normalized = os.path.normpath(name)
            if normalized.startswith("..") or os.path.isabs(normalized):
                raise ValueError(f"Unsafe path in tar: {member.name}")
            target = dest / normalized
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with extracted, open(target, "wb") as handle:
                shutil.copyfileobj(extracted, handle)
            os.chmod(target, member.mode)


def detect_strip_prefix(tar_path: Path) -> str | None:
    with tarfile.open(tar_path, "r:*") as tar:
        names = [m.name for m in tar.getmembers() if m.name]
    if not names:
        return None
    first_parts = {Path(name).parts[0] for name in names if Path(name).parts}
    if len(first_parts) == 1:
        prefix = next(iter(first_parts))
        if prefix in {"submission", "reference_submission"}:
            return prefix
    return None


def find_reference_in_paper(paper_dir: Path) -> tuple[str, Path] | None:
    candidate_dirs = [
        "reference_submission",
        "submission",
        "sample_submission",
        "gold_submission",
        "reference",
        "sample",
    ]
    for name in candidate_dirs:
        candidate = paper_dir / name
        if candidate.is_dir():
            return ("dir", candidate)
        if candidate.is_file():
            return ("file", candidate)

    tar_candidates = sorted(paper_dir.glob("*submission*.tar*"))
    if tar_candidates:
        return ("tar", tar_candidates[0])

    ref_tar_candidates = sorted(paper_dir.glob("*reference*.tar*"))
    if ref_tar_candidates:
        return ("tar", ref_tar_candidates[0])

    return None


def find_judge_eval_submission(judge_root: Path) -> Path | None:
    tar_candidates = sorted(judge_root.rglob("submission.tar"))
    if tar_candidates:
        return tar_candidates[0]
    tgz_candidates = sorted(judge_root.rglob("submission.tar.gz"))
    return tgz_candidates[0] if tgz_candidates else None


def write_placeholder(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    script_path = dest / "reproduce.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo 'Placeholder reference submission; no official outputs.' | tee reproduce.log\n"
    )
    os.chmod(script_path, 0o755)


def clean_reference_tree(dest: Path) -> None:
    for git_dir in dest.rglob(".git"):
        shutil.rmtree(git_dir, ignore_errors=True)

    for ds_store in dest.rglob(".DS_Store"):
        ds_store.unlink(missing_ok=True)

    for apple_double in dest.rglob("._*"):
        if apple_double.is_file():
            apple_double.unlink(missing_ok=True)

    for path in dest.rglob("*"):
        if path.is_dir():
            continue
        name = path.name
        suffix = path.suffix.lower()
        if name in ALLOWED_FILENAMES or suffix in ALLOWED_EXTENSIONS:
            continue
        path.unlink(missing_ok=True)

    for path in sorted(dest.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                next(path.iterdir())
            except StopIteration:
                path.rmdir()


def main() -> None:
    args = parse_args()
    root = repo_root()
    paper_dir = root / "data" / "papers" / PAPER_SLUG
    judge_root = root / "data" / "judge_eval" / PAPER_SLUG

    out_root = Path(args.out)
    dest = out_root / "paperbench" / "papers" / PAPER_SLUG / "reference_submission"
    if dest.exists():
        shutil.rmtree(dest)

    source = None
    source_path = None

    candidate = find_reference_in_paper(paper_dir)
    if candidate:
        source, source_path = candidate
    elif judge_root.exists():
        submission_tar = find_judge_eval_submission(judge_root)
        if submission_tar:
            source = "judge_eval_tar"
            source_path = submission_tar

    if source == "dir":
        shutil.copytree(source_path, dest, dirs_exist_ok=True)
        clean_reference_tree(dest)
        print(f"Reference submission source: A (directory) {source_path}")
        return

    if source == "file":
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest / source_path.name)
        clean_reference_tree(dest)
        print(f"Reference submission source: A (file) {source_path}")
        return

    if source == "tar" or source == "judge_eval_tar":
        dest.mkdir(parents=True, exist_ok=True)
        strip_prefix = detect_strip_prefix(source_path)
        if strip_prefix:
            strip_prefix = strip_prefix + "/"
        safe_extract_tar(source_path, dest, strip_prefix=strip_prefix)
        clean_reference_tree(dest)
        if source == "tar":
            print(f"Reference submission source: A (tar) {source_path}")
        else:
            print(f"Reference submission source: B (judge_eval tar) {source_path}")
        return

    write_placeholder(dest)
    print("Reference submission source: C (placeholder)")
    print("WARNING: No upstream reference submission found; PASS not guaranteed.")


if __name__ == "__main__":
    main()
