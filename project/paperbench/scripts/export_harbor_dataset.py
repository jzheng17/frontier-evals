#!/usr/bin/env python3
"""Export PaperBench data into a Harbor-datasets-friendly layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PAPER_OPTIONAL_FILES = [
    "paper.md",
    "paper.pdf",
    "addendum.md",
    "judge.addendum.md",
    "config.yaml",
    "blacklist.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./_harbor_export", help="Output directory")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--papers", help="Comma-separated paper slugs")
    group.add_argument(
        "--split",
        choices=["debug", "dev", "testing", "human", "all", "lite", "union"],
        help="Split name (or union for all split lists)",
    )
    group.add_argument("--all", action="store_true", help="Export all papers")
    parser.add_argument(
        "--papers-file",
        help="File containing paper slugs (overrides --split)",
    )
    parser.add_argument("--verify-only", action="store_true", help="Only print actions")
    parser.add_argument("--clean", action="store_true", help="Remove prior export")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_split(split_file: Path) -> list[str]:
    slugs = []
    with split_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            slugs.append(line)
    return slugs


def read_union_splits(splits_root: Path) -> list[str]:
    slugs = set()
    split_files = sorted(splits_root.glob("*.txt"))
    for split_file in split_files:
        for slug in read_split(split_file):
            slugs.add(slug)
    return sorted(slugs)


def list_all_papers(papers_root: Path) -> list[str]:
    return sorted([p.name for p in papers_root.iterdir() if p.is_dir()])


def find_expected_result(judge_root: Path) -> Path | None:
    direct = judge_root / "expected_result.json"
    if direct.exists():
        return direct
    if not judge_root.exists():
        return None
    candidates = sorted(judge_root.rglob("expected_result.json"))
    return candidates[0] if candidates else None


def find_submission_tar(judge_root: Path) -> Path | None:
    if not judge_root.exists():
        return None
    direct_tar = judge_root / "submission.tar"
    if direct_tar.exists():
        return direct_tar
    direct_tgz = judge_root / "submission.tar.gz"
    if direct_tgz.exists():
        return direct_tgz
    candidates_tar = sorted(judge_root.rglob("submission.tar"))
    if candidates_tar:
        return candidates_tar[0]
    candidates_tgz = sorted(judge_root.rglob("submission.tar.gz"))
    return candidates_tgz[0] if candidates_tgz else None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, verify_only: bool) -> None:
    if verify_only:
        return
    ensure_parent(dst)
    shutil.copy2(src, dst)


def copy_dir(src: Path, dst: Path, verify_only: bool) -> None:
    if verify_only:
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    root = repo_root()
    papers_root = root / "data" / "papers"
    judge_root = root / "data" / "judge_eval"
    splits_root = root / "experiments" / "splits"

    if args.papers_file:
        slugs = read_split(Path(args.papers_file))
    elif args.papers:
        slugs = [slug.strip() for slug in args.papers.split(",") if slug.strip()]
    elif args.split:
        if args.split == "union":
            slugs = read_union_splits(splits_root)
        else:
            split_file = splits_root / f"{args.split}.txt"
            slugs = read_split(split_file)
    elif args.all:
        slugs = list_all_papers(papers_root)
    else:
        raise SystemExit("Select one of --papers, --split, --all, or --papers-file.")

    if not slugs:
        raise SystemExit("No papers selected for export.")

    out_root = Path(args.out)
    dest_root = out_root / "paperbench"

    if args.clean and not args.verify_only:
        shutil.rmtree(dest_root, ignore_errors=True)

    missing_optional = 0
    papers_with_judge_eval = 0
    exported_papers = 0

    split_files = sorted(splits_root.glob("*.txt"))
    for split_file in split_files:
        dest_split = dest_root / "_upstream" / "experiments" / "splits" / split_file.name
        if args.verify_only or args.verbose:
            print(f"Copy split: {split_file} -> {dest_split}")
        copy_file(split_file, dest_split, args.verify_only)

    for slug in slugs:
        paper_dir = papers_root / slug
        if not paper_dir.exists():
            print(f"WARNING: Missing paper directory: {paper_dir}")
            continue

        dest_paper = dest_root / "papers" / slug
        rubric = paper_dir / "rubric.json"
        if not rubric.exists():
            raise SystemExit(f"Missing required rubric.json for {slug}")

        files_to_copy = [(rubric, dest_paper / "rubric.json")]

        for filename in PAPER_OPTIONAL_FILES:
            src = paper_dir / filename
            if src.exists():
                files_to_copy.append((src, dest_paper / filename))
            else:
                missing_optional += 1

        assets_dir = paper_dir / "assets"
        if assets_dir.exists():
            files_to_copy.append((assets_dir, dest_paper / "assets"))
        else:
            missing_optional += 1

        expected_result = find_expected_result(judge_root / slug)
        submission_tar = find_submission_tar(judge_root / slug)
        has_judge_eval = expected_result is not None or submission_tar is not None
        if has_judge_eval:
            papers_with_judge_eval += 1

        if expected_result is not None:
            files_to_copy.append(
                (expected_result, dest_paper / "judge_eval" / "expected_result.json")
            )
        if submission_tar is not None:
            files_to_copy.append(
                (submission_tar, dest_paper / "judge_eval" / submission_tar.name)
            )

        if args.verify_only or args.verbose:
            print(f"Paper: {slug}")
            for src, dst in files_to_copy:
                print(f"  Copy: {src} -> {dst}")

        for src, dst in files_to_copy:
            if src.is_dir():
                copy_dir(src, dst, args.verify_only)
            else:
                copy_file(src, dst, args.verify_only)
        exported_papers += 1

    print("Export summary:")
    print(f"  Papers exported: {exported_papers}")
    print(f"  Missing optional files: {missing_optional}")
    print(f"  Papers with judge_eval artifacts: {papers_with_judge_eval}")
    print(f"  Destination: {dest_root}")


if __name__ == "__main__":
    main()
