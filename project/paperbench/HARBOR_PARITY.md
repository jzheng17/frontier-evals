# Harbor Parity Adaptations

This document describes the engineering adaptations made to this PaperBench fork
(`jzheng17/frontier-evals`, branch `audrey/modal-runtime`) for parity experiments
with the [Harbor](https://github.com/laude-institute/harbor) evaluation framework.

## Overview

Three categories of changes were added on top of upstream PaperBench:

1. **Modal Runtime** — `ModalComputerRuntime` as a first-class `ComputerRuntime`
   alternative to Alcatraz, running containers on Modal's cloud with GPU support.
2. **HarborJudge** — A Responses API judge that scores all rubric leaves in a
   single LLM call, matching Harbor's verifier for direct parity comparison.
3. **OpenHands Solver** — A container-native agent solver that runs the OpenHands
   SDK inside the sandbox, independent of host network after launch.

## Modal Runtime

**Files:**
- `paperbench/runtime/modal.py` — `ModalComputerRuntime` and `ModalComputerInterface`
- `tools/pb_run.py` — Dispatcher that injects Alcatraz or Modal runtime overrides based on env YAML config
- `configs/modal_env.yaml` — Modal preset config (GPU types, sandbox timeouts)
- `configs/local_env.yaml` — Alcatraz/local Docker preset config

**Key design decisions:**
- Image is built via Modal's programmatic API (`modal.Image.debian_slim()` +
  `run_commands()` + `add_local_file()`), not `Image.from_dockerfile()`. The
  latter had opaque failures: no build logs, 2.4GB context uploads, and shell
  escaping issues.
- `add_local_file(copy=True)` is required when `run_commands()` follows (Modal
  constraint — the file must be copied into the image layer, not mounted).
- `ModalComputerInterface` maps nanoeval's abstract interface directly to Modal's
  Sandbox API: `send_shell_command` → `Sandbox.exec`, `upload`/`download` →
  `Sandbox.open`.

**Usage:**
```bash
python tools/pb_run.py --env-config configs/modal_env.yaml \
    paperbench.paper_split=pilot \
    paperbench.solver=paperbench.solvers.dummy.solver:PaperBenchDummySolver \
    paperbench.judge.scaffold=dummy \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

## HarborJudge

**Files:**
- `paperbench/judge/harbor.py` — `HarborJudge(Judge)` implementation
- `paperbench/judge/create_judge.py` — Factory registration (`judge=harbor`)
- `paperbench/scripts/run_judge.py` — CLI support for `judge=harbor`

**How it works:**
1. Collects all leaf nodes from the rubric tree.
2. Reads submission files (`.py`, `.sh`, `.yaml`, `.md`, `.json`, `.log`, etc.)
   and concatenates them into a single text block.
3. Sends a single Responses API call with system prompt (evaluator role) + user
   prompt (task ID, rubric JSON, submission content).
4. Parses the JSON response mapping each leaf `task_id` to a 0 or 1 score.
5. Builds the graded tree using rubric weights for weighted averaging.

**Key differences from SimpleJudge:**
- Single LLM call for all leaves (vs per-leaf calls in SimpleJudge).
- Uses `OpenAIResponsesTurnCompleter` (Responses API) instead of
  `OpenAICompletionsTurnCompleter` (Chat Completions API).
- Does not include paper PDF text in the prompt (only rubric + submission code).
- More lenient on Results criteria since it doesn't inspect `reproduce.log`
  execution evidence the way SimpleJudge does.

**Usage (standalone judging):**
```bash
uv run python paperbench/scripts/run_judge.py \
    submission_path=./path/to/submission/ \
    paper_id=semantic-self-consistency \
    judge=harbor \
    out_dir=./results/ \
    completer_config=preparedness_turn_completer.oai_responses_turn_completer:OpenAIResponsesTurnCompleter.Config \
    completer_config.model=gpt-5-mini
```

**Bugs fixed during development:**
1. Responses API `output_messages[0]` may be a `ResponseReasoningItem` (empty
   content), not the actual text. Fixed by scanning all output messages for JSON.
2. Submission text extraction initially only read `.txt/.md/.json/.log`. Extended
   to include `.py`, `.sh`, `.yaml`, `.yml`, `.cfg`, `.toml`, `.ini`, `.csv`,
   `.tsv`, `.r`, `.R` and added directory skipping for `.git`, `.venv`,
   `__pycache__`, `node_modules`.

## OpenHands Solver

**Files:**
- `paperbench/solvers/openhands/solver.py` — `OpenHandsSolver(BasePBSolver)`
- `paperbench/solvers/openhands/__init__.py`

**How it works:**
1. `_setup_computer()`: Creates a Python venv at `/opt/openhands-venv` inside the
   sandbox, installs `openhands-sdk` and `openhands-tools`, and uploads a runner
   script.
2. `_run_agent()`: Reads the LLM API key from the host environment, injects
   `LLM_MODEL`, `LLM_API_KEY`, and optionally `LLM_BASE_URL` as env vars, then
   executes the runner script with `asyncio.timeout(time_limit)`.
3. The runner script uses the OpenHands SDK to create an `Agent` with
   `TerminalTool` and `FileEditorTool`, starts a `Conversation`, and sends the
   task instruction from `/home/instructions.txt`.
4. After completion (or timeout), creates a submission tarball from
   `/home/submission/` and downloads it.

**Key advantage over BasicAgent:** OpenHands runs entirely inside the container
after launch, so a dropped host network connection does not affect the run.

**Configuration fields:**
| Field | Default | Description |
|-------|---------|-------------|
| `llm_model` | `gpt-5-mini` | Model for the OpenHands agent |
| `llm_api_key_env` | `OPENAI_API_KEY` | Host env var containing the API key |
| `llm_base_url` | `None` | Optional base URL (for parity proxy) |
| `time_limit` | `3600` | Agent timeout in seconds |
| `openhands_version` | `None` | Pin a specific SDK version |

## Environment Variables

For parity experiments using the New API proxy:

| Variable | Purpose |
|----------|---------|
| `LLM_API_KEY` / `OPENAI_API_KEY` | Parity API key |
| `LLM_BASE_URL` | Parity proxy base URL |
| `LLM_MODEL` | Model name (e.g., `gpt-5-mini`, `gpt-5.2`) |
| `GRADER_OPENAI_API_KEY` | Judge API key (defaults to `OPENAI_API_KEY`) |

## Parity Experiment Results

### Judge Parity (SSC, gpt-5.2 BasicAgent submission)

| Judge | Model | Score | Non-zero leaves | Duration |
|-------|-------|-------|-----------------|----------|
| SimpleJudge | gpt-5-mini | 0.4172 | ~40/77 | ~3min |
| HarborJudge | gpt-5-mini | 0.6914 | 55/77 | ~6min |

HarborJudge is more lenient on Results criteria (0.617 vs 0.048) because it does
not check `reproduce.log` execution evidence the way SimpleJudge does.

### Agent Runs (Modal)

| Task | Agent | Model | Timeout | Reward | Cost |
|------|-------|-------|---------|--------|------|
| SSC | BasicAgent | gpt-5-mini | 1hr | 0.81 | $0.24 |
| MU | BasicAgent | gpt-5.2 | 8hr | 0.189 | $7.17 |
