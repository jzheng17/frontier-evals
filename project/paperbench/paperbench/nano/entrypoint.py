import chz
from nanoeval.eval import EvalSpec, RunnerArgs
from nanoeval.evaluation import run
from nanoeval.library_config import LibraryConfig
from nanoeval.setup import nanoeval_entrypoint
from paperbench.nano.eval import PaperBench
from paperbench.nano.pb_logging import paperbench_library_config, setup_logging


@chz.chz
class DefaultRunnerArgs(RunnerArgs):
    concurrency: int = 5


def _extract_model_name(solver: object) -> str | None:
    """Extract model name from a PaperBench solver for nanoeval monitoring."""
    # OpenHandsSolver stores model as llm_model
    if hasattr(solver, "llm_model"):
        return solver.llm_model
    # BasicAgentSolver stores model inside completer_config
    if hasattr(solver, "completer_config") and hasattr(solver.completer_config, "model"):
        return solver.completer_config.model
    return None


async def main(
    paperbench: PaperBench,
    runner: DefaultRunnerArgs,
    library_config: LibraryConfig = paperbench_library_config,
) -> None:
    setup_logging(library_config)

    # Auto-populate runner.model_name from the solver so nanoeval monitoring works.
    if not runner.model_name:
        model_name = _extract_model_name(paperbench.solver)
        if model_name:
            runner = chz.replace(runner, model_name=model_name)

    await run(EvalSpec(eval=paperbench, runner=runner))


if __name__ == "__main__":
    nanoeval_entrypoint(chz.entrypoint(main))
