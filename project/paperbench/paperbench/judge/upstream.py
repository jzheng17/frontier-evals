from __future__ import annotations

from paperbench.judge.simple import SimpleJudge


class UpstreamJudge(SimpleJudge):
    """Alias for the upstream-equivalent judge configuration.

    TODO: Swap in upstream-compatible prompting/scoring logic.
    """
