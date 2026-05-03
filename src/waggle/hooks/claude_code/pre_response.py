#!/usr/bin/env python3
"""
Waggle Claude Code hook: UserPromptSubmit (pre-response).

Triggered before Claude responds to a user prompt.
Calls prime_context or query_graph to inject relevant memory context.

Protocol: reads JSON from stdin, writes JSON to stdout.
Always exits 0 — never blocks the user's session.
Timeout: 5 seconds.
"""
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

# Ensure waggle src is importable when run as a script
_HERE = Path(__file__).resolve()
for _candidate in [
    _HERE.parents[4] / "src",   # repo layout: src/waggle/hooks/claude_code/
    _HERE.parents[3],            # installed package
]:
    if (_candidate / "waggle").exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

_TIMEOUT_SECONDS = 5


def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ANN001
    raise TimeoutError("Waggle pre_response hook timed out")


def _silent_exit() -> None:
    """Exit 0 with empty output — never block the user."""
    print(json.dumps({}))
    sys.exit(0)


def main() -> None:
    # Set up timeout
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_TIMEOUT_SECONDS)

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            _silent_exit()

        payload: dict[str, Any] = json.loads(raw)
        prompt: str = payload.get("prompt", "") or ""
        session_id: str = str(payload.get("session_id", "") or "")

        if not prompt.strip():
            _silent_exit()

        # Import waggle in-process for low latency
        from waggle.config import AppConfig
        from waggle.embeddings import EmbeddingModel
        from waggle.graph import MemoryGraph

        config = AppConfig.from_env()
        if config.backend != "sqlite":
            _silent_exit()

        graph = MemoryGraph(
            config.db_path,
            EmbeddingModel(config.model_name),
            tenant_id=config.default_tenant_id,
        )

        # Try prime_context first (session start), fall back to query_graph
        try:
            result = graph.prime_context(session_id=session_id)
            context_text = result.summary if result.summary else ""
        except Exception:
            context_text = ""

        if not context_text:
            try:
                qr = graph.query(query=prompt[:500], max_nodes=8, max_depth=1)
                if qr.nodes:
                    context_text = "\n".join(
                        f"[{n.node_type.value}] {n.label}: {n.content[:200]}"
                        for n in qr.nodes[:5]
                    )
            except Exception:
                context_text = ""

        if context_text:
            # Inject context as a system reminder
            print(json.dumps({
                "type": "system_reminder",
                "content": f"[Waggle memory context]\n{context_text}",
            }))
        else:
            print(json.dumps({}))

    except (TimeoutError, Exception):
        _silent_exit()


if __name__ == "__main__":
    main()
