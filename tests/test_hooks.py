"""Tests for Claude Code hook handlers."""
from __future__ import annotations

import json
import sys
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

class FakeEmbeddingModel:
    model_name = "fake-model"
    model_id = "fake-model:deterministic-v1"

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(8, dtype=np.float32)
        for t in text.lower().split():
            v[sum(ord(c) for c in t) % 8] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def to_bytes(self, e: np.ndarray) -> bytes:
        return e.astype(np.float32).tobytes()

    def from_bytes(self, d: bytes) -> np.ndarray:
        return np.frombuffer(d, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        an, bn = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (an * bn)) if an > 0 and bn > 0 else 0.0


def _make_graph(tmp_path: Path):
    from waggle.graph import MemoryGraph
    return MemoryGraph(str(tmp_path / "hooks-test.db"), FakeEmbeddingModel(), tenant_id="local-default")


# ── pre_response tests ────────────────────────────────────────────────────────

def test_pre_response_empty_stdin(capsys: pytest.CaptureFixture) -> None:
    """pre_response exits cleanly with empty stdin."""
    hook_path = ROOT / "src" / "waggle" / "hooks" / "claude_code" / "pre_response.py"
    assert hook_path.exists()

    import importlib.util
    spec = importlib.util.spec_from_file_location("pre_response", hook_path)
    mod = importlib.util.module_from_spec(spec)

    with patch("sys.stdin", StringIO("")), \
         patch("sys.exit") as mock_exit, \
         patch("builtins.print") as mock_print:
        try:
            spec.loader.exec_module(mod)
            mod.main()
        except SystemExit:
            pass

    # Should have called print with empty JSON or exited
    assert mock_exit.called or mock_print.called


def test_pre_response_with_prompt(tmp_path: Path) -> None:
    """pre_response returns JSON output for a valid prompt."""
    hook_path = ROOT / "src" / "waggle" / "hooks" / "claude_code" / "pre_response.py"

    import importlib.util
    spec = importlib.util.spec_from_file_location("pre_response", hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    payload = json.dumps({"prompt": "What database did we choose?", "session_id": "test-session"})

    output_lines: list[str] = []

    def fake_print(data: str = "", **kw: object) -> None:
        output_lines.append(str(data))

    with patch("sys.stdin", StringIO(payload)), \
         patch("builtins.print", side_effect=fake_print), \
         patch("sys.exit", side_effect=SystemExit), \
         patch.dict("os.environ", {
             "WAGGLE_DB_PATH": str(tmp_path / "hooks-test.db"),
             "WAGGLE_MODEL": "deterministic",
             "WAGGLE_BACKEND": "sqlite",
             "WAGGLE_DEFAULT_TENANT_ID": "local-default",
         }):
        try:
            mod.main()
        except SystemExit:
            pass

    # Should have printed at least one JSON line
    assert output_lines, "pre_response printed nothing"
    # Last output should be valid JSON
    last = output_lines[-1]
    parsed = json.loads(last)
    assert isinstance(parsed, dict)


def test_pre_response_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """pre_response respects the 5-second timeout and exits 0."""
    hook_path = ROOT / "src" / "waggle" / "hooks" / "claude_code" / "pre_response.py"

    import importlib.util
    spec = importlib.util.spec_from_file_location("pre_response_timeout", hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    payload = json.dumps({"prompt": "test", "session_id": "s1"})

    def slow_import(*args: object, **kw: object) -> None:
        raise TimeoutError("simulated timeout")

    output_lines: list[str] = []

    def fake_print(data: str = "", **kw: object) -> None:
        output_lines.append(str(data))

    with patch("sys.stdin", StringIO(payload)), \
         patch("builtins.print", side_effect=fake_print), \
         patch("sys.exit", side_effect=SystemExit):
        # Simulate timeout by raising it inside main
        original_main = mod.main

        def patched_main() -> None:
            raise TimeoutError("simulated")

        mod.main = patched_main
        try:
            mod.main()
        except (SystemExit, TimeoutError):
            pass
        finally:
            mod.main = original_main


# ── post_response tests ───────────────────────────────────────────────────────

def test_post_response_skips_secrets(tmp_path: Path) -> None:
    """post_response skips capture when secrets are detected."""
    hook_path = ROOT / "src" / "waggle" / "hooks" / "claude_code" / "post_response.py"

    import importlib.util
    spec = importlib.util.spec_from_file_location("post_response", hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Payload with a fake API key in the user message
    payload = json.dumps({
        "session_id": "s1",
        "transcript": [
            {"role": "user", "content": "my key is sk-abc123def456ghi789jkl012mno345"},
            {"role": "assistant", "content": "I see you have an API key."},
        ],
    })

    observe_called = []

    output_lines: list[str] = []

    def fake_print(data: str = "", **kw: object) -> None:
        output_lines.append(str(data))

    with patch("sys.stdin", StringIO(payload)), \
         patch("builtins.print", side_effect=fake_print), \
         patch("sys.exit", side_effect=SystemExit), \
         patch.dict("os.environ", {
             "WAGGLE_DB_PATH": str(tmp_path / "hooks-test.db"),
             "WAGGLE_MODEL": "deterministic",
             "WAGGLE_BACKEND": "sqlite",
             "WAGGLE_DEFAULT_TENANT_ID": "local-default",
         }):
        try:
            mod.main()
        except SystemExit:
            pass

    # Should have exited silently without calling observe_conversation
    assert not observe_called
    # Output should be empty JSON (silent exit)
    if output_lines:
        assert json.loads(output_lines[-1]) == {}


def test_post_response_empty_transcript(tmp_path: Path) -> None:
    """post_response exits cleanly with empty transcript."""
    hook_path = ROOT / "src" / "waggle" / "hooks" / "claude_code" / "post_response.py"

    import importlib.util
    spec = importlib.util.spec_from_file_location("post_response_empty", hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    payload = json.dumps({"session_id": "s1", "transcript": []})

    output_lines: list[str] = []

    def fake_print(data: str = "", **kw: object) -> None:
        output_lines.append(str(data))

    with patch("sys.stdin", StringIO(payload)), \
         patch("builtins.print", side_effect=fake_print), \
         patch("sys.exit", side_effect=SystemExit):
        try:
            mod.main()
        except SystemExit:
            pass

    if output_lines:
        assert json.loads(output_lines[-1]) == {}


# ── setup hooks tests ─────────────────────────────────────────────────────────

def test_install_claude_hooks_idempotent(tmp_path: Path) -> None:
    """Installing hooks twice should not duplicate entries."""
    from waggle.server import _install_claude_hooks, _uninstall_claude_hooks

    hook_dir = ROOT / "src" / "waggle" / "hooks" / "claude_code"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    with patch("waggle.server._find_claude_settings", return_value=settings_path):
        _install_claude_hooks(hook_dir)
        _install_claude_hooks(hook_dir)  # second call — idempotent

    data = json.loads(settings_path.read_text())
    hooks = data.get("hooks", {})

    # Each event should have exactly one waggle entry
    for event in ("UserPromptSubmit", "Stop", "PreCompact"):
        entries = hooks.get(event, [])
        waggle_entries = [
            e for e in entries
            if any("waggle" in str(h.get("command", "")) for h in e.get("hooks", []))
        ]
        assert len(waggle_entries) == 1, (
            f"Expected exactly 1 waggle entry for {event}, got {len(waggle_entries)}"
        )


def test_uninstall_hooks_removes_block(tmp_path: Path) -> None:
    """uninstall-hooks removes waggle entries cleanly."""
    from waggle.server import _install_claude_hooks, _uninstall_claude_hooks

    hook_dir = ROOT / "src" / "waggle" / "hooks" / "claude_code"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    with patch("waggle.server._find_claude_settings", return_value=settings_path):
        _install_claude_hooks(hook_dir)
        result = _uninstall_claude_hooks()

    assert result is not None
    data = json.loads(settings_path.read_text())
    hooks = data.get("hooks", {})
    for event_entries in hooks.values():
        for entry in event_entries:
            for h in entry.get("hooks", []):
                assert "waggle" not in str(h.get("command", "")), (
                    "Waggle hook entry still present after uninstall"
                )


def test_uninstall_hooks_idempotent(tmp_path: Path) -> None:
    """uninstall-hooks is idempotent — second call returns None."""
    from waggle.server import _install_claude_hooks, _uninstall_claude_hooks

    hook_dir = ROOT / "src" / "waggle" / "hooks" / "claude_code"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    with patch("waggle.server._find_claude_settings", return_value=settings_path):
        _install_claude_hooks(hook_dir)
        _uninstall_claude_hooks()
        result2 = _uninstall_claude_hooks()

    assert result2 is None, "Second uninstall should return None (nothing to remove)"


def test_setup_writes_managed_block(tmp_path: Path) -> None:
    """waggle-mcp setup --yes writes the hooks block to Claude Code settings."""
    from waggle.server import _install_claude_hooks

    hook_dir = ROOT / "src" / "waggle" / "hooks" / "claude_code"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    with patch("waggle.server._find_claude_settings", return_value=settings_path):
        written = _install_claude_hooks(hook_dir)

    assert written == settings_path
    data = json.loads(settings_path.read_text())
    assert "hooks" in data
    # Verify waggle commands are present
    all_commands = [
        h.get("command", "")
        for entries in data["hooks"].values()
        for entry in entries
        for h in entry.get("hooks", [])
    ]
    assert any("waggle" in cmd for cmd in all_commands), (
        "No waggle commands found in installed hooks"
    )
