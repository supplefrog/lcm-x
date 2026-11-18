"""Run the reported reload orderings in a disposable Windows Hermes process."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
from threading import Event
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "host_probe", ROOT.parent / "round4-followup" / "native-review-probe.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def exercise(home, case, plugins, manager, prototype, state, agent, clone, history, first, stored, turn, new_agent):
    from hermes_cli.cli_session_mixin import CLISessionMixin

    def unload():
        assert manager.unload("hermes-lcm-x")
        assert plugins.get_plugin_context_engine() is None

    def reload():
        assert plugins.get_plugin_context_engine() is None
        plugins.discover_plugins(force=True)
        replacement = plugins.get_plugin_context_engine()
        assert replacement is not None and type(replacement) is not type(prototype)
        assert agent.context_compressor is clone
        return replacement

    database = home / "lcm.db"
    session = agent.session_id

    def rows(sid=None):
        return stored(sid or session, "lcm.db", "store_id")

    if case.startswith("new-"):
        cli = CLISessionMixin()
        cli.session_id, cli.agent, cli._session_db = session, agent, state
        cli.model, cli.max_turns, cli.platform = agent.model, 8, "cli"
        cli.conversation_history = history
        if case.endswith("empty"):
            cli._new_session_id = lambda *_: "empty-before-reload"
            cli.new_session(silent=True)
            assert cli.conversation_history == []
        unload()
        reload()
        cli._new_session_id = lambda *_: "first-turn-after-new"
        cli.new_session(silent=True)
        assert cli.conversation_history == [] and agent.session_id == cli.session_id
        assert agent.session_id != session
        if "-gap-" in case:
            unload()

        question, reply = "First question after reload and /new", "First stopped reply"
        expected = [("user", question), ("assistant", reply)]

        def stop_provider(*args, **kwargs):
            agent._current_streamed_assistant_text = reply
            raise InterruptedError("synthetic provider interruption")

        with patch.object(agent, "_interruptible_api_call", new=stop_provider):
            result = agent.run_conversation(user_message=question, conversation_history=[], task_id=case)
        assert result["interrupted"]
        assert stored(agent.session_id, "state.db", "id") == expected
        print("first interrupted turn, before replay:", rows(agent.session_id), flush=True)
        assert rows(agent.session_id) == expected
        clone.on_turn_complete(result["messages"], interrupted=True)
        clone.on_session_end(agent.session_id, result["messages"])
        assert rows(agent.session_id) == expected and rows() == first
        if plugins.get_plugin_context_engine() is not None:
            unload()
    else:
        if "-empty-" in case:
            agent = new_agent("empty-compaction-lane")
            clone = agent.context_compressor
            session = agent.session_id
            clone.on_session_start(session, platform="cli", context_length=64000)
            assert rows() == []
            history, first = [], []
        tail = [("user", "question during reload"), ("assistant", "answer during reload")]
        view = history + [{"role": role, "content": content} for role, content in tail]
        expected = first + tail
        def call():
            if case.startswith("compress-"):
                assert clone.compress(view, force=True) is view
            else:
                assert clone.should_compress_preflight(view) is False

        if case.endswith(("-gate", "-admitted")):
            gate = clone._compaction_gate
            armed = True
            retiring = None

            class Gate:
                def __enter__(self):
                    nonlocal armed
                    if armed and case.endswith("-gate"):
                        armed = False
                        unload()
                    return gate.__enter__()

                def __exit__(self, *args):
                    nonlocal armed, retiring
                    result = gate.__exit__(*args)
                    if armed and case.endswith("-admitted"):
                        armed = False
                        assert clone._compaction_inflight == 1
                        retiring = pool.submit(unload)
                        assert prototype._unload_group.closing.wait(5)
                    return result

                def __getattr__(self, name):
                    return getattr(gate, name)

            with ThreadPoolExecutor(1) as pool, patch.object(clone, "_compaction_gate", new=Gate()):
                call()
                if retiring is not None:
                    retiring.result(timeout=15)
        elif case.endswith("-gap"):
            unload()
            call()
        else:
            unload()
            replacement = reload()
            entered, release = Event(), Event()
            construct = replacement._clone_for_agent

            def pause_clone():
                entered.set()
                assert release.wait(15)
                return construct()

            with patch.object(replacement, "_clone_for_agent", new=pause_clone), ThreadPoolExecutor(1) as pool:
                calling = pool.submit(call)
                try:
                    assert entered.wait(5)
                    unload()
                finally:
                    release.set()
                calling.result(timeout=15)
        print("compaction input, before replay:", rows(), flush=True)
        assert rows() == expected
        call()
        assert rows() == expected
    database.unlink()
    assert not database.exists()
    print("Deleted lcm.db in the live Hermes process", flush=True)


if sys.argv[1] == "child":
    probe.exercise = exercise
    probe.run_child(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
else:
    probe.SOURCE = ROOT.parent / "follow-next"
    probe.ROOT = ROOT
    probe.__file__ = __file__
    probe.HOSTS["floor"] = ROOT.parent / "win-acceptance-2506" / "hermes-floor-clean"
    sys.exit(probe.run_parent(sys.argv[1], sys.argv[2], sys.argv[3:]))
