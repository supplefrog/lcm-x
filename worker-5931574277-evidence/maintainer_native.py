"""Real Windows Hermes probes for review 5931574277, using disposable homes."""

from collections import Counter
from contextlib import closing
import importlib.util
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "native_runner", ROOT.parent / "round4-followup" / "native-review-probe.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def exercise(home, case, plugins, manager, prototype, state, agent, clone, history, first, stored, turn, new_agent):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from hermes_cli.cli_session_mixin import CLISessionMixin

    class CLI(CLICommandsMixin, CLISessionMixin):
        def _display_resumed_history(self):
            pass

        def _restore_session_cwd(self, _):
            pass

        def _restore_session_yolo(self, _):
            pass

        def _restore_session_model(self, _):
            pass

    control = case.endswith("-main")
    case = case.removesuffix("-main")
    closing_native = case.endswith("-close")
    case = case.removesuffix("-close")
    cli = CLI()
    old = cli.session_id = agent.session_id
    cli.agent, cli._session_db, cli.conversation_history = agent, state, history
    cli.model, cli.max_turns, cli.platform = agent.model, 8, "cli"
    cli.reasoning_config = agent.reasoning_config

    def unload():
        assert manager.unload("hermes-lcm-x")
        assert plugins.get_plugin_context_engine() is None

    def reload():
        plugins.discover_plugins(force=True)
        assert plugins.get_plugin_context_engine() is not None
        assert agent.context_compressor is clone

    def interrupt(question, reply):
        def stop_provider(*_args, **_kwargs):
            agent._current_streamed_assistant_text = reply
            raise InterruptedError("synthetic provider interruption")

        with patch.object(agent, "_interruptible_api_call", new=stop_provider):
            result = agent.run_conversation(
                user_message=question, conversation_history=cli.conversation_history, task_id=case,
            )
        assert result["interrupted"]
        cli.conversation_history = result["messages"]
        return [("user", question), ("assistant", reply)]

    ids = {old}
    expected_new = []
    if case.startswith(("resume-", "branch-")):
        if case.startswith("resume-"):
            target = new_agent("resumed-target")
            target_view, _ = turn(target, 20, [])
            if "finalized" in case:
                target.context_compressor.on_session_end(target.session_id, target_view)
            state.end_session(target.session_id, "exit")
        cli._new_session_id = lambda *_args, **_kwargs: "branched-target"
        if case.endswith("gap"):
            unload()
        if case.startswith("resume-"):
            cli._handle_resume_command("/resume resumed-target")
            assert cli.session_id == "resumed-target"
        else:
            cli._handle_branch_command("/branch probe")
            assert cli.session_id == "branched-target"
        assert agent.session_id == cli.session_id
        ids.add(cli.session_id)
        if case.endswith("reload") or closing_native:
            if not closing_native:
                unload()
                reload()
            cli.conversation_history, expected_new = turn(agent, 23, cli.conversation_history)
        else:
            expected_new = interrupt("New question in the switch gap", "Stopped reply in the switch gap")
    elif case == "undo-first":
        cli.conversation_history, _ = turn(agent, 2, cli.conversation_history)
        unload()
        assert cli.undo_last(2, prefill=False) == 2 and not cli.conversation_history
        cli.conversation_history, expected_new = turn(agent, 3, cli.conversation_history)
    elif case in ("new-retry", "new") and (case == "new-retry" or closing_native):
        cli._new_session_id = lambda *_args, **_kwargs: "empty-before-retire"
        cli.new_session(silent=True)
        unload()
        cli._new_session_id = lambda *_args, **_kwargs: "new-in-gap"
        cli.new_session(silent=True)
        ids.add(cli.session_id)
        if closing_native:
            cli.conversation_history, expected_new = turn(agent, 24, cli.conversation_history)
        else:
            question = "Retry this question"
            expected_new = interrupt(question, "First stopped reply")
            assert cli.retry_last() == question
            expected_new += interrupt(question, "Second stopped reply")
    else:
        raise ValueError(case)

    native_rows = stored(agent.session_id, "state.db", "id")
    for row in expected_new:
        assert row in native_rows, (agent.session_id, row, native_rows)
    if closing_native:
        state.end_session(agent.session_id, "cli_close")
    clone.on_session_end(agent.session_id, cli.conversation_history)
    pairs = {sid: stored(sid, "lcm.db", "store_id") for sid in sorted(ids)}
    counts = Counter(row for rows in pairs.values() for row in rows)
    print("Native stored rows:", native_rows, flush=True)
    print("LCM rows after session end:", pairs, flush=True)
    print("Expected new occurrences:", expected_new, flush=True)
    if not control:
        if case == "new-retry":
            # The host wrote two different request occurrences, even though
            # /retry sent the same text. Detect duplicate ingestion by stamp.
            def occurrences(db, column, order):
                with closing(sqlite3.connect((home / db).as_uri() + "?mode=ro", uri=True)) as connection:
                    return connection.execute(
                        f"SELECT role, content, {column} FROM messages WHERE session_id=? ORDER BY {order}",
                        (agent.session_id,),
                    ).fetchall()

            host_turns = occurrences("state.db", "timestamp", "id")
            saved_turns = occurrences("lcm.db", "observed_at", "store_id")
            print("Retry host and LCM occurrence stamps:", host_turns, saved_turns, flush=True)
            assert len(set(host_turns)) == 4
            assert saved_turns == host_turns
        if plugins.get_plugin_context_engine() is not None:
            unload()
        db = home / "lcm.db"
        db.unlink()
        assert not db.exists()
        print("Deleted lcm.db in the live Hermes process", flush=True)
        assert all(counts[row] == expected for row, expected in Counter(expected_new).items()), counts
    else:
        print("Baseline occurrence counts:", {row: counts[row] for row in expected_new}, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        probe.exercise = exercise
        probe.run_child(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    else:
        probe.SOURCE = ROOT.parent / "follow-next"
        probe.ROOT = ROOT
        probe.__file__ = __file__
        sys.exit(probe.run_parent(sys.argv[1], sys.argv[2], sys.argv[3:]))
