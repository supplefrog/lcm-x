from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
import inspect
import os
import sqlite3
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "integration"
PLUGIN = "hermes-lcm-x"
HOSTS = {"upstream": ROOT.parent / "hermes-agent", "installed": ROOT.parent / "hermes-installed-d6d4be3"}


def run_child(home, host, case):
    from_fixture = ROOT.parent / "review-prepush-transition-routes" / "committed-profile-probe.py"
    fixture = from_fixture.read_text(encoding="utf-8")
    start = fixture.index('    if scenario == \'dispatch-race\':')
    namespace = {"__file__": str(from_fixture)}
    exec(compile(fixture[:start] + "\n    exercise(home, scenario, plugins, manager, prototype, state, agent, clone, history, first, stored, turn, new_agent)\n",
                 str(from_fixture), "exec"), namespace)
    namespace.update(HOST=host, exercise=exercise)
    namespace["run_child"](home, case)


def exercise(home, case, plugins, manager, prototype, state, agent, clone, history, first, stored, turn, new_agent):
    from agent import conversation_loop
    from hermes_cli import lifecycle
    from hermes_cli.cli_session_mixin import CLISessionMixin

    session = agent.session_id
    database = home / "lcm.db"

    def reload():
        plugins.discover_plugins(force=True)
        current = plugins.get_plugin_context_engine()
        if current is not None:
            assert type(current) is not type(prototype)
        assert agent.context_compressor is clone
        return current

    if case in ("window", "warm", "warm-session-end", "warm-twice", "busy"):
        if case.startswith("warm") or case == "busy":
            reload()
        notify = conversation_loop._notify_context_engine_turn_complete
        stopped_question = "Synthetic question interrupted during reload"
        stopped_reply = "Synthetic partial reply"
        closed = [("user", stopped_question), ("assistant", stopped_reply)]
        received = []

        def stop_provider(*_args, **_kwargs):
            agent._current_streamed_assistant_text = stopped_reply
            raise InterruptedError("Synthetic provider interruption")

        def stop_notification(agent, messages, **kwargs):
            assert kwargs["interrupted"]
            assert [(row["role"], row["content"]) for row in messages[-2:]] == closed
            assert stored(session, "state.db", "id")[-2:] == closed
            received.extend(dict(row) for row in messages)
            if case == "window":
                assert manager.unload(PLUGIN)
                assert plugins.get_plugin_context_engine() is None
                result = notify(agent, messages, **kwargs)
                assert stored(session, "lcm.db", "store_id") == first + closed
                reload()
                return result
            if case.startswith("warm"):
                clone.should_compress_preflight(messages[:-1])
                assert clone._reload_successor is not None
                assert stored(session, "lcm.db", "store_id") == first + closed[:1]
                return notify(agent, messages, **kwargs)

            lock_held = Event()
            entered = Event()

            def busy_writer():
                with closing(sqlite3.connect(database)) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    lock_held.set()
                    assert entered.wait(10)
                    # This hold exceeds the old 50 ms open/ingest budget.
                    Event().wait(0.3)
                    conn.rollback()

            with ThreadPoolExecutor(1) as pool:
                locked = pool.submit(busy_writer)
                assert lock_held.wait(10)
                entered.set()
                result = notify(agent, messages, **kwargs)
                locked.result(10)
                return result

        with patch.object(agent, "_interruptible_api_call", new=stop_provider), \
                patch.object(conversation_loop, "_notify_context_engine_turn_complete", new=stop_notification):
            result = agent.run_conversation(user_message=stopped_question, conversation_history=history,
                                            task_id="round4-stop")
        assert result["interrupted"] and received
        expected = first + closed
        before_replay = stored(session, "lcm.db", "store_id")
        print("AFTER STOP", len(before_replay), "expected", len(expected), flush=True)
        assert before_replay == expected
        if case == "warm-twice":
            reload()
        if case != "warm-session-end":
            history, pair = turn(agent, 42, result["messages"])
            expected += pair
        else:
            history = result["messages"]
        clone.on_session_end(session, history)
        assert stored(session, "lcm.db", "store_id") == expected, stored(session, "lcm.db", "store_id")
        print("AFTER REPLAY AND END", len(expected), flush=True)
        assert manager.unload(PLUGIN)
        database.unlink()
        clone.on_turn_complete(received, interrupted=True)
        assert not database.exists()
        print("DELETED lcm.db during live Hermes", flush=True)
        return

    if case == "lazy-reset":
        cli = CLISessionMixin()
        cli.session_id, cli.agent, cli._session_db = session, agent, state
        cli.model, cli.max_turns, cli.platform = agent.model, 8, "cli"
        cli.conversation_history = history
        cli._new_session_id = lambda _now: "empty-before-reload"
        cli.new_session(silent=True)
        assert cli.conversation_history == []
        reload()
        cli._new_session_id = lambda _now: "replacement-after-lazy-create"
        fired = []
        invoke = lifecycle.invoke_hook
        create = state.create_session

        def fail_once(*args, **kwargs):
            if kwargs.get("session_id") == cli._new_session_id(None) and not fired:
                fired.append("failed-create")
                raise sqlite3.OperationalError("synthetic transient native create failure")
            return create(*args, **kwargs)

        def observe(event, *args, **kwargs):
            if event == "on_session_reset":
                assert not state.get_session(agent.session_id)
                fired.append(event)
            return invoke(event, *args, **kwargs)

        with patch.object(state, "create_session", new=fail_once), patch.object(lifecycle, "invoke_hook", new=observe):
            cli.new_session(silent=True)
        assert fired == ["failed-create", "on_session_reset"]
        assert not state.get_session("empty-before-reload")
        expected = []
        history = []
        for ordinal in (1, 2, 3):
            history, pair = turn(agent, ordinal, history)
            expected += pair
            actual = stored(agent.session_id, "lcm.db", "store_id")
            print("LAZY RESET", ordinal, len(actual), "expected", len(expected), flush=True)
            assert actual == expected
        assert stored(agent.session_id, "state.db", "id") == expected
        return

    if case == "delayed-fresh":
        assert manager.unload(PLUGIN)
        fresh = new_agent("fresh-during-plugin-gap")
        assert getattr(fresh.context_compressor, "name", None) != "lcm-x"
        reload()
        chosen = next(cb for cb in manager._hooks["post_llm_call"] if cb.__name__ == "_on_post_llm_call")
        invoke = manager._invoke_hook_callback
        captured, proceed = Event(), Event()

        def pause_chosen(callback, payload):
            if callback is chosen and payload.get("session_id") == fresh.session_id:
                assert "context_compressor" not in payload
                assert payload["conversation_history"][-1].get("message_uid")
                assert len(stored(fresh.session_id, "state.db", "id")) == 2
                captured.set()
                assert proceed.wait(15)
            return invoke(callback, payload)

        with patch.object(manager, "_invoke_hook_callback", new=pause_chosen), ThreadPoolExecutor(1) as pool:
            running = pool.submit(turn, fresh, 1, [])
            try:
                assert captured.wait(12)
                previous = plugins.get_plugin_context_engine()
                assert reload() is not previous
            finally:
                proceed.set()
            _, expected = running.result(20)
        actual = stored(fresh.session_id, "lcm.db", "store_id")
        print("DELAYED FRESH", len(actual), "expected", len(expected), flush=True)
        assert actual == expected
        return

    if case == "compress-adoption":
        current = reload()
        raw = history + [
            {"role": "user", "content": "Preflight-free next question " * 40},
            {"role": "assistant", "content": "Preflight-free next answer " * 40},
        ]
        opened, release = Event(), Event()
        connect = sqlite3.connect

        def pause(path, *args, **kwargs):
            conn = connect(path, *args, **kwargs)
            if str(path) == database.as_uri() + "?mode=ro" and not opened.is_set():
                opened.set()
                assert release.wait(10)
            return conn

        with patch.object(sqlite3, "connect", new=pause), ThreadPoolExecutor(2) as pool:
            work = pool.submit(clone.compress, raw, current_tokens=64000, force=True)
            try:
                assert opened.wait(10)
                unloading = pool.submit(manager.unload, PLUGIN)
                assert current._unload_group.closing.wait(10)
            finally:
                release.set()
            result = work.result(15)
            assert unloading.result(15)
        expected = first + [(row["role"], row["content"]) for row in raw[-2:]]
        actual = stored(session, "lcm.db", "store_id")
        print("COMPRESS DURABLE", len(actual), "expected", len(expected), flush=True)
        assert actual == expected
        assert isinstance(result, list)
        database.unlink()
        print("DELETED lcm.db during live Hermes", flush=True)
        return

    raise ValueError(case)


def run_parent(sha, host_name, cases):
    host = HOSTS[host_name]
    archive = ROOT / f"{sha[:12]}-probe.tar"
    if not archive.exists():
        subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), sha], cwd=SOURCE, check=True)
    failures = 0
    for case in cases:
        home = ROOT / f"{sha[:8]}-{host_name}-{case}-{uuid4().hex[:6]}"
        plugin = home / "plugins" / PLUGIN
        plugin.mkdir(parents=True)
        (home / "tmp").mkdir()
        with tarfile.open(archive) as packed:
            allowed = ("adapters/", "access_context/", "access_policy/", "skills/")
            packed.extractall(plugin, members=[m for m in packed.getmembers()
                                               if "/" not in m.name or m.name.startswith(allowed)], filter="data")
        (home / "config.yaml").write_text(
            "plugins:\n  enabled: [hermes-lcm-x]\ncontext:\n  engine: lcm-x\nmodel:\n  default: test/model\n",
            encoding="utf-8",
        )
        keep = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT")
        env = {key: os.environ[key] for key in keep if key in os.environ}
        env.update({key: str(home) for key in ("HERMES_HOME", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA")})
        env.update(TMP=str(home/"tmp"), TEMP=str(home/"tmp"), PYTHONUTF8="1", HERMES_DISABLE_LAZY_INSTALLS="1",
                   LCM_EMBEDDINGS_ENABLED="0", LCM_TEMPORAL_ROLLUPS_ENABLED="0", LCM_SHUTDOWN_COMPACTION_WAIT_S="2.0")
        outcome = subprocess.run([sys.executable, "-I", "-B", __file__, "child", str(home), str(host), case],
                                 cwd=host, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=90)
        log = home / "probe.log"
        log.write_text(outcome.stdout + "\n" + outcome.stderr, encoding="utf-8")
        print(sha, host_name, case, "exit", outcome.returncode, log, flush=True)
        print(outcome.stdout, outcome.stderr[-4000:], flush=True)
        failures += outcome.returncode != 0
    return int(bool(failures))


if __name__ == "__main__":
    if sys.argv[1] == "child":
        run_child(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    else:
        sys.exit(run_parent(sys.argv[1], sys.argv[2], sys.argv[3:]))
