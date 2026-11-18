from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import inspect
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tarfile
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
HOST = ROOT.parent / 'hermes-installed-d6d4be3'
SOURCE = ROOT.parent / 'follow-next'
PLUGIN = 'hermes-lcm-x'
REVISIONS = ('ed706592', 'ba61ce61')


def block_network(*_args, **_kwargs):
    raise OSError('Probe network disabled')


def run_child(home, scenario):
    sys.path.insert(0, str(HOST))
    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    socket.create_connection = block_network
    socket.getaddrinfo = block_network

    from hermes_cli import plugins
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from run_agent import AIAgent

    assert Path(plugins.__file__).resolve().is_relative_to(HOST)
    assert get_hermes_home().resolve() == home
    plugins.discover_plugins()
    prototype = plugins.get_plugin_context_engine()
    manager = plugins.get_plugin_manager()
    assert Path(inspect.getfile(type(prototype))).resolve().is_relative_to(home / 'plugins' / PLUGIN)
    state = SessionDB(db_path=home / 'state.db')

    def new_agent(sid):
        with patch('agent.process_bootstrap.OpenAI'):
            agent = AIAgent(
                api_key='test-key-1234567890', base_url='https://openrouter.ai/api/v1',
                model='test/model', quiet_mode=True, session_db=state, session_id=sid,
                skip_context_files=True, skip_memory=True, platform='cli', enabled_toolsets=['todo'],
            )
        agent.client, agent.tool_delay, agent.save_trajectories = MagicMock(), 0, False
        agent._compression_feasibility_checked = True
        agent.context_compressor.update_model('test/model', 64000, provider='openrouter')
        return agent

    def turn(agent, ordinal, history):
        prompt = f'{agent.session_id} synthetic question {ordinal}'
        reply = f'{agent.session_id} synthetic answer {ordinal}'
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply, tool_calls=None), finish_reason='stop')],
            model='test/model', usage=SimpleNamespace(prompt_tokens=100, completion_tokens=6, total_tokens=106),
        )
        agent.client.chat.completions.create.side_effect = [response]
        with patch('agent.title_generator.call_llm', return_value=response):
            result = agent.run_conversation(
                user_message=prompt, conversation_history=history, task_id=f'{agent.session_id}-{ordinal}',
            )
        assert result['final_response'] == reply, result
        return result['messages'], [('user', prompt), ('assistant', reply)]

    def stored(sid, name, order):
        path = home / name
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            return db.execute(
                f'SELECT role,content FROM messages WHERE session_id=? ORDER BY {order}', (sid,),
            ).fetchall()

    agent = new_agent('retained-old')
    clone = agent.context_compressor
    history, first = turn(agent, 1, [])
    assert stored(agent.session_id, 'lcm.db', 'store_id') == first

    if scenario == 'dispatch-race':
        def first_callback():
            return next(cb for cb in manager._hooks['post_llm_call'] if cb.__name__ == '_on_post_llm_call')

        original = first_callback()
        invoke = manager._invoke_hook_callback
        captured, resume = Event(), Event()
        keys = []

        def pause_selected_callback(callback, payload):
            if callback is original:
                keys.extend(sorted(payload))
                captured.set()
                assert resume.wait(25), 'reload gate timed out'
            return invoke(callback, payload)

        with patch.object(manager, '_invoke_hook_callback', pause_selected_callback), ThreadPoolExecutor(1) as executor:
            pending = executor.submit(turn, agent, 2, history)
            try:
                assert captured.wait(20), 'native dispatch never captured the LCM callback'
                plugins.discover_plugins(force=True)
                assert plugins.get_plugin_context_engine() is not prototype
                assert first_callback() is not original
                assert agent.context_compressor is clone
            finally:
                resume.set()
            _, second = pending.result(timeout=35)
        expected = first + second
        print('HOOK_PAYLOAD', keys, flush=True)
    elif scenario == 'new-with-resident':
        old_id = agent.session_id
        plugins.discover_plugins(force=True)
        assert plugins.get_plugin_context_engine() is not prototype
        other = new_agent('resident-other')
        _, other_rows = turn(other, 1, [])
        assert stored(other.session_id, 'lcm.db', 'store_id') == other_rows

        cli = CLISessionMixin()
        cli.session_id, cli.agent, cli._session_db = old_id, agent, state
        cli.model, cli.max_turns, cli.platform = agent.model, 8, 'cli'
        cli.conversation_history = history
        cli._new_session_id = lambda _now: 'retained-new'
        cli.new_session(silent=True)
        assert agent.context_compressor is clone and agent.session_id == 'retained-new'
        assert stored(old_id, 'lcm.db', 'store_id') == first
        history, second = turn(agent, 2, cli.conversation_history)
        _, third = turn(agent, 3, history)
        expected = second + third
        print('ENGINE_BINDINGS', clone._session_id, other.context_compressor._session_id, flush=True)
    else:
        raise ValueError(scenario)

    lcm = stored(agent.session_id, 'lcm.db', 'store_id')
    native = stored(agent.session_id, 'state.db', 'id')
    print('RESULT', json.dumps({'scenario': scenario, 'lcm': lcm, 'native': native, 'expected': expected}), flush=True)
    assert native == expected, 'native Hermes control did not store the expected transcript'
    assert lcm == expected, 'LCM did not store the expected transcript exactly once'


def run_parent():
    arguments = sys.argv[1:]
    revisions = REVISIONS
    if arguments[:1] == ['--revision']:
        revisions, arguments = (arguments[1],), arguments[2:]
    cases = arguments or ['dispatch-race', 'new-with-resident']
    failures = 0
    for revision in revisions:
        actual = subprocess.check_output(['git', 'rev-parse', revision], cwd=SOURCE, text=True).strip()
        short_rev = actual[:8]
        archive = ROOT / f'{actual}.tar'
        if not archive.exists():
            subprocess.run(['git', 'archive', '--format=tar', '-o', str(archive), actual], cwd=SOURCE, check=True)
        for case in cases:
            home = ROOT / f'{short_rev}-{case}'
            if home.exists():
                home = ROOT / f'{short_rev}-{case}-{uuid4().hex[:6]}'
            installed = home / 'plugins' / PLUGIN
            installed.mkdir(parents=True)
            (home / 'tmp').mkdir()
            with tarfile.open(archive) as source:
                members = [entry for entry in source.getmembers() if '/' not in entry.name or entry.name.startswith(('adapters/', 'access_context/', 'access_policy/', 'skills/'))]
                source.extractall(installed, members=members, filter='data')
            (home / 'config.yaml').write_text('plugins:\n  enabled: [hermes-lcm-x]\ncontext:\n  engine: lcm-x\nmodel:\n  default: test/model\n', encoding='utf-8')
            env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATH', 'PATHEXT') if key in os.environ}
            env.update({key: str(home) for key in ('HERMES_HOME', 'USERPROFILE', 'HOME', 'APPDATA', 'LOCALAPPDATA')})
            env.update(TEMP=str(home / 'tmp'), TMP=str(home / 'tmp'), HERMES_DISABLE_LAZY_INSTALLS='1',
                       OPENROUTER_API_KEY='test-key-1234567890', LCM_EMBEDDINGS_ENABLED='0')
            outcome = subprocess.run([sys.executable, '-I', '-B', __file__, '--child', str(home), case],
                                     cwd=HOST, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=80)
            report = f'{actual} {case} exit {outcome.returncode} {home}\n{outcome.stdout}\n{outcome.stderr}'
            (home / 'probe.log').write_text(report, encoding='utf-8')
            print(report, flush=True)
            failures += bool(outcome.returncode)
    sys.exit(bool(failures))


if __name__ == '__main__':
    if sys.argv[1:2] == ['--child']:
        run_child(Path(sys.argv[2]), sys.argv[3])
    else:
        run_parent()
