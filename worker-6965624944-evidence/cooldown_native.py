"""Exercise the merged cooldown through a real Windows plugin manager."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_probe', ROOT.parent / 'round4-followup' / 'native-review-probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def exercise(home, case, plugins, manager, prototype, state, agent, clone, history, first, stored, turn, new_agent):
    assert case == 'merged-cooldown'
    session = agent.session_id
    database = home / 'lcm.db'
    assert stored(session, 'lcm.db', 'store_id') == first
    assert manager.unload('hermes-lcm-x')
    assert plugins.get_plugin_context_engine() is None
    assert clone._compression_block_reason() is None
    for n in range(2):
        plugins.discover_plugins(force=True)
        current = plugins.get_plugin_context_engine()
        assert current is not None and type(current) is not type(prototype)
        assert agent.context_compressor is clone
        clone.should_compress_preflight(history)
        successor = clone._retained_successor()
        assert successor is not None
        assert clone._compression_block_reason() is None, 'stale cooldown crossed registration'
        signature = (successor._store_complete_frontier(), 10, None)
        successor._record_retained_stop(signature)
        successor._record_retained_stop(signature)
        assert successor._retained_stop_hold_active()
        assert clone._compression_block_reason() == 'cooldown:lcm_retained_stop'
        assert stored(session, 'lcm.db', 'store_id') == first
        print('Successor', n + 1, 'reports current cooldown; exact stored rows retained', flush=True)
        assert manager.unload('hermes-lcm-x')
        assert plugins.get_plugin_context_engine() is None
        assert clone._compression_block_reason() is None
    database.unlink()
    assert not database.exists()
    print('Deleted lcm.db in the live Hermes process', flush=True)


if sys.argv[1] == 'child':
    probe.exercise = exercise
    probe.run_child(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
else:
    probe.SOURCE = ROOT.parent / 'follow-next'
    probe.ROOT = ROOT
    probe.__file__ = __file__
    probe.HOSTS['floor'] = ROOT.parent / 'win-acceptance-2506' / 'hermes-floor-clean'
    sys.exit(probe.run_parent(sys.argv[1], sys.argv[2], sys.argv[3:]))
