"""Check worker logging in a real spawn child without loading inference models."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


CHILD_PROBE = """
import logging
import sys
from types import ModuleType, SimpleNamespace
from src.modules.diarization.presentation.schedules.jobs.process_pending_diarization_job import _diarization_worker

module_name = 'src.modules.diarization.infrastructure.services.audio_diarizer'
fake_audio = ModuleType(module_name)
class ProbeDiarizer:
    def __init__(self, **kwargs):
        pass
    def run(self, **kwargs):
        logging.getLogger(module_name).info('Timing: spawn logging probe took 0.001 seconds')
        return SimpleNamespace(segments=[], language='pt', duration=0.0, speakers=[])
fake_audio.AudioDiarizer = ProbeDiarizer
sys.modules[module_name] = fake_audio

class ProbeQueue:
    def put(self, message):
        print('WORKER_RESULT=' + message['type'], flush=True)
        if message['type'] == 'error':
            raise RuntimeError(message['error'])

assert logging.getLogger().level == logging.WARNING
_diarization_worker(dict(hf_token='test-token', model_size='large-v2',
    file_path='unused.wav', language='pt', num_speakers=None,
    min_speakers=None, max_speakers=None), ProbeQueue())
"""


class WorkerLoggingTests(unittest.TestCase):
    def run_probe(self, allowed_levels):
        # The parent deliberately has INFO enabled, as with the Uvicorn API.
        # The spawned child must initialize its own handler and level.
        code = f"""
import logging
import multiprocessing
logging.basicConfig(level=logging.INFO)
process = multiprocessing.get_context('spawn').Process(target=exec, args=({CHILD_PROBE!r}, {{}}))
process.start()
process.join(10)
if process.is_alive():
    process.terminate()
    process.join()
    raise RuntimeError('Worker logging probe timed out')
exitcode = process.exitcode
process.close()
raise SystemExit(exitcode)
"""
        env = os.environ.copy()
        env.update(POSTGRES_USER="test", POSTGRES_PASSWORD="test",
                   POSTGRES_HOST="localhost", POSTGRES_DATABASE="test",
                   LIST_LOG_LEVELS=allowed_levels)
        result = subprocess.run(
            [sys.executable, "-c", code], env=env,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WORKER_RESULT=success", result.stdout)
        return result.stdout

    def test_spawn_worker_emits_info_timings(self):
        output = self.run_probe("INFO,WARNING,ERROR")
        self.assertEqual(output.count("Timing: spawn logging probe"), 1)

    def test_spawn_worker_respects_configured_level_filter(self):
        output = self.run_probe("ERROR")
        self.assertNotIn("Timing: spawn logging probe", output)


if __name__ == "__main__":
    unittest.main()
