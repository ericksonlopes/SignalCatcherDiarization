"""Focused regression tests, without downloading models or contacting PostgreSQL."""

import importlib
import sys
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker


class DiarizationImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Provide an isolated configuration and database before importing the app.
        cls.engine = create_engine("sqlite:///:memory:")
        cls.session_factory = sessionmaker(bind=cls.engine)
        connector = ModuleType("src.core.database.connector")
        connector.Base = declarative_base()
        connector.engine = cls.engine
        connector.Session = cls.session_factory

        @contextmanager
        def connect():
            with cls.session_factory() as session:
                yield session

        connector.ConnectorPostgres = connect
        config = ModuleType("src.core.config.settings")
        config.settings = SimpleNamespace(
            DOWNLOAD_YOUTUBE_PATH=None, HF_TOKEN="test-token"
        )
        cls.module_patch = patch.dict(
            sys.modules,
            {
                "src.core.config.settings": config,
                "src.core.database.connector": connector,
            },
        )
        cls.module_patch.start()
        cls.job = importlib.import_module(
            "src.modules.diarization.presentation.schedules.jobs.process_pending_diarization_job"
        )
        cls.heavy_imports = [
            name for name in ("torch", "whisperx") if name in sys.modules
        ]
        cls.torch = MagicMock()
        cls.torch.cuda.is_available.return_value = False
        cls.whisperx = MagicMock()
        cls.inference_patch = patch.dict(
            sys.modules, {"torch": cls.torch, "whisperx": cls.whisperx}
        )
        cls.inference_patch.start()
        cls.audio = importlib.import_module(
            "src.modules.diarization.infrastructure.services.audio_diarizer"
        )
        cls.repo = importlib.import_module(
            "src.modules.diarization.infrastructure.repositories.diarization_task_repository"
        )
        cls.model = cls.repo.DiarizationTaskModel
        connector.Base.metadata.create_all(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.inference_patch.stop()
        cls.module_patch.stop()
        cls.engine.dispose()

    def setUp(self):
        self.whisperx.reset_mock()
        with self.engine.begin() as connection:
            connection.execute(
                text("""
                CREATE TABLE IF NOT EXISTS step_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id VARCHAR NOT NULL,
                    entity_type VARCHAR NOT NULL,
                    previous_step VARCHAR,
                    new_step VARCHAR NOT NULL,
                    changed_at TIMESTAMP NOT NULL,
                    details VARCHAR
                )
            """)
            )

    def _history(self, task_id):
        with self.engine.connect() as connection:
            return (
                connection.execute(
                    text(
                        "SELECT * FROM step_tracking WHERE entity_id=:task_id ORDER BY id"
                    ),
                    {"task_id": task_id},
                )
                .mappings()
                .all()
            )

    def test_linked_task_records_entire_history_under_task_identity(self):
        repository = self.repo.DiarizationTaskRepository()
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        task = repository.create_task(
            file_path="audio.wav", entity_id="video", entity_type="YOUTUBE"
        )
        for step in [
            "TRANSCRIPTION",
            "ALIGNMENT",
            "DIARIZATION",
            "DIARIZED",
            "COMPLETED",
        ]:
            repository.update_task_step(task.id, step)
        rows = self._history(task.id)
        self.assertEqual(
            [row["new_step"] for row in rows],
            [
                "PENDING",
                "TRANSCRIPTION",
                "ALIGNMENT",
                "DIARIZATION",
                "DIARIZED",
                "COMPLETED",
            ],
        )
        self.assertTrue(all(row["entity_type"] == "diarization" for row in rows))
        self.assertIsNone(rows[0]["previous_step"])
        self.assertEqual(rows[-1]["previous_step"], "DIARIZED")
        self.assertGreaterEqual(datetime.fromisoformat(rows[0]["changed_at"]), before)
        stored = repository.get_task(task.id)
        self.assertEqual((stored.entity_id, stored.entity_type), ("video", "YOUTUBE"))

    def test_unlinked_task_records_errors_and_does_not_duplicate_same_step(self):
        repository = self.repo.DiarizationTaskRepository()
        task = repository.create_task(file_path="upload.wav")
        repository.update_task_step(task.id, "TRANSCRIPTION")
        repository.update_task_step(task.id, "TRANSCRIPTION")
        repository.update_task_step(task.id, "ERROR", error_message="processing failed")
        rows = self._history(task.id)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["details"], "processing failed")
        self.assertEqual(rows[-1]["entity_type"], "diarization")

    def test_two_tasks_for_same_video_keep_separate_histories(self):
        repository = self.repo.DiarizationTaskRepository()
        first = repository.create_task(
            file_path="first.wav", entity_id="same-video", entity_type="YOUTUBE"
        )
        repository.update_task_step(first.id, "COMPLETED")
        second = repository.create_task(
            file_path="second.wav", entity_id="same-video", entity_type="YOUTUBE"
        )
        repository.update_task_step(second.id, "ERROR")
        self.assertEqual(
            [row["new_step"] for row in self._history(first.id)],
            ["PENDING", "COMPLETED"],
        )
        self.assertEqual(
            [row["new_step"] for row in self._history(second.id)], ["PENDING", "ERROR"]
        )

    def test_scheduler_import_does_not_load_inference_libraries(self):
        self.assertEqual(self.heavy_imports, [])

    def test_worker_claim_recovery_and_cancelled_result_are_guarded(self):
        repository = self.repo.DiarizationTaskRepository()
        task = repository.create_task("claim.wav")
        with self.session_factory() as session:
            session.get(self.model, task.id).queue_priority = 2
            session.commit()
        self.assertTrue(repository.claim_task(task.id, "worker-one"))
        self.assertFalse(repository.claim_task(task.id, "worker-two"))
        self.assertTrue(repository.renew_task_lease(task.id, "worker-one"))
        self.assertEqual(repository.recover_interrupted_tasks(), 0)
        with self.session_factory() as session:
            stored = session.get(self.model, task.id)
            stored.lease_expires_at = datetime.now(timezone.utc).replace(
                tzinfo=None
            ) - timedelta(seconds=1)
            session.commit()
        self.assertEqual(repository.recover_interrupted_tasks(), 1)
        self.assertFalse(
            repository.update_task_step(task.id, "COMPLETED", worker_token="worker-one")
        )
        self.assertEqual(repository.get_task(task.id).step, "PENDING")
        self.assertEqual(self._history(task.id)[-1]["new_step"], "PENDING")

    def test_duplicate_worker_requests_reuse_active_task(self):
        repository = self.repo.DiarizationTaskRepository()
        first = repository.create_task(
            "repeat.wav", entity_id="repeat-video", entity_type="YOUTUBE"
        )
        second = repository.create_task(
            "repeat.wav", entity_id="repeat-video", entity_type="YOUTUBE"
        )
        self.assertEqual(first.id, second.id)

    def test_only_one_execution_lock_can_be_held(self):
        with self.job.DiarizationExecutionLock() as first:
            self.assertTrue(first.acquired)
            with self.job.DiarizationExecutionLock() as second:
                self.assertFalse(second.acquired)
            first.ensure_owned()
        with self.job.DiarizationExecutionLock() as next_run:
            self.assertTrue(next_run.acquired)

    def test_restart_requeues_unexpired_task_and_claim_refuses_second_execution(self):
        repository = self.repo.DiarizationTaskRepository()
        repository.recover_interrupted_tasks(force=True)
        first = repository.create_task("restart-one.wav")
        second = repository.create_task("restart-two.wav")
        with self.session_factory() as session:
            session.get(self.model, first.id).queue_priority = 3
            session.commit()
        self.assertTrue(repository.claim_task(first.id, "first-owner"))
        self.assertFalse(repository.claim_task(second.id, "second-owner"))
        self.assertEqual(repository.recover_interrupted_tasks(force=True), 1)
        restored = repository.get_task(first.id)
        self.assertEqual(restored.step, "PENDING")
        self.assertIsNone(restored.worker_token)
        self.assertIsNone(restored.progress_percent)
        self.assertFalse(
            repository.update_task_step(
                first.id, "COMPLETED", worker_token="first-owner"
            )
        )

    def test_progress_updates_are_monotonic_and_do_not_add_history(self):
        repository = self.repo.DiarizationTaskRepository()
        task = repository.create_task("audio.wav")
        repository.update_task_step(task.id, "ALIGNMENT")
        repository.update_task_progress(task.id, "ALIGNMENT", 63.9)
        repository.update_task_progress(task.id, "ALIGNMENT", 10)
        self.assertEqual(repository.get_task(task.id).progress_percent, 63)
        self.assertEqual(len(self._history(task.id)), 2)
        repository.update_task_step(task.id, "DIARIZATION")
        self.assertIsNone(repository.get_task(task.id).progress_percent)
        repository.update_task_progress(task.id, "ALIGNMENT", 100)
        self.assertIsNone(repository.get_task(task.id).progress_percent)
        repository.update_task_progress(task.id, "DIARIZATION", 500)
        self.assertEqual(repository.get_task(task.id).progress_percent, 100)
        repository.update_task_step(task.id, "CANCELLED")
        repository.update_task_progress(task.id, "DIARIZATION", 100)
        self.assertIsNone(repository.get_task(task.id).progress_percent)
        self.assertEqual(repository.get_task(task.id).step, "CANCELLED")

    def test_alignment_passes_library_progress_and_reports_completion(self):
        values = []

        def align(*args, progress_callback=None, **kwargs):
            progress_callback(25)
            progress_callback(70)
            return {"segments": []}

        with (
            patch.object(self.whisperx, "align", align),
            patch.object(
                self.audio.model_loader, "get_align_model", return_value=(object(), {})
            ),
        ):
            self.audio.AudioDiarizer("token")._align(
                {"segments": [], "language": "pt"}, object(), None, values.append
            )
        self.assertEqual(values, [0, 25, 70, 100])

    def test_diarization_passes_library_progress(self):
        values = []

        def pipeline(audio, progress_callback=None, **kwargs):
            progress_callback(45)
            progress_callback(100)
            return object()

        self.whisperx.assign_word_speakers.side_effect = lambda _, result: result
        with patch.object(
            self.audio.model_loader, "get_diarization_pipeline", return_value=pipeline
        ):
            self.audio.AudioDiarizer("token")._diarize(
                object(),
                {"segments": [], "language": "pt"},
                None,
                None,
                None,
                values.append,
            )
        self.assertEqual(values, [0, 45, 100])

    def test_reporter_throttles_updates_and_resets_for_next_stage(self):
        queue = MagicMock()
        reporter = self.job.StageProgressReporter(queue)
        with patch.object(
            self.job, "monotonic", side_effect=[10, 10.5, 12.1, 12.2, 12.3, 12.4]
        ):
            for step, value in [
                ("ALIGNMENT", 0),
                ("ALIGNMENT", 10),
                ("ALIGNMENT", 20),
                ("ALIGNMENT", 15),
                ("ALIGNMENT", 100),
                ("DIARIZATION", 0),
            ]:
                reporter(step, value)
        self.assertEqual(
            [call.args[0] for call in queue.put.call_args_list],
            [
                {"type": "stage_progress", "step": "ALIGNMENT", "percent": 0},
                {"type": "stage_progress", "step": "ALIGNMENT", "percent": 20},
                {"type": "stage_progress", "step": "ALIGNMENT", "percent": 100},
                {"type": "stage_progress", "step": "DIARIZATION", "percent": 0},
            ],
        )

    def test_job_persists_progress_separately_from_transitions(self):
        _, repository = self._run_job(
            [
                {"type": "progress", "step": "ALIGNMENT"},
                {"type": "stage_progress", "step": "ALIGNMENT", "percent": 65},
                {"type": "success", "result": {"language": "pt"}},
            ],
            alive=False,
        )
        repository.update_task_progress.assert_called_once_with(
            "task", "ALIGNMENT", 65, worker_token=unittest.mock.ANY
        )
        self.assertEqual(
            [
                call.kwargs["step"]
                for call in repository.update_task_step.call_args_list
            ],
            ["ALIGNMENT", "COMPLETED"],
        )

    def test_alignment_preserves_detected_language(self):
        self.whisperx.align.return_value = {"segments": [], "word_segments": []}
        with patch.object(
            self.audio.model_loader, "get_align_model", return_value=(object(), {})
        ):
            result = self.audio.AudioDiarizer("token")._align(
                {"segments": [], "language": "pt"}, object(), None
            )
        self.assertEqual(result["language"], "pt")

    def test_lazy_alignment_wrapper_forwards_percentage_callback(self):
        values = []
        implementation = ModuleType("whisperx.alignment")

        def internal_align(*args, progress_callback=None, **kwargs):
            progress_callback(35)
            return {"segments": []}

        implementation.align = internal_align

        def align(*args, **kwargs):
            return internal_align(*args, **kwargs)

        align.__module__ = "whisperx"
        with (
            patch.dict(sys.modules, {"whisperx.alignment": implementation}),
            patch.object(self.whisperx, "align", align),
            patch.object(
                self.audio.model_loader, "get_align_model", return_value=(object(), {})
            ),
        ):
            self.audio.AudioDiarizer("token")._align(
                {"segments": [], "language": "pt"}, object(), None, values.append
            )
        self.assertEqual(values, [0, 35, 100])

    def test_failed_alignment_preserves_original_transcript(self):
        original = {"segments": [], "language": "pt"}
        with patch.object(
            self.audio.model_loader,
            "get_align_model",
            side_effect=RuntimeError("unavailable"),
        ):
            result = self.audio.AudioDiarizer("token")._align(original, object(), None)
        self.assertIs(result, original)

    def test_segment_assignment_retains_aligned_output_without_word_work(self):
        transcript = {
            "language": "pt",
            "segments": [
                {
                    "start": 1.234,
                    "end": 2.345,
                    "text": " Olá ",
                    "words": [{"word": "Olá", "start": 1.234, "end": 2.345}],
                }
            ],
            "word_segments": [{"word": "Olá"}],
        }

        def assign(diarized, result):
            self.assertNotIn("word_segments", result)
            self.assertNotIn("words", result["segments"][0])
            result["segments"][0]["speaker"] = "SPEAKER_00"
            return result

        self.whisperx.assign_word_speakers.side_effect = assign
        pipeline = MagicMock()
        with patch.object(
            self.audio.model_loader, "get_diarization_pipeline", return_value=pipeline
        ):
            segments, language = self.audio.AudioDiarizer("token")._diarize(
                object(), transcript, 2, None, None
            )
        pipeline.assert_called_once_with(unittest.mock.ANY, num_speakers=2)
        self.assertEqual(language, "pt")
        self.assertEqual(
            segments[0].to_dict(),
            {
                "speaker": "SPEAKER_00",
                "start": 1.234,
                "end": 2.345,
                "duration": 1.111,
                "text": "Olá",
            },
        )

    def test_timing_is_logged_even_when_stage_fails(self):
        with patch.object(self.audio, "perf_counter", side_effect=[10.0, 12.5]):
            with self.assertLogs(self.audio.logger, level="INFO") as logs:
                with self.assertRaises(RuntimeError):
                    with self.audio._measure_time("test stage"):
                        raise RuntimeError("failed")
        self.assertIn("test stage took 2.500 seconds", logs.output[0])

    def _pipeline_progress(self, error=None):
        progress = []
        with (
            patch.object(self.audio.os.path, "exists", return_value=True),
            patch.object(
                self.audio,
                "load_whisperx_audio",
                return_value=SimpleNamespace(shape=(100,)),
            ),
            patch.object(self.audio.AudioDiarizer, "_transcribe", return_value={}),
            patch.object(self.audio.AudioDiarizer, "_align", return_value={}),
            patch.object(
                self.audio.AudioDiarizer,
                "_diarize",
                return_value=([], "pt"),
                side_effect=error,
            ),
            patch.object(self.audio.model_loader, "unload_whisper"),
            patch.object(self.audio.model_loader, "unload_align"),
            patch.object(self.audio.model_loader, "unload_diarization"),
        ):
            if error:
                with self.assertRaises(RuntimeError):
                    self.audio.AudioDiarizer("token").run(
                        "audio.wav", progress_callback=progress.append
                    )
            else:
                self.audio.AudioDiarizer("token").run(
                    "audio.wav", progress_callback=progress.append
                )
        return progress

    def test_diarized_is_emitted_only_after_speaker_assignment_succeeds(self):
        self.assertEqual(
            self._pipeline_progress(),
            ["TRANSCRIPTION", "ALIGNMENT", "DIARIZATION", "DIARIZED"],
        )
        self.assertEqual(
            self._pipeline_progress(RuntimeError("inference failed")),
            ["TRANSCRIPTION", "ALIGNMENT", "DIARIZATION"],
        )

    def test_diarization_reaches_100_only_after_speaker_assignment(self):
        events = []

        def diarize(audio, result, num, minimum, maximum, callback):
            callback(100)
            events.append("assigned")
            return [], "pt"

        with (
            patch.object(self.audio.os.path, "exists", return_value=True),
            patch.object(
                self.audio,
                "load_whisperx_audio",
                return_value=SimpleNamespace(shape=(100,)),
            ),
            patch.object(self.audio.AudioDiarizer, "_transcribe", return_value={}),
            patch.object(self.audio.AudioDiarizer, "_align", return_value={}),
            patch.object(self.audio.AudioDiarizer, "_diarize", side_effect=diarize),
            patch.object(self.audio.model_loader, "unload_whisper"),
            patch.object(self.audio.model_loader, "unload_align"),
            patch.object(self.audio.model_loader, "unload_diarization"),
        ):
            self.audio.AudioDiarizer("token").run(
                "audio.wav",
                stage_progress_callback=lambda step, percent: events.append(
                    (step, percent)
                ),
            )
        self.assertEqual(
            events, [("DIARIZATION", 99), "assigned", ("DIARIZATION", 100)]
        )

    def test_older_library_without_callback_keeps_processing(self):
        def old_align(*args, return_char_alignments=False):
            return {"segments": []}

        values = []
        with (
            patch.object(self.whisperx, "align", old_align),
            patch.object(
                self.audio.model_loader, "get_align_model", return_value=(object(), {})
            ),
        ):
            result = self.audio.AudioDiarizer("token")._align(
                {"segments": [], "language": "pt"}, object(), None, values.append
            )
        self.assertEqual(values, [])
        self.assertEqual(result["language"], "pt")

    def _run_job(
        self,
        messages,
        update_effect=None,
        start_error=None,
        alive=True,
        stopped_step="PENDING",
    ):
        repository = MagicMock()
        repository.get_pending_tasks.return_value = [
            SimpleNamespace(
                id="task",
                file_path="audio.wav",
                language="pt",
                num_speakers=None,
                min_speakers=None,
                max_speakers=None,
                model_size="large-v2",
            )
        ]
        repository.update_task_step.side_effect = update_effect
        repository.claim_task.return_value = True
        repository.renew_task_lease.return_value = True
        repository.get_task.return_value = SimpleNamespace(
            step=stopped_step, worker_token=None
        )
        process = MagicMock()
        process.pid = None if start_error else 123
        process.start.side_effect = start_error
        process.is_alive.side_effect = [alive, False] if alive else [False, False]
        progress = MagicMock()
        progress.get.side_effect = messages
        context = MagicMock()
        context.Process.return_value = process
        context.Queue.return_value = progress
        with (
            patch.object(
                self.job, "DiarizationTaskRepository", return_value=repository
            ),
            patch.object(self.job.multiprocessing, "get_context", return_value=context),
            patch.object(self.job.os.path, "exists", return_value=True),
        ):
            self.job.process_pending_diarization_tasks_job()
        progress.close.assert_called_once()
        progress.join_thread.assert_called_once()
        process.close.assert_called_once()
        return process, repository

    def test_database_failure_stops_child_and_closes_resources(self):
        process, repository = self._run_job(
            [{"type": "progress", "step": "ALIGNMENT"}],
            update_effect=[RuntimeError("database unavailable"), None],
        )
        process.terminate.assert_called_once()
        process.join.assert_any_call(timeout=5.0)
        self.assertEqual(repository.update_task_step.call_args.kwargs["step"], "ERROR")

    def test_requeued_or_cancelled_task_is_not_logged_as_processing_error(self):
        for step in ["PENDING", "CANCELLED"]:
            with (
                self.subTest(step=step),
                self.assertLogs(self.job.logger, level="INFO") as logs,
            ):
                process, repository = self._run_job(
                    [{"type": "progress", "step": "ALIGNMENT"}],
                    update_effect=[False],
                    stopped_step=step,
                )
            process.terminate.assert_called_once()
            self.assertEqual(repository.update_task_step.call_count, 1)
            self.assertTrue(all(record.levelno < 30 for record in logs.records))

    def test_expired_owner_is_reported_separately_and_requeued(self):
        repository = MagicMock()
        repository.get_task.return_value = SimpleNamespace(
            step="ALIGNMENT", worker_token="owner"
        )
        with self.assertLogs(self.job.logger, level="WARNING") as logs:
            self.job._handle_worker_stop(repository, "task", "owner")
        repository.recover_interrupted_tasks.assert_called_once_with()
        repository.update_task_step.assert_not_called()
        self.assertEqual(logs.records[0].levelno, 30)

    def test_success_waits_for_child_without_terminating_it(self):
        process, repository = self._run_job(
            [{"type": "success", "result": {"language": "pt"}}],
            alive=False,
        )
        process.join.assert_any_call(timeout=5.0)
        process.terminate.assert_not_called()
        self.assertEqual(
            repository.update_task_step.call_args.kwargs["step"], "COMPLETED"
        )

    def test_worker_persists_diarized_before_completed(self):
        _, repository = self._run_job(
            [
                {"type": "progress", "step": "DIARIZED"},
                {"type": "success", "result": {"language": "pt"}},
            ],
            alive=False,
        )
        self.assertEqual(
            [
                call.kwargs["step"]
                for call in repository.update_task_step.call_args_list
            ],
            ["DIARIZED", "COMPLETED"],
        )

    def test_process_start_failure_still_closes_resources(self):
        process, _ = self._run_job([], start_error=RuntimeError("spawn failed"))
        process.join.assert_not_called()
        process.terminate.assert_not_called()

    def test_history_insert_failure_does_not_rollback_task_state(self):
        # step_tracking deliberately does not exist: exercise a real SQL failure
        # inside a real SQLAlchemy savepoint, rather than mocking transaction calls.
        with self.engine.begin() as connection:
            connection.execute(text("DROP TABLE step_tracking"))
        with self.session_factory() as session:
            task = self.model(
                file_path="audio.wav", entity_id="video", entity_type="youtube_video"
            )
            session.add(task)
            session.commit()
            task_id = task.id
        self.repo.DiarizationTaskRepository().update_task_step(
            task_id, "COMPLETED", result_json={"language": "pt"}
        )
        with self.session_factory() as session:
            task = session.get(self.model, task_id)
            self.assertEqual(task.step, "COMPLETED")
            self.assertEqual(task.result_json, {"language": "pt"})


if __name__ == "__main__":
    unittest.main()
