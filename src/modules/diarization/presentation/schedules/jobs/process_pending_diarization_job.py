import logging
import multiprocessing
import os
import queue
import traceback
from time import monotonic
from uuid import uuid4

from src.core.config.settings import settings
from src.modules.diarization.infrastructure.repositories.diarization_task_repository import (
    DiarizationTaskRepository,
)
from src.modules.diarization.infrastructure.services.execution_lock import (
    DiarizationExecutionLock,
)

logger = logging.getLogger(__name__)


class WorkerStopped(Exception):
    """The task no longer authorizes this subprocess to continue."""


def _handle_worker_stop(repository, task_id: str, worker_token: str) -> None:
    current = repository.get_task(task_id)
    if current is None:
        logger.info("Processamento %s encerrado: tarefa removida.", task_id)
    elif current.step == "PENDING":
        logger.info("Processamento %s interrompido: tarefa devolvida à fila.", task_id)
    elif current.step == "CANCELLED":
        logger.info("Processamento %s interrompido: tarefa cancelada.", task_id)
    elif current.worker_token != worker_token:
        logger.info("Processamento %s encerrado: execução substituída.", task_id)
    else:
        # A heartbeat timeout is distinct from a user's queue/cancel action.
        # Ownership guards prevent this from overwriting a concurrent new attempt.
        logger.warning(
            "Processamento %s encerrado: prazo de atividade do worker expirou.", task_id
        )
        repository.recover_interrupted_tasks()


class StageProgressReporter:
    """Send at most one update per two seconds, plus the first and final values."""

    def __init__(self, progress_queue):
        self.queue = progress_queue
        self.step = None
        self.percent = -1
        self.sent_at = 0.0

    def __call__(self, step: str, percent: float):
        value = max(0, min(100, int(percent)))
        now = monotonic()
        if step != self.step:
            self.step, self.percent, self.sent_at = step, -1, 0.0
        if value <= self.percent:
            return
        if self.percent >= 0 and value < 100 and now - self.sent_at < 2:
            return
        self.queue.put({"type": "stage_progress", "step": step, "percent": value})
        self.percent, self.sent_at = value, now


def _diarization_worker(k_dict, p_queue):
    try:
        # Spawn starts with a fresh logging configuration. Use the same handler
        # and level filtering as the API before loading the inference libraries.
        from src.core.logger.logger import logger as app_logger

        logging.basicConfig(
            handlers=[app_logger.get_intercept_handler()],
            level=logging.INFO,
            force=True,
        )
        from src.modules.diarization.infrastructure.services.audio_diarizer import (
            AudioDiarizer,
        )

        diarizer = AudioDiarizer(
            hf_token=k_dict["hf_token"], model_size=k_dict["model_size"]
        )

        def on_prog(step_val: str):
            p_queue.put({"type": "progress", "step": step_val})

        res = diarizer.run(
            file_path=k_dict["file_path"],
            language=k_dict["language"],
            num_speakers=k_dict["num_speakers"],
            min_speakers=k_dict["min_speakers"],
            max_speakers=k_dict["max_speakers"],
            progress_callback=on_prog,
            stage_progress_callback=StageProgressReporter(p_queue),
        )

        res_json = {
            "segments": [s.to_dict() for s in res.segments],
            "language": res.language,
            "duration": res.duration,
            "speakers": res.speakers,
        }
        p_queue.put({"type": "success", "result": res_json})
    except Exception as exc:
        p_queue.put(
            {"type": "error", "error": str(exc), "traceback": traceback.format_exc()}
        )


def process_pending_diarization_tasks_job():
    with DiarizationExecutionLock() as execution:
        if not execution.acquired:
            logger.debug("Outro worker já está executando uma diarização.")
            return
        _process_pending_diarization_tasks_job(execution)


def _process_pending_diarization_tasks_job(execution):
    """Busca tarefas pendentes na fila e as processa uma por uma."""
    logger.debug("Executando job de diarização...")

    repository = DiarizationTaskRepository()
    # An exclusive execution lock proves the previous owner has exited. Recover
    # even unexpired or legacy task leases, which otherwise appear active after restart.
    recovered = repository.recover_interrupted_tasks(force=True)
    if recovered:
        logger.info(
            "%s tarefas interrompidas devolvidas à fila para reprocessamento.",
            recovered,
        )

    pending_tasks = repository.get_pending_tasks(limit=1)

    if not pending_tasks:
        return

    task = pending_tasks[0]

    logger.info(f"Iniciando processamento da tarefa de diarização {task.id}")
    p = None
    progress_queue = None
    worker_succeeded = False
    worker_token = str(uuid4())
    try:
        if not repository.claim_task(task.id, worker_token):
            return
        file_path = task.file_path
        if settings.DOWNLOAD_YOUTUBE_PATH:
            rel_path = file_path.lstrip("/\\")
            rel_path = rel_path.replace("/", os.sep)
            file_path = os.path.join(settings.DOWNLOAD_YOUTUBE_PATH, rel_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        hf_token = settings.HF_TOKEN
        if not hf_token:
            raise ValueError("HF_TOKEN não está configurado.")

        ctx = multiprocessing.get_context("spawn")
        progress_queue = ctx.Queue()

        kwargs_dict = {
            "file_path": file_path,
            "language": task.language,
            "num_speakers": task.num_speakers,
            "min_speakers": task.min_speakers,
            "max_speakers": task.max_speakers,
            "hf_token": hf_token,
            "model_size": task.model_size,
        }

        p = ctx.Process(target=_diarization_worker, args=(kwargs_dict, progress_queue))
        p.start()
        last_heartbeat = monotonic()

        while True:
            if monotonic() - last_heartbeat >= 5:
                execution.ensure_owned()
                if not repository.renew_task_lease(task.id, worker_token):
                    raise WorkerStopped()
                last_heartbeat = monotonic()
            try:
                msg = progress_queue.get(timeout=5.0)
                if msg["type"] == "progress":
                    if not repository.update_task_step(
                        task.id, step=msg["step"], worker_token=worker_token
                    ):
                        raise WorkerStopped()
                elif msg["type"] == "stage_progress":
                    repository.update_task_progress(
                        task.id, msg["step"], msg["percent"], worker_token=worker_token
                    )
                elif msg["type"] == "success":
                    if not repository.update_task_step(
                        task.id,
                        step="COMPLETED",
                        result_json=msg["result"],
                        worker_token=worker_token,
                    ):
                        raise WorkerStopped()
                    logger.info(
                        f"Tarefa de diarização {task.id} finalizada com sucesso."
                    )
                    worker_succeeded = True
                    break
                elif msg["type"] == "error":
                    logger.error(f"Erro no worker: {msg['traceback']}")
                    raise RuntimeError(msg["error"])
            except queue.Empty:
                if not p.is_alive():
                    raise RuntimeError(
                        "O processo de diarização morreu inesperadamente (possível falta de memória / OOM Killer)."
                    )

    except WorkerStopped:
        _handle_worker_stop(repository, task.id, worker_token)
    except Exception as e:
        logger.exception(f"Erro ao processar tarefa {task.id}")
        repository.update_task_step(
            task.id, step="ERROR", error_message=str(e), worker_token=worker_token
        )
    finally:
        # Do not leave inference running if persisting progress or results fails.
        try:
            if p is not None:
                try:
                    if p.pid is not None:
                        if worker_succeeded:
                            p.join(timeout=5.0)
                        if p.is_alive():
                            p.terminate()
                            p.join(timeout=5.0)
                        if p.is_alive():
                            p.kill()
                            p.join()
                        else:
                            p.join()
                finally:
                    p.close()
        finally:
            if progress_queue is not None:
                progress_queue.close()
                progress_queue.join_thread()
