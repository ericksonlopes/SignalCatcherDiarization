import logging
import multiprocessing
import os
import queue
import traceback

from src.core.config.settings import settings
from src.modules.diarization.infrastructure.repositories.diarization_task_repository import DiarizationTaskRepository
from src.modules.diarization.infrastructure.services.audio_diarizer import AudioDiarizer

logger = logging.getLogger(__name__)


def _diarization_worker(k_dict, p_queue):
    try:

        diarizer = AudioDiarizer(
            hf_token=k_dict["hf_token"],
            model_size=k_dict["model_size"]
        )

        def on_prog(step_val: str):
            p_queue.put({"type": "progress", "step": step_val})

        res = diarizer.run(
            file_path=k_dict["file_path"],
            language=k_dict["language"],
            num_speakers=k_dict["num_speakers"],
            min_speakers=k_dict["min_speakers"],
            max_speakers=k_dict["max_speakers"],
            progress_callback=on_prog
        )

        res_json = {
            "segments": [s.to_dict() for s in res.segments],
            "language": res.language,
            "duration": res.duration,
            "speakers": res.speakers
        }
        p_queue.put({"type": "success", "result": res_json})
    except Exception as exc:
        p_queue.put({"type": "error", "error": str(exc), "traceback": traceback.format_exc()})

def process_pending_diarization_tasks_job():
    """Busca tarefas pendentes na fila e as processa uma por uma."""
    logger.debug("Executando job de diarização...")
    
    repository = DiarizationTaskRepository()
    
    pending_tasks = repository.get_pending_tasks(limit=1)
    
    if not pending_tasks:
        return
        
    task = pending_tasks[0]
    
    logger.info(f"Iniciando processamento da tarefa de diarização {task.id}")
    repository.update_task_step(task.id, step="TRANSCRIPTION")
    
    try:
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
            "model_size": task.model_size
        }

        p = ctx.Process(target=_diarization_worker, args=(kwargs_dict, progress_queue))
        p.start()
        
        while True:
            try:
                msg = progress_queue.get(timeout=5.0)
                if msg["type"] == "progress":
                    repository.update_task_step(task.id, step=msg["step"])
                elif msg["type"] == "success":
                    repository.update_task_step(task.id, step="COMPLETED", result_json=msg["result"])
                    logger.info(f"Tarefa de diarização {task.id} finalizada com sucesso.")
                    break
                elif msg["type"] == "error":
                    logger.error(f"Erro no worker: {msg['traceback']}")
                    raise RuntimeError(msg["error"])
            except queue.Empty:
                if not p.is_alive():
                    raise RuntimeError("O processo de diarização morreu inesperadamente (possível falta de memória / OOM Killer).")
                    
        p.join()
        
    except Exception as e:
        logger.exception(f"Erro ao processar tarefa {task.id}")
        repository.update_task_step(task.id, step="ERROR", error_message=str(e))
    finally:
        import gc
        gc.collect()
