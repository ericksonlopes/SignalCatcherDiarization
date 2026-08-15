import os
import logging
from src.core.config.settings import settings
from src.modules.diarization.infrastructure.repositories.diarization_task_repository import DiarizationTaskRepository
from src.modules.diarization.infrastructure.services.audio_diarizer import AudioDiarizer

logger = logging.getLogger(__name__)

def process_pending_diarization_tasks_job():
    """Busca tarefas pendentes na fila e as processa uma por uma."""
    logger.debug("Executando job de diarização...")
    
    repository = DiarizationTaskRepository()
    
    # Busca apenas 1 tarefa por vez para evitar concorrência ou sobrecarga de memória (o whisper usa muita RAM/VRAM)
    pending_tasks = repository.get_pending_tasks(limit=1)
    
    if not pending_tasks:
        return
        
    task = pending_tasks[0]
    
    logger.info(f"Iniciando processamento da tarefa de diarização {task.id}")
    
    # Marca como PROCESSING (Wait, the user said: pending -> transcription -> alignment -> diarization -> completed)
    # The diarizer will immediately emit TRANSCRIPTION, but before that we can set it to TRANSCRIPTION or let the callback handle it.
    # Actually, we can just let it stay PENDING until the first callback or we can set it to PROCESSING. Let's stick to TRANSCRIPTION if possible.
    # But wait, the previous code had PROCESSING. Let's just pass PROCESSING or TRANSCRIPTION. Let's use TRANSCRIPTION to start.
    repository.update_task_step(task.id, step="TRANSCRIPTION")
    
    try:
        file_path = task.file_path
        # Map samba path (/youtube/..., /spotify/...) to local or container path
        if settings.DOWNLOAD_YOUTUBE_PATH:
            rel_path = file_path.lstrip("/\\")
            rel_path = rel_path.replace("/", os.sep)
            file_path = os.path.join(settings.DOWNLOAD_YOUTUBE_PATH, rel_path)
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
        hf_token = settings.HF_TOKEN
        if not hf_token:
            raise ValueError("HF_TOKEN não está configurado.")
            
        import multiprocessing
        import traceback
        import queue
        
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
        
        def _worker(k_dict, p_queue):
            try:
                from src.modules.diarization.infrastructure.services.audio_diarizer import AudioDiarizer
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

        p = ctx.Process(target=_worker, args=(kwargs_dict, progress_queue))
        p.start()
        
        # Fica ouvindo a fila até o processo terminar ou enviar success/error
        while True:
            try:
                # Aguarda uma mensagem, com timeout para checar se o processo morreu abruptamente
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
                # Se a fila ta vazia, checa se o processo ainda ta vivo
                if not p.is_alive():
                    # Processo morreu abruptamente (ex: OOM Killer sem conseguir mandar erro)
                    raise RuntimeError("O processo de diarização morreu inesperadamente (possível falta de memória / OOM Killer).")
                    
        p.join()
        
    except Exception as e:
        logger.exception(f"Erro ao processar tarefa {task.id}")
        repository.update_task_step(task.id, step="ERROR", error_message=str(e))
    finally:
        # Garante que o Garbage Collector limpe tudo, mesmo se der erro
        import gc
        gc.collect()
