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
            
        diarizer = AudioDiarizer(
            hf_token=hf_token,
            model_size=task.model_size
        )
        
        def on_progress(step_val: str):
            repository.update_task_step(task.id, step=step_val)
            
        result = diarizer.run(
            file_path=file_path,
            language=task.language,
            num_speakers=task.num_speakers,
            min_speakers=task.min_speakers,
            max_speakers=task.max_speakers,
            progress_callback=on_progress
        )
        
        # Converte para dicionário e salva
        result_json = {
            "segments": [s.to_dict() for s in result.segments],
            "language": result.language,
            "duration": result.duration,
            "speakers": result.speakers
        }
        
        repository.update_task_step(task.id, step="COMPLETED", result_json=result_json)
        logger.info(f"Tarefa de diarização {task.id} finalizada com sucesso.")
        
        # Limpeza total da memória (evita que o APScheduler ou threads segurem cache)
        del diarizer
        del result
        del result_json
        
    except Exception as e:
        logger.exception(f"Erro ao processar tarefa {task.id}")
        repository.update_task_step(task.id, step="ERROR", error_message=str(e))
    finally:
        # Garante que o Garbage Collector limpe tudo, mesmo se der erro
        import gc
        gc.collect()
