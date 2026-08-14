import logging
import os
import shutil
import tempfile
from typing import Optional

from fastapi import HTTPException, UploadFile, status

from src.modules.diarization.application.dtos import DiarizationPathRequest, DiarizationTaskResponse
from src.modules.diarization.infrastructure.repositories.diarization_task_repository import DiarizationTaskRepository

logger = logging.getLogger(__name__)


class EnqueueDiarizationUseCase:
    """Caso de uso para enfileirar uma diarização de áudio (background)."""

    def __init__(self):
        self.repository = DiarizationTaskRepository()

    def execute_from_path(self, request: DiarizationPathRequest) -> DiarizationTaskResponse:
        """Enfileira a diarização a partir do caminho de um arquivo existente."""
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=status.HTTP_444_NOT_FOUND if hasattr(status,
                                                                 "HTTP_444_NOT_FOUND") else status.HTTP_404_NOT_FOUND,
                detail=f"Arquivo de áudio não encontrado no caminho: {request.file_path}",
            )

        task = self.repository.create_task(
            file_path=request.file_path,
            entity_id=request.entity_id,
            entity_type=request.entity_type,
            language=request.language,
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            model_size=request.model_size,
        )

        return DiarizationTaskResponse(
            task_id=task.id,
            step=task.step,
            message="Tarefa enfileirada com sucesso."
        )

    async def execute_from_file(
            self,
            file: UploadFile,
            entity_id: Optional[str] = None,
            entity_type: Optional[str] = None,
            language: Optional[str] = None,
            num_speakers: Optional[int] = None,
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None,
            model_size: str = "large-v2",
    ) -> DiarizationTaskResponse:
        """Salva o arquivo de upload e enfileira a diarização."""
        ext = os.path.splitext(file.filename or "")[1] or ".wav"

        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=upload_dir)
        temp_path = temp_file.name

        try:
            temp_file.close()
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            logger.info("Arquivo salvo para processamento assíncrono em: %s", temp_path)

            task = self.repository.create_task(
                file_path=temp_path,
                entity_id=entity_id,
                entity_type=entity_type,
                language=language,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                model_size=model_size,
            )

            return DiarizationTaskResponse(
                task_id=task.id,
                step=task.step,
                message="Arquivo enviado e tarefa enfileirada com sucesso."
            )

        except Exception as e:
            logger.exception("Erro ao processar o arquivo de upload %s: %s", file.filename, e)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Falha no processamento do arquivo de áudio enviado: {str(e)}",
            )


class GetDiarizationTaskUseCase:
    """Caso de uso para consultar o status de uma tarefa."""

    def __init__(self):
        self.repository = DiarizationTaskRepository()

    def execute(self, task_id: str) -> dict:
        task = self.repository.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada.")

        return {
            "task_id": task.id,
            "step": task.step,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "error_message": task.error_message,
            "result": task.result_json,
        }
