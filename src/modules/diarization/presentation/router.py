from typing import Annotated, Optional

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile, status

from src.modules.diarization.application.dtos import (
    DiarizationPathRequest,
    DiarizationTaskResponse,
)
from src.modules.diarization.application.use_cases import EnqueueDiarizationUseCase, GetDiarizationTaskUseCase

router = APIRouter()


def get_enqueue_diarization_use_case() -> EnqueueDiarizationUseCase:
    return EnqueueDiarizationUseCase()


def get_task_use_case() -> GetDiarizationTaskUseCase:
    return GetDiarizationTaskUseCase()


EnqueueUseCaseDep = Annotated[EnqueueDiarizationUseCase, Depends(get_enqueue_diarization_use_case)]
GetTaskUseCaseDep = Annotated[GetDiarizationTaskUseCase, Depends(get_task_use_case)]


@router.post(
    "/process-file",
    response_model=DiarizationTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enfileirar diarização de áudio enviado por upload",
    description="Realiza o upload do arquivo e enfileira a tarefa para execução em background.",
)
async def diarize_uploaded_file(
        file: Annotated[UploadFile, File(description="Arquivo de áudio para diarização (.wav, .mp3, .m4a, etc.)")],
        use_case: EnqueueUseCaseDep,
        entity_id: Annotated[Optional[str], Form(description="ID da entidade externa associada ao áudio.")] = None,
        entity_type: Annotated[Optional[str], Form(description="Tipo da entidade externa (ex: 'youtube_video').")] = None,
        language: Annotated[Optional[str], Form(description="Código ISO do idioma (ex: 'pt', 'en').")] = None,
        num_speakers: Annotated[Optional[int], Form(ge=1, description="Número exato de falantes no áudio.")] = None,
        min_speakers: Annotated[Optional[int], Form(ge=1, description="Número mínimo de falantes.")] = None,
        max_speakers: Annotated[Optional[int], Form(ge=1, description="Número máximo de falantes.")] = None,
        model_size: Annotated[str, Form(description="Tamanho do modelo Whisper.")] = "large-v2",
):
    return await use_case.execute_from_file(
        file=file,
        entity_id=entity_id,
        entity_type=entity_type,
        language=language,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        model_size=model_size,
    )


@router.post(
    "/process-path",
    response_model=DiarizationTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enfileirar diarização de áudio por caminho local",
    description="Enfileira a diarização a partir do caminho de um arquivo de áudio já presente no servidor.",
)
def diarize_by_path(
        request: Annotated[DiarizationPathRequest, Body(description="Parâmetros de configuração.")],
        use_case: EnqueueUseCaseDep,
):
    return use_case.execute_from_path(request)


@router.get(
    "/task/{task_id}",
    status_code=status.HTTP_200_OK,
    summary="Consultar status de uma tarefa de diarização",
)
def get_task_status(
        task_id: str,
        use_case: GetTaskUseCaseDep,
):
    return use_case.execute(task_id)
