from typing import Optional

from pydantic import BaseModel, Field

from src.modules.diarization.domain.entities import Segment


class DiarizationPathRequest(BaseModel):
    file_path: str = Field(..., description="Caminho absoluto ou relativo para o arquivo de áudio no servidor.")
    entity_id: Optional[str] = Field(None, description="ID da entidade externa associada ao áudio.")
    entity_type: Optional[str] = Field(None, description="Tipo da entidade externa (ex: 'youtube_video').")
    language: Optional[str] = Field(None,
                                    description="Código do idioma (ex: 'pt', 'en'). Se não informado, será detectado automaticamente.")
    num_speakers: Optional[int] = Field(None, ge=1, description="Número exato de falantes no áudio.")
    min_speakers: Optional[int] = Field(None, ge=1, description="Número mínimo de falantes.")
    max_speakers: Optional[int] = Field(None, ge=1, description="Número máximo de falantes.")
    model_size: str = Field("large-v2",
                            description="Tamanho do modelo Whisper (ex: 'tiny', 'base', 'small', 'medium', 'large-v2').")


class DiarizationResponse(BaseModel):
    file_path: str
    language: str
    duration: float
    speakers: list[str]
    segments: list[Segment]


class DiarizationTaskResponse(BaseModel):
    task_id: str
    step: str
    message: str
