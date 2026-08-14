import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, JSON, Integer
from src.core.database.connector import Base

def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class DiarizationTaskModel(Base):
    __tablename__ = "diarization"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(String, nullable=False)
    step = Column(String, nullable=False, default="PENDING", index=True)  # PENDING, TRANSCRIPTION, ALIGNMENT, DIARIZATION, COMPLETED, ERROR
    
    # Referência Externa
    entity_id = Column(String, nullable=True)
    entity_type = Column(String, nullable=True)
    
    # Parâmetros
    language = Column(String, nullable=True)
    num_speakers = Column(Integer, nullable=True)
    min_speakers = Column(Integer, nullable=True)
    max_speakers = Column(Integer, nullable=True)
    model_size = Column(String, nullable=False, default="large-v2")
    
    # Resultado/Erro
    result_json = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)
