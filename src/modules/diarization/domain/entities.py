from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Segment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    speaker: str
    start: float
    end: float
    text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
        }

    @classmethod
    def create(cls, speaker: str, start: float, end: float, text: str) -> "Segment":
        return cls(speaker=speaker, start=start, end=end, text=text)

    model_config = {"from_attributes": True}


class DiarizationResult(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    language: str = "unknown"
    file_path: str = ""

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        return max(s.end for s in self.segments)

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments})
