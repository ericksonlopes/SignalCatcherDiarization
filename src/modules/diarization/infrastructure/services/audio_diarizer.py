import logging
import os

import numpy as np
import torch
import whisperx

from src.core.services.model_loader_service import model_loader
from src.modules.diarization.domain.entities import DiarizationResult, Segment
from src.modules.diarization.infrastructure.utils.audio_utils import get_best_device, load_whisperx_audio

logger = logging.getLogger(__name__)

# Global torch configuration to avoid RuntimeError: "set_num_threads is not allowed after parallel work has started"
_device_type = get_best_device()
if _device_type == "cpu":
    _cpu_count = os.cpu_count() or 4
    try:
        torch.set_num_threads(_cpu_count)
        torch.set_num_interop_threads(max(1, _cpu_count // 2))
        logger.info("Global torch CPU config: threads=%d", _cpu_count)
    except RuntimeError:
        # Already set or parallel work started
        pass


class AudioDiarizer:
    def __init__(
            self,
            hf_token: str,
            model_size: str = "large-v2",
            batch_size: int = 16,
    ):
        self.hf_token = hf_token
        self.model_size = model_size
        self._device = _device_type
        self._compute_type = "float16" if self._device == "cuda" else "int8"

        self.batch_size = batch_size
        if self._device == "cpu":
            self.batch_size = min(self.batch_size, 4)
            logger.info("CPU mode enabled: batch_size=%d (limitado para evitar OOM)", self.batch_size)

    def _transcribe(self, audio: np.ndarray, language: str | None) -> dict:
        logger.info(
            "[1/3] Transcription starting (model=%s, device=%s, compute=%s)",
            self.model_size,
            self._device,
            self._compute_type,
        )

        logger.info("[1/3] Getting/Loading whisperx model...")
        model = model_loader.get_whisper_model(
            self.model_size,
            self._device,
            compute_type=self._compute_type,
            language=language,
            threads=_cpu_count if self._device == "cpu" else 4,
        )

        logger.info(
            "[1/3] Model ready, starting transcription (batch_size=%d)...",
            self.batch_size,
        )

        result = model.transcribe(audio, batch_size=self.batch_size, print_progress=True)
        logger.info(
            "[1/3] Transcription complete: %d segments, language=%s",
            len(result.get("segments", [])),
            result.get("language", "?"),
        )
        return result

    def _align(self, result: dict, audio: np.ndarray, language: str | None) -> dict:
        logger.info("[2/3] Word alignment starting")
        lang = language or result.get("language", "en")
        try:
            model_a, metadata = model_loader.get_align_model(language_code=lang, device=self._device)
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self._device,
                return_char_alignments=False,
            )
            logger.info("[2/3] Alignment complete")
        except Exception as e:
            logger.warning("[2/3] Skipped alignment: %s", e)
        return result

    def _diarize(
            self,
            audio: np.ndarray,
            result: dict,
            num_speakers: int | None,
            min_speakers: int | None,
            max_speakers: int | None,
    ) -> tuple[list[Segment], str]:
        logger.info("[3/3] Speaker diarization starting")
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers

        logger.info("[3/3] Getting/Loading diarization pipeline...")
        diarize_model = model_loader.get_diarization_pipeline(
            hf_token=self.hf_token,
            device=self._device,
        )

        logger.info("[3/3] Running diarization with kwargs=%s...", kwargs)
        diarize_segments = diarize_model(audio, **kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = [
            Segment.create(
                speaker=seg.get("speaker", "UNKNOWN"),
                start=round(seg["start"], 3),
                end=round(seg["end"], 3),
                text=seg["text"].strip(),
            )
            for seg in result["segments"]
        ]
        logger.info("[3/3] Diarization complete: %d segments", len(segments))
        return segments, result.get("language", "?")

    def run(
            self,
            file_path: str,
            language: str | None = None,
            num_speakers: int | None = None,
            min_speakers: int | None = None,
            max_speakers: int | None = None,
            progress_callback: callable = None,
    ) -> DiarizationResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Starting optimized diarization pipeline for: %s", file_path)

        # Load audio once
        audio = load_whisperx_audio(file_path)
        logger.info("Audio loaded, shape=%s", audio.shape)

        if progress_callback:
            progress_callback("TRANSCRIPTION")
        result_trans = self._transcribe(audio, language)
        model_loader.unload_whisper() # FREE RAM
        
        if progress_callback:
            progress_callback("ALIGNMENT")
        result_aligned = self._align(result_trans, audio, language)
        model_loader.unload_align() # FREE RAM

        if progress_callback:
            progress_callback("DIARIZATION")
        segments, lang = self._diarize(audio, result_aligned, num_speakers, min_speakers, max_speakers)
        model_loader.unload_diarization() # FREE RAM

        return DiarizationResult(
            segments=segments,
            language=language or lang,
            file_path=file_path,
        )
