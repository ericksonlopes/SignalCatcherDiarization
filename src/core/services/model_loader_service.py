import whisperx


class ModelLoader:
    def __init__(self):
        self._whisper_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarization_pipeline = None

    def get_whisper_model(self, model_size: str, device: str, compute_type: str, language: str | None = None):
        """Loads and caches the whisper transcription model."""
        if self._whisper_model is None:
            self._whisper_model = whisperx.load_model(
                model_size,
                device,
                compute_type=compute_type,
                language=language
            )
        return self._whisper_model

    def get_align_model(self, language_code: str, device: str):
        """Loads and caches the alignment model."""
        if self._align_model is None:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=device
            )
        return self._align_model, self._align_metadata

    def get_diarization_pipeline(self, hf_token: str, device: str):
        """Loads and caches the diarization pipeline."""
        if self._diarization_pipeline is None:
            from whisperx.diarize import DiarizationPipeline
            self._diarization_pipeline = DiarizationPipeline(
                token=hf_token,
                device=device
            )
        return self._diarization_pipeline


# Singleton instance
model_loader = ModelLoader()
