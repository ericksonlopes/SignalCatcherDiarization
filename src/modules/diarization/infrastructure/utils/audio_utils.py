import numpy as np
import torch
import whisperx


def get_best_device() -> str:
    """Returns 'cuda' if available, otherwise 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_whisperx_audio(file_path: str) -> np.ndarray:
    """Loads audio file into a numpy array for whisperx processing."""
    return whisperx.load_audio(file_path)
