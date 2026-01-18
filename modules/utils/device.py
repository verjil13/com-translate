from __future__ import annotations

import os
import re
import jaconv
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import Any, Mapping, Optional

from modules.ocr.base import OCREngine
from modules.utils.textblock import TextBlock, adjust_text_line_coordinates
from modules.utils.download import ModelDownloader, ModelID

# --------------------- Device & ONNX Provider Helpers --------------------- #

def torch_available() -> bool:
    """Check if torch is available without raising import errors."""
    try:
        import torch
        return True
    except ImportError:
        return False

def resolve_device(use_gpu: bool, backend: str = "onnx") -> str:
    """Return the best available device string for the specified backend."""
    if not use_gpu:
        return "cpu"
    if backend.lower() == "torch":
        return _resolve_torch_device(fallback_to_onnx=True)
    else:
        return _resolve_onnx_device()

def _resolve_torch_device(fallback_to_onnx: bool = False) -> str:
    try:
        import torch
    except ImportError:
        if fallback_to_onnx:
            return _resolve_onnx_device()
        return "cpu"

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"

def _resolve_onnx_device() -> str:
    """Resolve the best available ONNX device (CPU or CUDA)."""
    providers = ort.get_available_providers()
    if not providers:
        return "cpu"
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    return "cpu"

def get_providers(device: Optional[str] = None) -> list[Any]:
    """Return ONNX providers list (CPU or CUDA only)."""
    try:
        available = ort.get_available_providers()
    except Exception:
        available = []

    if device and device.lower() == "cpu":
        return ["CPUExecutionProvider"]
    if not available:
        return ["CPUExecutionProvider"]

    # Only CUDA and CPU
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def tensors_to_device(data: Any, device: str) -> Any:
    """Move tensors in nested containers to device."""
    try:
        import torch
    except Exception:
        return data

    torch_device = device if device.lower() in ("cpu", "cuda", "mps", "xpu") else "cpu"

    if isinstance(data, torch.Tensor):
        return data.to(torch_device)
    if isinstance(data, Mapping):
        return {k: tensors_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        seq = [tensors_to_device(v, device) for v in data]
        return type(data)(seq) if isinstance(data, tuple) else seq
    return data

# --------------------- Manga OCR Engine --------------------- #

class MangaOCREngineONNX(OCREngine):
    """OCR engine using ONNX-exported MangaOCR models."""
    def __init__(self):
        self.model = None
        self.device = 'cpu'
        self.expansion_percentage = 5
        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.current_file_dir, '..', '..', '..'))

    def initialize(self, device: str = 'cpu', expansion_percentage: int = 5) -> None:
        self.device = device
        self.expansion_percentage = expansion_percentage
        if self.model is None:
            ModelDownloader.get(ModelID.MANGA_OCR_BASE_ONNX)
            self.model = MangaOCRONNX(device=device)

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        for blk in blk_list:
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(
                    blk.xyxy, self.expansion_percentage, self.expansion_percentage, img
                )

            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                cropped_img = img[y1:y2, x1:x2]
                try:
                    blk.text = self.model(cropped_img)
                except Exception:
                    blk.text = ""
            else:
                blk.text = ""
        return blk_list

# --------------------- MangaOCR ONNX Wrapper --------------------- #

class MangaOCRONNX:
    """Wrapper around ONNX encoder/decoder sessions for Manga OCR."""
    def __init__(self, device: str = 'cpu'):
        self.device = device

        encoder_path = ModelDownloader.get_file_path(ModelID.MANGA_OCR_BASE_ONNX, "encoder_model.onnx")
        decoder_path = ModelDownloader.get_file_path(ModelID.MANGA_OCR_BASE_ONNX, "decoder_model.onnx")
        vocab_path = ModelDownloader.get_file_path(ModelID.MANGA_OCR_BASE_ONNX, "vocab.txt")

        providers = get_providers(self.device)
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

        self.vocab = self._load_vocab(vocab_path)
        self.encoder_image_input = self._find_input_name(self.encoder, candidates=('image', 'pixel_values', 'input'))
        self.encoder_output_name = self.encoder.get_outputs()[0].name

        self.decoder_token_input = self._find_input_name(self.decoder, candidates=('token_ids', 'input_ids', 'input'))
        self.decoder_encoder_input = self._find_input_name(self.decoder, candidates=('encoder_hidden_states', 'encoder_outputs', 'encoder_last_hidden_state'))

    def _find_input_name(self, session: ort.InferenceSession, candidates=('input',)) -> str:
        names = [inp.name for inp in session.get_inputs()]
        for cand in candidates:
            for n in names:
                if cand in n:
                    return n
        return names[0]

    def _load_vocab(self, vocab_file: str) -> list:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            return f.read().splitlines()

    def __call__(self, img: np.ndarray) -> str:
        img_in = self._preprocess(img)
        token_ids = self._generate(img_in)
        text = self._decode(token_ids)
        return self._postprocess(text)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError('Empty image passed to MangaOCRONNX')

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        img = img.convert('L').convert('RGB')
        img = img.resize((224, 224), resample=Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose((2, 0, 1))[None]
        return arr

    def _generate(self, image: np.ndarray) -> list:
        encoder_feed = {self.encoder_image_input: image}
        encoder_hidden = self.encoder.run(None, encoder_feed)[0]

        token_ids = [2]
        for _ in range(300):
            decoder_feed = {
                self.decoder_token_input: np.array([token_ids], dtype=np.int64),
                self.decoder_encoder_input: encoder_hidden,
            }
            logits = self.decoder.run(None, decoder_feed)[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            token_ids.append(next_token)
            if next_token == 3:
                break
        return token_ids

    def _decode(self, token_ids: list) -> str:
        text = ''
        for tid in token_ids:
            if tid < 5:
                continue
            if tid < len(self.vocab):
                text += self.vocab[tid]
        return text

    def _postprocess(self, text: str) -> str:
        text = ''.join(text.split())
        text = text.replace('…', '...')
        text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
        return jaconv.h2z(text, ascii=True, digit=True) 
