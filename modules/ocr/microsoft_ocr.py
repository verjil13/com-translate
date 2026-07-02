import numpy as np
import cv2
import base64
import os
import threading
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import PaddleOCRChatHandler

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from app.ui.settings.settings_page import SettingsPage
import gc
import torch
import re
import math
from spandrel import ModelLoader, ImageModelDescriptor
from pathlib import Path

_MODEL = None
_DEVICE = None

class MicrosoftOCR(OCREngine):
    """OCR engine using PaddleOCR-VL GGUF (llama.cpp)"""
    # -------------------------
    # INIT
    # -------------------------
    def __init__(self):
        self.api_key = None
        self.expansion_percentage = 5
        self.model = ""

        self.llm = None  # ← локальная модель

    # -------------------------
    # INIT
    # -------------------------
    def initialize(
        self,
        settings: SettingsPage,
        model: str = "Gemini-2.0-Flash",
        expansion_percentage: int = 5,
    ) -> None:
        self.expansion_percentage = expansion_percentage

        if self.llm is not None:
            print("🔄 Unloading previous model...")
            self.unload_model()

        BASE_DIR = os.getcwd()

        MODEL_PATH = os.path.join(BASE_DIR, "models/PaddleOCR-VL-1.5", "PaddleOCR-VL-1.5-BF16.gguf")
        MMPROJ_PATH = os.path.join(BASE_DIR, "models/PaddleOCR-VL-1.5", "mmproj-BF16.gguf")

        self.llm = Llama(
            model_path=MODEL_PATH,
            chat_handler=PaddleOCRChatHandler(
                clip_model_path=MMPROJ_PATH,
                verbose=False,
            ),
            n_gpu_layers=-1,
            n_ctx=1280,
            n_batch=256,
            verbose=False,
        )

    def unload_model(self):
        if self.llm is not None:
            print("🔄 Unloading previous model...")

            # Принудительный cleanup llama.cpp
            try:
                if hasattr(self.llm, "close"):
                    self.llm.close()
            except:
                pass

            try:
                if hasattr(self.llm, "_model"):
                    self.llm._model = None
                if hasattr(self.llm, "ctx"):
                    self.llm.ctx = None
            except:
                pass

            try:
                del self.llm
            except:
                pass

            self.llm = None
            self.current_model = None

            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
    # -------------------------
    # PUBLIC API
    # -------------------------
    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]):
        return self._process_by_blocks(img, blk_list)

    # -------------------------
    # RESIZE FUNCTION
    # -------------------------
    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        max_area = 360 * 360
        min_area = 16 * 16

        current_area = w * h

        # вычисляем scale по площади
        if current_area > max_area:
            scale = math.sqrt(max_area / current_area)
        elif current_area < min_area:
            scale = math.sqrt(min_area / current_area)
        else:
            scale = 1.0

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        if new_w == w and new_h == h:
            return img

        # interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        interpolation = cv2.INTER_LANCZOS4
        return cv2.resize(
            img,
            (new_w, new_h),
            interpolation=interpolation,
        )

    # -------------------------
    # BLOCK PROCESSING
    # -------------------------
    def _cropped(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        new_w = max(1, int(w * 2))
        new_h = max(1, int(h * 2))

        interpolation = cv2.INTER_LANCZOS4 

        return cv2.resize(
            img,
            (new_w, new_h),
            interpolation=interpolation,
        )

    def _process_by_blocks(self, img: np.ndarray, blk_list: list[TextBlock]):

        for blk in blk_list:

            if blk.xyxy is not None:
                x1, y1, x2, y2 = blk.xyxy
            elif blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(
                    blk.xyxy,
                    self.expansion_percentage,
                    self.expansion_percentage,
                    img,
                )

            if x1 >= x2 or y1 >= y2:
                continue

            image = img[y1:y2, x1:x2]
            # image = self._cropped(image)
            # cropped = self.upscale_hat(
            #    cropped,
            #    model_path=r"H:\com-translate\models\upscale\2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.safetensors",
            #    scale=2,
            # )
            # 4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.safetensors
            # 2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.safetensors
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = cv2.GaussianBlur(image, (3, 3), 0)
            ###
            image = image.copy()
            image[image < 30] = 0
            image[image > 225] = 255
            ###
            # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            image = self._resize_if_needed(image)

            # OUTPUT_DIR = Path(r"G:\Torrent\Manga\test\out")
            # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            # filename = f"{time.time_ns()}.png"
            # cv2.imwrite(str(OUTPUT_DIR / filename), image)

            start = time.time()
            blk.text = self._get_ocr(image)

            elapsed = time.time() - start
            if elapsed > 10:
                print(f"BLOCK TOO SLOW: {elapsed:.2f}s")

        return blk_list

    # -------------------------
    # OCR CORE (THREAD + TIMEOUT)
    # -------------------------
    def _get_ocr(self, image: np.ndarray) -> str:

        result = [""]
        error = [None]

        def worker():
            try:
                # --- PIL → base64 ---
                buffer = cv2.imencode(".jpg", image)[1]
                base64_img = base64.b64encode(buffer).decode("utf-8")

                data_uri = f"data:image/jpeg;base64,{base64_img}"

                response = self.llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                     "image_url": {"url": data_uri},
                                    
                                },
                                {
                                    "type": "text",
                                    "text": "OCR:",
                                },
                            ],
                        }
                    ],
                    temperature=0,
                    max_tokens=1024,
                )

                result[0] = response["choices"][0]["message"]["content"].strip()

            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=30)

        if thread.is_alive():
            print("⛔ OCR TIMEOUT (30s)")
            return ""

        if error[0]:
            print("⛔ OCR ERROR:", error[0])
            return ""

        text = result[0]

        # cleanup
        for prefix in ["OCR:", "OCR :", "Text:", "Answer:", "Answer :"]:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break

        text = self._normalize_text(text)
        text = self.remove_repeated_patterns(text)
        print(f"[OCR] {repr(text[:100])}")

        return text

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        # заменяем переносы строк и табы
        text = text.replace("\n", " ").replace("\t", " ")

        # убираем множественные пробелы
        text = " ".join(text.split())

        return text.strip()

    def remove_repeated_patterns(
        self,
        text: str,
        max_pattern_len: int = 10,
        single_char_limit: int = 5,
        sequence_limit: int = 3,
    ) -> str:

        if not text:
            return ""

        max_pattern_len = int(max_pattern_len)
        single_char_limit = int(single_char_limit)
        sequence_limit = int(sequence_limit)

        for pattern_len in range(max_pattern_len, 0, -1):

            limit = single_char_limit if pattern_len == 1 else sequence_limit

            regex = re.compile(
                rf"(.{{{pattern_len}}})(?:\1){{{limit},}}", flags=re.DOTALL
            )

            replacement = r"\1" * limit

            while True:
                new_text = regex.sub(replacement, text)

                if new_text == text:
                    break

                text = new_text

        return text

    def upscale_hat(
        self,
        img_bgr: np.ndarray,
        model_path: str,
        scale: int = 2,
        tile: int = 256,
        overlap: int = 32,
    ) -> np.ndarray:
        """
        One-shot HAT upscaler (Spandrel)
        model loads only once (cached globally)
        """

        global _MODEL, _DEVICE

        # =========================
        # lazy load model
        # =========================
        if _MODEL is None:
            print("Loading model...")

            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = ModelLoader().load_from_file(model_path)
            assert isinstance(model, ImageModelDescriptor)

            _MODEL = model.to(_DEVICE)
            _MODEL.eval()

            print("Using device:", _DEVICE)

        model = _MODEL
        device = _DEVICE

        # =========================
        # grid
        # =========================
        def build_grid(size, tile, overlap):
            stride = tile - overlap
            coords = []
            pos = 0
            while True:
                if pos + tile >= size:
                    coords.append(max(0, size - tile))
                    break
                coords.append(pos)
                pos += stride
            return coords

        h, w = img_bgr.shape[:2]

        out_h, out_w = h * scale, w * scale

        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 3), dtype=np.float32)

        xs = build_grid(w, tile, overlap)
        ys = build_grid(h, tile, overlap)

        with torch.no_grad():

            for y in ys:
                for x in xs:

                    patch = img_bgr[y : y + tile, x : x + tile]
                    ph, pw = patch.shape[:2]

                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                    tensor = (
                        torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
                        / 255.0
                    )

                    tensor = tensor.to(device)

                    sr = model(tensor)

                    sr = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    sr = np.clip(sr, 0, 1)

                    oh = ph * scale
                    ow = pw * scale

                    sr = sr[:oh, :ow]

                    oy = y * scale
                    ox = x * scale

                    output[oy : oy + oh, ox : ox + ow] += sr
                    weight[oy : oy + oh, ox : ox + ow] += 1.0

        output /= np.maximum(weight, 1e-8)
        output = (output * 255).astype(np.uint8)

        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output
