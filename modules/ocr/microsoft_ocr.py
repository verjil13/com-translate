from PIL import Image
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

        MODEL_PATH = os.path.join(BASE_DIR, "models\PaddleOCR-VL-1.5", "PaddleOCR-VL-1.5-BF16.gguf")
        MMPROJ_PATH = os.path.join(BASE_DIR, "models\PaddleOCR-VL-1.5", "mmproj-BF16.gguf")

        # MODEL_PATH = os.path.join(BASE_DIR, "models\PaddleOCR-VL-1.5-Q8", "PaddleOCR-VL-1.5-Q8_0.gguf")
        # MMPROJ_PATH = os.path.join(BASE_DIR, "models\PaddleOCR-VL-1.5-Q8", "mmproj-PaddleOCR-VL-1.5-Q8_0.gguf")

        # MODEL_PATH = os.path.join(BASE_DIR, "models\PaddleOCR-VL-1.5-Q4", "PaddleOCR-VL-1.5-Q4_K_M.gguf")
        # MMPROJ_PATH = os.path.join(BASE_DIR, "models\PaddleOCR-VL-1.5-Q4", "mmproj-PaddleOCR-VL-1.5-Q4_1.gguf")

        self.llm = Llama(
            model_path=MODEL_PATH,
            chat_handler=PaddleOCRChatHandler(
                clip_model_path=MMPROJ_PATH,
            ),
            n_gpu_layers=-1,
            n_ctx=0,
            n_batch=1024,
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
    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        max_size = 768

        if w <= max_size and h <= max_size:
            return img.resize((int(w), int(h)), Image.BILINEAR)

        scale = min(max_size / w, max_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return img.resize((new_w, new_h), Image.BILINEAR)

    # -------------------------
    # BLOCK PROCESSING
    # -------------------------

    def _process_by_blocks(self, img: np.ndarray, blk_list: list[TextBlock]):

        for blk in blk_list:

            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = map(int, blk.bubble_xyxy)
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(
                    blk.xyxy,
                    self.expansion_percentage,
                    self.expansion_percentage,
                    img,
                )

            if x1 >= x2 or y1 >= y2:
                continue

            cropped = img[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cropped).convert("RGB")

            cropped_pil = self._resize_if_needed(cropped_pil)

            start = time.time()
            blk.text = self._get_ocr(cropped_pil)

            elapsed = time.time() - start
            if elapsed > 10:
                print(f"⛔ BLOCK TOO SLOW: {elapsed:.2f}s")

        return blk_list

    # -------------------------
    # OCR CORE (THREAD + TIMEOUT)
    # -------------------------
    def _get_ocr(self, image: Image.Image) -> str:

        result = [""]
        error = [None]

        def worker():
            try:
                # --- PIL → base64 ---
                buffer = cv2.imencode(".jpg", np.array(image))[1]
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
        
        print(f"[OCR] {repr(text[:100])}")
        return text

    def _normalize_text(self, text: str) -> str:
        """
        - заменяет переносы строк на пробелы
        - убирает лишние пробелы
        """

        if not text:
            return ""

        # заменяем переносы строк и табы
        text = text.replace("\n", " ").replace("\t", " ")

        # убираем множественные пробелы
        text = " ".join(text.split())

        return text.strip()
