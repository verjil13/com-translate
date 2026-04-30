from PIL import Image
import torch
import numpy as np
import sys
import os

from transformers import AutoModelForCausalLM

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates


class GeminiOCR(OCREngine):
    """OCR engine using PaddleOCR-VL-For-Manga"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.expansion_percentage = 5

    def initialize(
        self,
        settings=None,
        model_path: str = None,
        expansion_percentage: int = 5,
    ) -> None:
        self.expansion_percentage = expansion_percentage

        model_dir = r"H:\models--jzhang533--PaddleOCR-VL-For-Manga"

        print(f"Загрузка PaddleOCR-VL-For-Manga из: {model_dir}")

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        try:
            from processing_paddleocr_vl import PaddleOCRVLProcessor

            self.processor = PaddleOCRVLProcessor.from_pretrained(
                model_dir, trust_remote_code=True, local_files_only=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                local_files_only=True,
            )

            self.model.eval()
            print("✅ Модель PaddleOCR-VL-For-Manga успешно загружена!")

        except Exception as e:
            print("❌ Ошибка загрузки модели:", e)
            raise

    # process_image и _process_by_blocks остаются прежними
    def process_image(
        self, img: np.ndarray, blk_list: list[TextBlock]
    ) -> list[TextBlock]:
        return self._process_by_blocks(img, blk_list)

    def _process_by_blocks(
        self, img: np.ndarray, blk_list: list[TextBlock]
    ) -> list[TextBlock]:
        for blk in blk_list:
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = map(int, blk.bubble_xyxy)
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(
                    blk.xyxy, self.expansion_percentage, self.expansion_percentage, img
                )

            if x1 >= x2 or y1 >= y2:
                continue

            cropped = img[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cropped).convert("RGB")

            blk.text = self._get_ocr(cropped_pil)

        return blk_list

    def _get_ocr(self, image: Image.Image) -> str:
        try:
            inputs = self.processor(images=image, text="", return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                )

            raw_output = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            text = raw_output.strip()
            for prefix in ["OCR:", "OCR :", "Text:", "Ответ:", "Ответ :"]:
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
                    break

            print(f"[OCR RAW] → {repr(text[:100])}")
            return text

        except Exception as e:
            print("[PaddleOCR-VL ERROR]", str(e))
            return ""
