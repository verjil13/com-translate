from PIL import Image
import torch
import numpy as np

from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates


class GeminiOCR(OCREngine):
    """OCR engine using PaddleOCR-VL (WORKING HF VERSION)"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" #№if torch.cuda.is_available() else "cpu"
        self.expansion_percentage = 5

    def initialize(
        self,
        settings=None,
        model_path: str = None,
        expansion_percentage: int = 5,
    ) -> None:

        self.expansion_percentage = expansion_percentage

        # ❗ ВАЖНО: используем repo_id, НЕ snapshot path
        model_id = "PaddlePaddle/PaddleOCR-VL-1.5"

        print("DEVICE:", self.device)
        print("Loading model:", model_id)

        try:
            # ---- Processor (REMOTE CODE REQUIRED) ----
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                #trust_remote_code=True,
            )

            # ---- Model ----
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16# if self.device == "cuda" else torch.float32,
                #device_map="auto" if self.device == "cuda" else None,
                #trust_remote_code=True,
            ).to(self.device).eval() #правильно

            # self.model.eval() #неправильно

            print("✅ PaddleOCR-VL loaded successfully")

        except Exception as e:
            print("❌ Model loading error:", e)
            raise

    # ---- public API (НЕ ТРОГАЕМ) ----
    def process_image(
        self, img: np.ndarray, blk_list: list[TextBlock]
    ) -> list[TextBlock]:
        return self._process_by_blocks(img, blk_list)

    # ---- block processing ----
    def _process_by_blocks(self, img: np.ndarray, blk_list: list[TextBlock]):

        for blk in blk_list:
            torch.cuda.empty_cache()
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

            blk.text = self._get_ocr(cropped_pil)

        return blk_list

    # ---- OCR CORE ----
    def _get_ocr(self, image: Image.Image) -> str:
        try:
            PROMPT = "OCR:"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                )

            result = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] : -1]
            )

            text = result.strip()

            # cleanup
            for prefix in ["OCR:", "OCR :", "Text:", "Answer:", "Answer :"]:
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
                    break

            print(f"[OCR] {repr(text[:100])}")
            return text

        except Exception as e:
            print("[OCR ERROR]", e)
            return ""
