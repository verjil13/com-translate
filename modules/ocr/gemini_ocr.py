from PIL import Image
import torch
import numpy as np

from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
import threading
import time

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
        torch.cuda.empty_cache()
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

            start = time.time()

            text = self._get_ocr(cropped_pil)

            elapsed = time.time() - start
            if elapsed > 10:
                print(f"⛔ BLOCK TOO SLOW: {elapsed:.2f}s")

            blk.text = text

        return blk_list

    # ---- OCR CORE ----


    def _get_ocr(self, image: Image.Image) -> str:
        result = [""]
        error = [None]

        def worker():
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
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        use_cache=False,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                    )

                decoded = self.processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
                )

                result[0] = decoded.strip()

            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)

        if thread.is_alive():
            print("⛔ OCR TIMEOUT (10s) — skipping block")
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

        print(f"[OCR] {repr(text[:100])}")
        return text
