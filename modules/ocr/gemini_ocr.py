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
        self.device = "cuda"
        self.expansion_percentage = 5

    # -------------------------
    # INIT
    # -------------------------
    def initialize(
        self,
        settings=None,
        model_path: str = None,
        expansion_percentage: int = 5,
    ) -> None:

        self.expansion_percentage = expansion_percentage

        model_id = "PaddlePaddle/PaddleOCR-VL-1.5"

        print("DEVICE:", self.device)
        print("Loading model:", model_id)

        try:
            self.processor = AutoProcessor.from_pretrained(model_id)

            self.model = (
                AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                )
                .to(self.device)
                .eval()
            )

            print("✅ PaddleOCR-VL loaded successfully")

        except Exception as e:
            print("❌ Model loading error:", e)
            raise

    # -------------------------
    # PUBLIC API
    # -------------------------
    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]):
        return self._process_by_blocks(img, blk_list)

    # -------------------------
    # RESIZE FUNCTION (NEW)
    # -------------------------
    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        max_size = 384       

        if min(w,h)>=48:
            w/=1.5
            h/=1.5

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

            # 🔥 RESIZE FIX HERE
            cropped_pil = self._resize_if_needed(cropped_pil) #384

            start = time.time()
            blk.text = self._get_ocr(cropped_pil)

            elapsed = time.time() - start
            if elapsed > 10:
                print(f"⛔ BLOCK TOO SLOW: {elapsed:.2f}s")

        torch.cuda.empty_cache()

        return blk_list

    # -------------------------
    # OCR CORE (THREAD + TIMEOUT)
    # -------------------------
    def _get_ocr(self, image: Image.Image) -> str:

        result = [""]
        error = [None]

        def worker():
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "OCR:"},
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
                        max_new_tokens=2048,
                        #do_sample=False,
                        #use_cache=False,
                        #eos_token_id=self.processor.tokenizer.eos_token_id,
                        #pad_token_id=self.processor.tokenizer.pad_token_id,
                        repetition_penalty=1.1,
                        temperature=0.0,  # детерминистично
                        top_p=1.0,
                        early_stopping=True,  # может помо
                    )

                decoded = self.processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )

                result[0] = decoded.strip()

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

        print(f"[OCR] {repr(text[:100])}")
        return text
