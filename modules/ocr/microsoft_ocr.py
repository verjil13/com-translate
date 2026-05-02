from PIL import Image
import torch
import numpy as np
import threading
import time

from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates


class MicrosoftOCR(OCREngine):
    """OCR engine for PaddleOCR-VL / Manga variants (robust version)"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda"
        self.expansion_percentage = 5

    # ---------------------------
    # INIT
    # ---------------------------
    def initialize(
        self,
        settings=None,
        model_path: str = None,
        expansion_percentage: int = 5,
    ) -> None:

        self.expansion_percentage = expansion_percentage
        model_id = model_path or "PaddlePaddle/PaddleOCR-VL-For-Manga"

        print("DEVICE:", self.device)
        print("Loading model:", model_id)

        try:
            # ---- Processor ----
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            self._patch_processor_image_size()

            # ---- Config ----
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            # 🔥 ROBUST ROPE FIX
            if hasattr(config, "rope_scaling"):

                # 1. если строка
                if isinstance(config.rope_scaling, str):
                    print("🔧 rope_scaling is string → fixing")
                    config.rope_scaling = {
                        "rope_type": "linear",
                        "factor": 1.0,
                    }

                # 2. если dict
                elif isinstance(config.rope_scaling, dict):
                    rope_type = (
                        config.rope_scaling.get("rope_type")
                        or config.rope_scaling.get("type")
                    )

                    if rope_type in ["default", None]:
                        print("🔧 Fixing rope_type -> linear")
                        config.rope_scaling["rope_type"] = "linear"

                    # 💥 КЛЮЧЕВОЕ: добавляем factor если его нет
                    if "factor" not in config.rope_scaling:
                        print("🔧 Adding rope factor")
                        config.rope_scaling["factor"] = 1.0

                # 3. если None / мусор
                else:
                    print("🔧 Adding full rope_scaling")
                    config.rope_scaling = {
                        "rope_type": "linear",
                        "factor": 1.0,
                    }

            # ---- Model ----
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                .to(self.device)
                .eval()
            )

            print("✅ Model loaded successfully")

        except Exception as e:
            print("❌ Model loading error:", e)
            raise

    # ---------------------------
    # PATCH IMAGE PROCESSOR
    # ---------------------------
    def _patch_processor_image_size(self):
        """
        Fix models with:
        {'max_pixels', 'min_pixels'}
        """

        try:
            if hasattr(self.processor, "image_processor"):
                ip = self.processor.image_processor

                if hasattr(ip, "size") and isinstance(ip.size, dict):
                    keys = set(ip.size.keys())

                    if "max_pixels" in keys or "min_pixels" in keys:
                        ip.size = {"shortest_edge": 1024}
                        print("🔧 Patched image_processor.size")

        except Exception as e:
            print("⚠️ Patch warning:", e)

    # ---------------------------
    # PUBLIC API
    # ---------------------------
    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]):
        return self._process_by_blocks(img, blk_list)

    # ---------------------------
    # BLOCK PROCESSING
    # ---------------------------
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
            blk.text = self._get_ocr(cropped_pil)

            elapsed = time.time() - start
            if elapsed > 10:
                print(f"⛔ SLOW BLOCK: {elapsed:.2f}s")

        return blk_list

    # ---------------------------
    # OCR WITH TIMEOUT
    # ---------------------------
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
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        use_cache=True,
                    )

                text = self.processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                ).strip()

                result[0] = text

            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=30)

        if thread.is_alive():
            print("⛔ OCR TIMEOUT")
            return ""

        if error[0]:
            print("⛔ OCR ERROR:", error[0])
            return ""

        return result[0]
