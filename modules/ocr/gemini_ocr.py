import numpy as np
from llama_cpp import Llama

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from app.ui.settings.settings_page import SettingsPage


class GeminiOCR(OCREngine):
    """OCR engine using local GGUF model via llama.cpp (no LM Studio)."""

    def __init__(self):
        self.expansion_percentage = 5
        self.model = None

    def initialize(
        self,
        settings: SettingsPage,
        model_path: str = "paddleocr-vl-for-manga.gguf",
        expansion_percentage: int = 5,
    ) -> None:
        self.expansion_percentage = expansion_percentage

        # 🔥 загружаем модель напрямую
        self.model = Llama(
            model_path=r"H:\LModel\adambarbato\PaddleOCR-VL-For-Manga-GGUF\PaddleOCR-VL-For-Manga-BF16.gguf",
            mmproj=r"H:\LModel\adambarbato\PaddleOCR-VL-For-Manga-GGUF\PaddleOCR-VL-For-Manga-mmproj-BF16.gguf",
            # model_path=r"H:\LModel\noctrex\PaddleOCR-VL-1.5-GGUF\PaddleOCR-VL-1.5-F16.gguf",
            # mmproj=r"H:\LModel\noctrex\PaddleOCR-VL-1.5-GGUF\mmproj-F32.gguf",
            n_ctx=4096,
            n_gpu_layers=50,  # 0 если CPU
            verbose=False,
        )

    def process_image(
        self, img: np.ndarray, blk_list: list[TextBlock]
    ) -> list[TextBlock]:
        return self._process_by_blocks(img, blk_list)

    def _process_by_blocks(
        self, img: np.ndarray, blk_list: list[TextBlock]
    ) -> list[TextBlock]:

        for blk in blk_list:
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(
                    blk.xyxy, self.expansion_percentage, self.expansion_percentage, img
                )

            if (
                x1 < x2
                and y1 < y2
                and x1 >= 0
                and y1 >= 0
                and x2 <= img.shape[1]
                and y2 <= img.shape[0]
            ):
                cropped_img = img[y1:y2, x1:x2]
                encoded_img = self.encode_image(cropped_img)

                blk.text = self._get_gemini_block_ocr(encoded_img)

        return blk_list

    def _get_gemini_block_ocr(self, base64_image: str) -> str:
        try:
            print("\n" + "=" * 80)
            print("[OCR REQUEST]")

            # показываем, что реально отправляется
            request_payload = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": ""},
                ],
            }

            print("messages =", request_payload)

            response = self.model.create_chat_completion(
                messages=[request_payload],
                temperature=0,
            )

            print("\n[RAW RESPONSE DICT]")
            print(response)

            text = response["choices"][0]["message"]["content"]

            print("\n[PARSED TEXT]")
            print(text)

            print("=" * 80 + "\n")

            return text.strip()

        except Exception as e:
            print("[OCR ERROR]", e)
            return ""
