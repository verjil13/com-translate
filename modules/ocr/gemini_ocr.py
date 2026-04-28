import numpy as np
import requests

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from ..utils.translator_utils import MODEL_MAP
from app.ui.settings.settings_page import SettingsPage


class GeminiOCR(OCREngine):
    """OCR engine using LM Studio (PaddleOCR-VL) with block processing method."""

    def __init__(self):
        self.api_key = None
        self.expansion_percentage = 5
        self.model = ""

    def initialize(
        self,
        settings: SettingsPage,
        model: str = "Gemini-2.0-Flash",
        expansion_percentage: int = 5,
    ) -> None:
        """
        Initialize the OCR engine.
        """
        self.expansion_percentage = expansion_percentage

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
        url = "http://localhost:1234/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 123",
        }

        payload = {
            "model": "paddleocr-vl-for-manga",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    ],
                }
            ],
            "temperature": 0,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()

            data = response.json()
            text = data["choices"][0]["message"]["content"]

            return text.strip()

        except Exception as e:
            print(f"LM Studio error: {e}")
            return ""
