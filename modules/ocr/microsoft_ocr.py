import numpy as np
import requests
import cv2

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from app.ui.settings.settings_page import SettingsPage


class MicrosoftOCR(OCREngine):
    """OCR engine using LM Studio (PaddleOCR-VL) with strict sequential batching."""

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
        self.expansion_percentage = expansion_percentage

    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    def process_image(
        self, img: np.ndarray, blk_list: list[TextBlock]
    ) -> list[TextBlock]:

        crops = []
        valid_blocks = []

        # -------------------------
        # 1. crop blocks
        # -------------------------
        for blk in blk_list:
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(
                    blk.xyxy,
                    self.expansion_percentage,
                    self.expansion_percentage,
                    img,
                )

            if (
                x1 < x2
                and y1 < y2
                and x1 >= 0
                and y1 >= 0
                and x2 <= img.shape[1]
                and y2 <= img.shape[0]
            ):
                crops.append(img[y1:y2, x1:x2])
                valid_blocks.append(blk)

        if not crops:
            return blk_list

        # -------------------------
        # 2. SEQUENTIAL GREEDY BATCHING
        # -------------------------
        MAX_BATCH_SIZE = 3
        RATIO_LIMIT = 2.0

        batches = []
        i = 0
        n = len(crops)

        while i < n:

            current = [crops[i]]
            current_blocks = [valid_blocks[i]]
            j = i + 1

            while j < n and len(current) < MAX_BATCH_SIZE:

                candidate = current + [crops[j]]

                widths = [c.shape[1] for c in candidate]
                ratio_ok = max(widths) / min(widths) <= RATIO_LIMIT

                if ratio_ok:
                    current.append(crops[j])
                    current_blocks.append(valid_blocks[j])
                    j += 1
                else:
                    break

            batches.append((current, current_blocks))
            i = j

        # -------------------------
        # 3. PROCESS EACH BATCH
        # -------------------------
        MAX_WIDTH = 768
        MIN_WIDTH = 96

        for batch_imgs, batch_blocks in batches:

            # --- dynamic width
            target_width = max(c.shape[1] for c in batch_imgs)
            target_width = min(max(target_width, MIN_WIDTH), MAX_WIDTH)

            # --- resize ВСЕХ заранее
            resized = []
            for crop in batch_imgs:
                if crop.shape[1] != target_width:
                    crop = self._resize_to_width(crop, target_width)
                resized.append(crop)

            # --- средняя высота
            heights = [img.shape[0] for img in resized]
            avg_h = sum(heights) / len(heights)

            sep_h = int(avg_h * 0.25)
            sep_h = max(16, min(sep_h, 64))

            # --- stack
            stacked_images = []

            for k, img_r in enumerate(resized):
                stacked_images.append(img_r)

                if k < len(resized) - 1:
                    sep = self._create_separator(target_width, sep_h, "<E#N#D>")
                    stacked_images.append(sep)

            stacked_img = np.vstack(stacked_images)

            # --- OCR
            encoded_img = self.encode_image(stacked_img)
            raw_text = self._get_gemini_batch_ocr(encoded_img)

            parts = self._parse_ocr_output(raw_text, len(batch_blocks))

            for blk, text in zip(batch_blocks, parts):
                blk.text = text

        return blk_list

    # -------------------------
    # PARSER
    # -------------------------
    def _parse_ocr_output(self, raw_text: str, expected_count: int) -> list[str]:
        text = raw_text.strip()

        result = []

        if "<E#N#D>" in text:
            parts = text.split("<E#N#D>")

            for part in parts:
                part = part.strip()

                if "\n" in part:
                    sub = [s.strip() for s in part.split("\n")]
                    result.extend(sub)
                else:
                    result.append(part)
        else:
            result = [l.strip() for l in text.split("\n")]

        if len(result) < expected_count:
            result += [""] * (expected_count - len(result))
        else:
            result = result[:expected_count]

        return result

    # -------------------------
    # OCR REQUEST
    # -------------------------
    def _get_gemini_batch_ocr(self, base64_image: str) -> str:
        url = "http://localhost:1234/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 123",
        }
        # print(f"data:image/jpeg;base64,{base64_image}")
        payload = {
            "model": "paddleocr-vl-for-manga",
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"LM Studio error: {e}")
            return ""

    # -------------------------
    # UTILS
    # -------------------------
    def _resize_to_width(self, img: np.ndarray, w: int) -> np.ndarray:
        h, ow = img.shape[:2]
        scale = w / ow
        return cv2.resize(img, (w, int(h * scale)))

    # -------------------------
    # SEPARATOR (BLACK OUTLINE + WHITE TEXT)
    # -------------------------
    def _create_separator(self, width: int, height: int, text: str) -> np.ndarray:
        sep = np.zeros((height, width, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = height / 80
        thickness = max(2, height // 20)

        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

        x = (width - tw) // 2
        y = (height + th) // 2

        # --- BLACK OUTLINE (жирная читаемость)
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                cv2.putText(
                    sep,
                    text,
                    (x + dx, y + dy),
                    font,
                    scale,
                    (0, 0, 0),
                    thickness + 2,
                    cv2.LINE_AA,
                )

        # --- WHITE TEXT
        cv2.putText(
            sep,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        return sep
