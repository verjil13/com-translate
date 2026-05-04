import torch
import numpy as np
import gc

from .base import DetectionEngine


class YOLOPTDetection(DetectionEngine):

    def __init__(self, settings):
        super().__init__(settings)
        self.model = None
        self.device = "cpu"

    def initialize(self, device="cpu"):
        self.device = device

        from ultralytics import YOLO

        model_path = r"models\detection\12x.pt"
        self.model = YOLO(model_path)

        # совместимость
        try:
            self.model.to(device)
        except Exception:
            pass

    def _clear_cache(self):
        """
        Очистка GPU/CPU кэша после инференса страницы
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def detect(self, image: np.ndarray):
        # inference
        results = self.model(image)[0]

        text_boxes = []
        bubble_boxes = []

        # защита от пустого результата
        if results.boxes is None or len(results.boxes) == 0:
            self._clear_cache()
            return self.create_text_blocks(
                image,
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])

            if cls == 0:  # text
                text_boxes.append([x1, y1, x2, y2])

            elif cls == 1:  # bubble
                bubble_boxes.append([x1, y1, x2, y2])

        # 🔥 НИКОГДА None → только numpy
        text_boxes = (
            np.array(text_boxes, dtype=np.float32)
            if text_boxes
            else np.empty((0, 4), dtype=np.float32)
        )

        bubble_boxes = (
            np.array(bubble_boxes, dtype=np.float32)
            if bubble_boxes
            else np.empty((0, 4), dtype=np.float32)
        )

        # 🔥 очистка кэша после обработки страницы
        self._clear_cache()

        return self.create_text_blocks(
            image,
            text_boxes,
            bubble_boxes,
        )
