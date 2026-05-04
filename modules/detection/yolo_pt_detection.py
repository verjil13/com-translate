import torch
import numpy as np

from .base import DetectionEngine


class YOLOPTDetection(DetectionEngine):

    def __init__(self, settings):
        super().__init__(settings)
        self.model = None

    def initialize(self, device="cpu"):
        self.device = device

        from ultralytics import YOLO

        self.model = YOLO(r"models\detection\12x.pt")

        # ultralytics сам управляет device лучше через predict,
        # но оставим для совместимости
        self.model.to(device)

    def detect(self, image: np.ndarray):
        results = self.model(image)[0]

        text_boxes = []
        bubble_boxes = []

        # защита от пустого результата
        if results.boxes is None:
            return self.create_text_blocks(image, np.array([]), np.array([]))

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])

            if cls == 0:  # text
                text_boxes.append([x1, y1, x2, y2])

            elif cls == 1:  # bubble
                bubble_boxes.append([x1, y1, x2, y2])

        # 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: НИКАКИХ None
        text_boxes = (
            np.array(text_boxes, dtype=np.float32)
            if len(text_boxes) > 0
            else np.array([])
        )
        bubble_boxes = (
            np.array(bubble_boxes, dtype=np.float32)
            if len(bubble_boxes) > 0
            else np.array([])
        )

        return self.create_text_blocks(image, text_boxes, bubble_boxes)
