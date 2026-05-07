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
        results = self.model(
            image,
            conf=0.15,  # ниже → ловит больше текста
            iou=0.25,  # аккуратнее с перекрытиями
            # imgsz=1024,    # важно для мелкого текста
            max_det=200,
            augment=True,
        )[0]

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

        # 🔥 ФИЛЬТРАЦИЯ
        text_boxes = self._filter_boxes(
            text_boxes,
            image.shape,
            max_rel_area=0.4,   # режем огромные блоки
            max_inside=2        # убираем "контейнеры"
        )

        bubble_boxes = self._filter_boxes(
            bubble_boxes,
            image.shape,
            max_rel_area=0.9,   # пузыри можно оставлять большими
            max_inside=10
        )

        # 🔥 очистка кэша после обработки страницы
        self._clear_cache()

        return self.create_text_blocks(
            image,
            text_boxes,
            bubble_boxes,
        )


    def _filter_boxes(self, boxes: np.ndarray, image_shape, max_rel_area=0.4, max_inside=2):
        """
        Фильтрация боксов:
        - удаляет слишком большие
        - удаляет боксы, которые содержат много других
        - расширяет слишком высокие и узкие боксы

        boxes: (N, 4)
        """
        if len(boxes) == 0:
            return boxes

        H, W = image_shape[:2]
        img_area = H * W

        def contains(big, small):
            return (
                big[0] <= small[0]
                and big[1] <= small[1]
                and big[2] >= small[2]
                and big[3] >= small[3]
            )

        filtered = []

        for i, b1 in enumerate(boxes):
            x1, y1, x2, y2 = b1

            w = x2 - x1
            h = y2 - y1

            area = w * h
            rel_area = area / img_area

            # ❌ слишком большой бокс
            if rel_area > max_rel_area:
                continue

            # ❌ содержит слишком много других
            inside = 0
            for j, b2 in enumerate(boxes):
                if i == j:
                    continue
                if contains(b1, b2):
                    inside += 1

            if inside > max_inside:
                continue

            # ✅ слишком высокий и узкий —
            # расширяем по ширине в 3 раза
            if h > w * 3:
                x1 -= w
                x2 += w

                # защита от выхода за границы
                x1 = max(0, x1)
                x2 = min(W, x2)

            filtered.append([x1, y1, x2, y2])

        if not filtered:
            return np.empty((0, 4), dtype=np.float32)

        return np.array(filtered, dtype=np.float32)
