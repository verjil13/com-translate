import numpy as np
import pyphen

from typing import Tuple, List

from PIL import Image, ImageFont, ImageDraw
from PySide6.QtGui import (
    QFont,
    QTextDocument,
    QTextCursor,
    QTextBlockFormat,
    QTextOption,
    QFontMetrics
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from modules.utils.textblock import TextBlock
from modules.utils.textblock import adjust_blks_size
from modules.detection.utils.geometry import shrink_bbox
from app.ui.canvas.text.vertical_layout import VerticalTextDocumentLayout
from modules.utils.language_utils import get_language_code

from dataclasses import dataclass


# ============================================================
# pyphen (русский перенос)
# ============================================================

_ru_dic = pyphen.Pyphen(lang="ru")


def hyphenate_ru(word: str) -> Tuple[str, str]:
    """
    Типографский перенос русского слова.
    Возвращает (левая_часть, правая_часть)
    """
    parts = _ru_dic.inserted(word).split("-")
    if len(parts) < 2:
        return word, ""
    return "-".join(parts[:-1]), parts[-1]


# ============================================================
# DATA
# ============================================================

@dataclass
class TextRenderingSettings:
    alignment_id: int
    font_family: str
    min_font_size: int
    max_font_size: int
    color: str
    upper_case: bool
    outline: bool
    outline_color: str
    outline_width: str
    bold: bool
    italic: bool
    underline: bool
    line_spacing: str
    direction: Qt.LayoutDirection


# ============================================================
# PIL HELPERS
# ============================================================

def array_to_pil(rgb_image: np.ndarray):
    return Image.fromarray(rgb_image)


def pil_to_array(pil_image: Image):
    return np.array(pil_image)


def is_vertical_language_code(lang_code: str | None) -> bool:
    """Return True if the language code should use vertical layout.

    Currently treats Japanese and simplified/traditional Chinese as
    vertical-capable languages.
    """
    if not lang_code:
        return False
    code = lang_code.lower()
    return code in {"zh-cn", "zh-tw", "ja"}

def is_vertical_block(blk, lang_code: str | None) -> bool:
    """Return True if this block should be rendered vertically.

    A block is considered vertical when its direction flag is "vertical"
    and the target language code is one of the vertical-capable ones.
    """
    return getattr(blk, "direction", "") == "vertical" and is_vertical_language_code(lang_code)

def pil_word_wrap(image: Image, tbbox_top_left: Tuple, font_pth: str, text: str, 
                  roi_width, roi_height, align: str, spacing, init_font_size: int, min_font_size: int = 10):
    """Break long text to multiple lines, and reduce point size
    until all text fits within a bounding box."""
    mutable_message = text
    font_size = init_font_size
    font = ImageFont.truetype(font_pth, font_size)



# ============================================================
# PYSIDE WORD WRAP (исправленный)
# ============================================================

def pyside_word_wrap(
    text: str,
    font_input: str,
    roi_width: int,
    roi_height: int,  # оставлен для совместимости, но не используется
    line_spacing=1.0,
    outline_width=0,
    bold=False,
    italic=False,
    underline=False,
    alignment=Qt.AlignLeft,
    direction=Qt.LeftToRight,
    max_font_size: int = 40,
    min_font_size: int = 10,
    vertical: bool = False
) -> tuple[str, int]:
    """
    Авто-перенос текста с подбором шрифта по ширине блока.

    Особенности:
    - Параметр roi_height оставлен для совместимости, игнорируется.
    - adjusted_width = roi_width * 1.25 используется для всех измерений.
    - Перенос по пробелам не уменьшает шрифт.
    - Перенос части слова учитывается: если нужен перенос части слова, шрифт уменьшается.
    - Подбор размера идет сверху вниз (от max_font_size до min_font_size).
    """

    from PySide6.QtGui import QFont, QFontMetrics

    if not text:
        return "", min_font_size

    text = str(text).strip()
    if not text:
        return "", min_font_size

    # --- немного увеличиваем ширину для удобства ---
    adjusted_width = roi_width * 1.25

    # --- подготовка шрифта ---
    def prepare_font(size: int) -> QFont:
        f = QFont(font_input.strip() or "Arial", size)
        f.setBold(bold)
        f.setItalic(italic)
        f.setUnderline(underline)
        return f

    # --- функция wrap текста ---
    def wrap_text(src: str, font: QFont) -> tuple[str, bool]:
        """
        Wrap текста по ширине блока (adjusted_width).
        Возвращает:
            wrapped_text: str
            has_hyphen: bool — True, если пришлось переносить часть слова
        """
        metrics = QFontMetrics(font)
        lines: list[str] = []
        current_line = ""
        has_hyphen = False

        for word in src.split():
            space = " " if current_line else ""
            test_line = current_line + space + word

            # Слово помещается в текущую строку — просто добавляем
            if metrics.horizontalAdvance(test_line) <= adjusted_width:
                current_line = test_line
                continue

            # Перенос на новую строку по пробелу
            if current_line:
                lines.append(current_line)
                current_line = ""

            # Слово помещается само на новой строке
            if metrics.horizontalAdvance(word) <= adjusted_width:
                current_line = word
                continue

            # --- слово слишком длинное — перенос по части слова ---
            has_hyphen = True
            remaining = word
            parts: list[str] = []
            max_char_width = max([metrics.horizontalAdvance(c) for c in remaining])

            while remaining:
                for i in range(len(remaining), 0, -1):
                    candidate = remaining[:i]

                    # Проверка на висячую букву
                    if i < len(remaining):
                        remaining_width = metrics.horizontalAdvance(remaining[i:])
                        if remaining_width < 2 * max_char_width:
                            continue

                    test_candidate = candidate + ("-" if i < len(remaining) else "")
                    if metrics.horizontalAdvance(test_candidate) <= adjusted_width:
                        break
                else:
                    i = 1
                    test_candidate = remaining[:1] + ("-" if len(remaining) > 1 else "")

                parts.append(test_candidate)
                remaining = remaining[i:]

            lines.extend(parts)

        if current_line:
            lines.append(current_line)

        return "\n".join(lines), has_hyphen

    # --- подбор размера сверху вниз ---
    for size in range(max_font_size, min_font_size - 1, -1):
        font_for_measure = prepare_font(size)
        wrapped_text, has_hyphen = wrap_text(text, font_for_measure)

        # Если текст помещается без переноса части слова — используем этот размер
        if not has_hyphen:
            return wrapped_text, size

    # --- если ни один больше не подошел — минимальный размер ---
    font_for_measure = prepare_font(min_font_size)
    wrapped_text, _ = wrap_text(text, font_for_measure)
    return wrapped_text, min_font_size



# ============================================================
# GET BEST RENDER AREA
# ============================================================

def get_best_render_area(
    blk_list: List[TextBlock],
    img,
    inpainted_img
):
    """
    Автоматический режим:
    - определяет область для рендера
    - ЦЕНТРИРУЕТ текст по вертикали и горизонтали внутри пузыря
    """


    if inpainted_img is None or inpainted_img.size == 0:
        return blk_list

    for blk in blk_list:
        if blk.text_class != "text_bubble" or blk.bubble_xyxy is None:
            continue

        translation = blk.translation or ""
        if not translation.strip():
            continue

        has_spaces = " " in translation.strip()
        is_vertical_text = not has_spaces

        # Базовая область
        text_draw_bounds = shrink_bbox(
            blk.bubble_xyxy,
            0.3 if is_vertical_text else 0.05
        )

        x1, y1, x2, y2 = text_draw_bounds
        box_w = x2 - x1
        box_h = y2 - y1

        # --------------------------------------------------
        # ❌ СТАРЫЙ РУЧНОЙ СДВИГ (ОСТАВЛЕН, КАК ПРОСИЛ)
        # --------------------------------------------------
        # vertical_offset = int(box_h * 0.08)
        # blk.xyxy[:] = [x1, y1 + vertical_offset, x2, y2]
        # continue

        # --------------------------------------------------
        # ✅ НОВОЕ: АВТОЦЕНТРИРОВАНИЕ
        # --------------------------------------------------

        # Берём текущий bbox (его размер уже подогнан ранее)
        cur_x1, cur_y1, cur_x2, cur_y2 = blk.xyxy
        cur_w = cur_x2 - cur_x1
        cur_h = cur_y2 - cur_y1

        # Центр пузыря
        center_x = x1 + ((0.9 * box_w) // 2)
        center_y = y1 + ((1.2 * box_h) // 2)

        # Новый bbox — по центру
        new_x1 = int(center_x - cur_w // 2)
        new_y1 = int(center_y - cur_h // 2)
        new_x2 = new_x1 + cur_w
        new_y2 = new_y1 + cur_h

        blk.xyxy[:] = [new_x1, new_y1, new_x2, new_y2]

    adjust_blks_size(blk_list, img, -5, -5)
    return blk_list


# ============================================================
# MANUAL MODE (БЕЗ ИЗМЕНЕНИЙ)
# ============================================================

def manual_wrap(
    main_page,
    blk_list: List[TextBlock],
    font_family: str,
    line_spacing,
    outline_width,
    bold,
    italic,
    underline,
    alignment,
    direction,
    init_font_size: int = 40,
    min_font_size: int = 10
):
    for blk in blk_list:
        x1, y1, width, height = blk.xywh
        translation = blk.translation
        if not translation:
            continue
            
        vertical = is_vertical_block(blk, trg_lng_cd)    

        # 1️⃣ Подбираем текст и размер шрифта
        wrapped_text, font_size = pyside_word_wrap(
            translation,
            font_family,
            width,
            height,
            line_spacing,
            outline_width,
            bold,
            italic,
            underline,
            alignment,
            direction,
            init_font_size,
            min_font_size,
            vertical
        )

        # 2️⃣ Центрируем bbox блока под размер текста
        # вычисляем ширину и высоту текста в пикселях
        font = QFont(font_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)
        metrics = QFontMetrics(font)
        text_lines = wrapped_text.split("\n")
        text_w = max(metrics.horizontalAdvance(line) for line in text_lines)
        text_h = metrics.height() * len(text_lines)  # высота всего текста
        # центрирование внутри исходного блока
        new_x1 = x1 + (width - text_w) // 2
        new_y1 = y1 + (height - text_h) // 2
        new_x2 = new_x1 + text_w
        new_y2 = new_y1 + text_h
        blk.xyxy[:] = [new_x1, new_y1, new_x2, new_y2]

        # 3️⃣ Рендерим текст уже в центрированном блоке
        main_page.blk_rendered.emit(wrapped_text, font_size, blk)


