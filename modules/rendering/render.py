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



# ============================================================
# PYSIDE WORD WRAP (исправленный)
# ============================================================

def pyside_word_wrap(
    text: str,
    font_input: str,
    roi_width: int,
    roi_height: int,
    line_spacing,
    outline_width,
    bold=False,
    italic=False,
    underline=False,
    alignment=Qt.AlignLeft,
    direction=Qt.LeftToRight,
    init_font_size: int = 40,
    min_font_size: int = 10
) -> tuple[str, int]:
    """
    Обертка текста в блоке с авто-переносом и запретом висячих букв.
    Возвращает (wrapped_text, font_size)
    """
    from PySide6.QtGui import QFont, QFontMetrics, QTextDocument, QTextCursor, QTextBlockFormat, QTextOption
    from PySide6.QtWidgets import QApplication

    text = text.strip()
    if not text:
        return "", min_font_size

    adjusted_width = roi_width * 1.1  # можно увеличивать, если нужно больше места

    # ----------------------------
    # Подготовка шрифта
    # ----------------------------
    def prepare_font(size: int) -> QFont:
        f = QFont(font_input.strip() or QApplication.font().family(), size)
        f.setBold(bold)
        f.setItalic(italic)
        f.setUnderline(underline)
        return f

    # ----------------------------
    # Оценка размера текста
    # ----------------------------
    def eval_metrics(txt: str, font_sz: int) -> tuple[float, float]:
        doc = QTextDocument()
        doc.setDefaultFont(prepare_font(font_sz))
        doc.setPlainText(txt)

        opt = QTextOption()
        opt.setTextDirection(direction)
        opt.setAlignment(alignment)
        doc.setDefaultTextOption(opt)

        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.Document)
        fmt = QTextBlockFormat()
        fmt.setLineHeight(float(line_spacing) * 100.0, 4)
        cursor.mergeBlockFormat(fmt)

        size = doc.size()
        w, h = size.width(), size.height()
        if outline_width > 0:
            w += 2 * outline_width
            h += 2 * outline_width
        return w, h

    # ----------------------------
    # Основной перенос слов с проверкой висячих букв
    # ----------------------------
    def wrap_text(src: str, font) -> str:
        words = src.split()
        if not words:
            return ""

        metrics = QFontMetrics(font)
        lines: list[str] = []
        current_line = ""

        for word in words:
            space = " " if current_line else ""
            test_line = current_line + space + word

            # Слово помещается в текущую строку
            if metrics.horizontalAdvance(test_line) <= adjusted_width:
                current_line = test_line
                continue

            # ----------------------------
            # Перенос слова
            # ----------------------------
            if current_line:
                lines.append(current_line)
                current_line = ""

            # Если слово помещается в новую строку
            if metrics.horizontalAdvance(word) <= adjusted_width:
                current_line = word
                continue

            # ----------------------------
            # Слишком длинное слово — перенос по слогам (pyphen)
            # ----------------------------
            remaining = word
            parts: list[str] = []

            max_char_width = max([metrics.horizontalAdvance(c) for c in remaining])
            #print(max_char_width)

            while remaining:
                # Ищем максимальный кусок, который помещается
                for i in range(len(remaining), 0, -1):
                    candidate = remaining[:i]
                    # Проверка: не оставляем "висячую букву"
                    if i < len(remaining):
                        # Если оставшийся кусок < ширины одной буквы, уменьшаем candidate
                        remaining_width = metrics.horizontalAdvance(remaining[i:])
                        if remaining_width < 2*max_char_width:
                            continue
                    test_candidate = candidate + ("-" if i < len(remaining) else "")
                    if metrics.horizontalAdvance(test_candidate) <= adjusted_width:
                        break
                else:
                    # если ничего не влезло, берем хотя бы 1 символ
                    i = 1
                    test_candidate = remaining[:1] + ("-" if len(remaining) > 1 else "")

                parts.append(test_candidate)
                remaining = remaining[i:]

            lines.extend(parts)

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    # ----------------------------
    # Начальный wrap
    # ----------------------------
    font_for_measure = prepare_font(init_font_size)
    wrapped_text = wrap_text(text, font_for_measure)

    # ----------------------------
    # Подгонка размера шрифта
    # ----------------------------
    best_size = min_font_size
    lo, hi = min_font_size, init_font_size
    while lo <= hi:
        mid = (lo + hi) // 2
        w, h = eval_metrics(wrapped_text, mid)
        if w <= adjusted_width and h <= roi_height:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_size != init_font_size:
        font_for_measure = prepare_font(best_size)
        wrapped_text = wrap_text(text, font_for_measure)

    return wrapped_text, best_size



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
        center_x = x1 + 1*box_w // 2
        center_y = y1 + 1.2*box_h // 2

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
            min_font_size
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

