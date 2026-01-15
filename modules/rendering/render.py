import numpy as np
import pyphen

from typing import Tuple, List

from PIL import Image, ImageFont, ImageDraw
from PySide6.QtGui import (
    QFont,
    QTextDocument,
    QTextCursor,
    QTextBlockFormat,
    QTextOption
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
# PYSIDE WORD WRAP
# ============================================================

def pyside_word_wrap(
    text: str,
    font_input: str,
    roi_width: int,
    roi_height: int,
    line_spacing,
    outline_width,
    bold,
    italic,
    underline,
    alignment,
    direction,
    init_font_size: int,
    min_font_size: int = 10
) -> tuple[str, int]:

    from PySide6.QtGui import (
        QFont, QFontMetrics, QTextDocument,
        QTextCursor, QTextBlockFormat, QTextOption
    )
    from PySide6.QtWidgets import QApplication

    text = text.strip()
    if not text:
        return "", min_font_size

    # ──────────────────────────────
    # Font helpers
    # ──────────────────────────────
    def prepare_font(font_size: int) -> QFont:
        family = font_input.strip() or QApplication.font().family()
        font = QFont(family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)
        return font

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

    # ──────────────────────────────
    # Word wrapping
    # ──────────────────────────────
    def wrap_text(src: str, font, roi_width: int) -> str:
        words = src.split()
        if not words:
            return ""

        metrics = QFontMetrics(font)
        lines: list[str] = []
        current_line = ""

        for word in words:
            space = " " if current_line else ""
            test_line = current_line + space + word

            # Слово влезает в текущую строку
            if metrics.horizontalAdvance(test_line) <= roi_width:
                current_line = test_line
                continue

            # Фиксируем текущую строку
            if current_line:
                lines.append(current_line)
                current_line = ""

            # Слово целиком влезает в новую строку
            if metrics.horizontalAdvance(word) <= roi_width:
                current_line = word
                continue

            # Короткие слова не рвём
            if len(word) < 5:
                current_line = word
                continue

            # ───── перенос длинного слова ─────
            remaining = word
            parts: list[str] = []

            while remaining:
                for i in range(len(remaining), 0, -1):
                    candidate = remaining[:i]
                    with_hyphen = candidate + "-" if i < len(remaining) else candidate

                    if metrics.horizontalAdvance(with_hyphen) <= roi_width:
                        break
                else:
                    i = 1
                    candidate = remaining[:1]
                    with_hyphen = candidate + "-"

                parts.append(with_hyphen)
                remaining = remaining[i:]

            # ───── Устранение висячей буквы ─────
            # Если последняя часть после переноса — одна буква, переносим букву с предыдущей части
            while len(parts) >= 2:
                last = parts[-1].rstrip("-")
                prev = parts[-2]

                if len(last) == 1 and prev.endswith("-"):
                    # Переносим одну букву с предыдущей части на последнюю
                    prev_core = prev[:-1]
                    if len(prev_core) >= 2:
                        # Формируем корректные две части
                        parts[-2] = prev_core[:-1] + "-"
                        parts[-1] = prev_core[-1] + last
                    else:
                        break  # больше нельзя корректно разделить
                else:
                    break

            lines.extend(parts)

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    # ──────────────────────────────
    # Initial wrap
    # ──────────────────────────────
    font_for_measure = prepare_font(init_font_size)
    wrapped_text = wrap_text(text, font_for_measure, roi_width)

    # ──────────────────────────────
    # Font size fitting
    # ──────────────────────────────
    best_size = min_font_size
    lo, hi = min_font_size, init_font_size

    while lo <= hi:
        mid = (lo + hi) // 2
        w, h = eval_metrics(wrapped_text, mid)

        if w <= roi_width and h <= roi_height:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_size != init_font_size:
        final_font = prepare_font(best_size)
        wrapped_text = wrap_text(text, final_font, roi_width)

    return wrapped_text, best_size







# ============================================================
# PIL RENDER
# ============================================================

def pil_word_wrap(
    image: Image,
    tbbox_top_left: Tuple,
    font_pth: str,
    text: str,
    roi_width,
    roi_height,
    align: str,
    spacing,
    init_font_size: int,
    min_font_size: int = 10
):
    mutable_message = text
    font_size = init_font_size
    font = ImageFont.truetype(font_pth, font_size)

    def eval_metrics(txt, font):
        (l, t, r, b) = ImageDraw.Draw(image).multiline_textbbox(
            xy=tbbox_top_left,
            text=txt,
            font=font,
            align=align,
            spacing=spacing
        )
        return r - l, b - t

    while font_size > min_font_size:
        font = font.font_variant(size=int(font_size))
        width, height = eval_metrics(mutable_message, font)

        if height > roi_height or width > roi_width:
            font_size -= 0.75
            mutable_message = text
        else:
            break

    return mutable_message, int(font_size)


# ============================================================
# DRAW TEXT
# ============================================================

def wrap_text(src: str, font, roi_width: int) -> str:
    words = src.split()
    if not words:
        return ""

    lines = []
    current_line = ""

    for word in words:
        test_line = word if not current_line else current_line + " " + word

        # измеряем ширину
        metrics = QFontMetrics(font)
        test_width = metrics.horizontalAdvance(test_line)

        if test_width <= roi_width:
            current_line = test_line
        else:
            # перенос длинного слова
            if current_line:
                lines.append(current_line)
                current_line = ""

            # проверяем, помещается ли слово целиком
            if metrics.horizontalAdvance(word) <= roi_width:
                current_line = word
            else:
                # перенос по слогам
                left, right = hyphenate_ru(word)
                if not right:
                    # не удалось перенести
                    current_line = word
                else:
                    lines.append(f"{left}-")
                    current_line = right

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


# ============================================================
# GET BEST RENDER AREA  (ВАЖНО!)
# ============================================================

def get_best_render_area(
    blk_list: List[TextBlock],
    img,
    inpainted_img
):
    if inpainted_img is None or inpainted_img.size == 0:
        return blk_list

    for blk in blk_list:
        if blk.text_class == "text_bubble" and blk.bubble_xyxy is not None:
            translation = blk.translation or ""
            has_spaces = " " in translation.strip()
            is_vertical_text = not has_spaces

            if is_vertical_text:
                text_draw_bounds = shrink_bbox(blk.bubble_xyxy, 0.3)
            else:
                text_draw_bounds = shrink_bbox(blk.bubble_xyxy, 0.05)

            x1, y1, x2, y2 = text_draw_bounds
            blk.xyxy[:] = [x1, y1, x2, y2]

    adjust_blks_size(blk_list, img, -5, -5)
    return blk_list

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
        if not translation or len(translation) == 0:
            continue
        
        translation, font_size = pyside_word_wrap(
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
            init_font_size=init_font_size,
            min_font_size=min_font_size
        )
        
        main_page.blk_rendered.emit(translation, font_size, blk)