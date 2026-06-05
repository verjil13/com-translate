import numpy as np

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
from modules.utils.language_utils import get_language_code, is_no_space_lang

from dataclasses import dataclass


@dataclass
class TextRenderingSettings:
    alignment_id: int
    font_family: str
    min_font_size: float
    max_font_size: float
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

def _split_at_fitting_hyphen(
    current_line: str,
    word: str,
    measure_side,
    max_side: float,
) -> Tuple[str, str] | None:
    """Return the longest hyphen-preserving split that fits, if any."""

    best_split = None
    for idx, char in enumerate(word):
        if char != "-" or idx <= 0 or idx >= len(word) - 1:
            continue
        prefix = word[: idx + 1]
        candidate = prefix if not current_line else f"{current_line} {prefix}"
        if measure_side(candidate) <= max_side:
            best_split = (prefix, word[idx + 1 :])
    return best_split

def _wrap_text_greedily(text: str, measure_side, max_side: float) -> str:
    """Greedy wrapping that only splits inside words at existing hyphens."""

    words = text.split()
    lines: List[str] = []

    while words:
        line = ""
        while words:
            next_word = words[0]
            candidate = next_word if not line else f"{line} {next_word}"
            if measure_side(candidate) <= max_side:
                line = candidate
                words.pop(0)
                continue

            hyphen_split = _split_at_fitting_hyphen(line, next_word, measure_side, max_side)
            if hyphen_split is not None:
                prefix, suffix = hyphen_split
                line = prefix if not line else f"{line} {prefix}"
                words[0] = suffix
                break

            if line:
                break

            line = words.pop(0)
            break

        lines.append(line)

    return "\n".join(lines)

def _wrap_no_space_text_greedily(text: str, measure_side, max_side: float) -> str:
    """Greedy wrapping for languages that do not rely on spaces between words."""

    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    wrapped_paragraphs: List[str] = []

    for paragraph in paragraphs:
        chars = [char for char in paragraph if char != " "]
        if not chars:
            wrapped_paragraphs.append("")
            continue

        lines: List[str] = []
        line = ""

        for char in chars:
            candidate = f"{line}{char}"
            if not line or measure_side(candidate) <= max_side:
                line = candidate
                continue

            lines.append(line)
            line = char

        if line:
            lines.append(line)

        wrapped_paragraphs.append("\n".join(lines))

    return "\n".join(wrapped_paragraphs)

def pil_word_wrap(image: Image, tbbox_top_left: Tuple, font_pth: str, text: str, 
                  roi_width, roi_height, align: str, spacing, init_font_size: float, min_font_size: float = 10):
    """Break long text to multiple lines, and reduce point size
    until all text fits within a bounding box."""
    mutable_message = text
    font_size = init_font_size
    font = ImageFont.truetype(font_pth, font_size)
    ###
    def eval_metrics(txt, font):
        """Quick helper function to calculate width/height of text."""
        (left, top, right, bottom) = ImageDraw.Draw(image).multiline_textbbox(xy=tbbox_top_left, text=txt, font=font, align=align, spacing=spacing)
        return (right-left, bottom-top)

    while font_size > min_font_size:
        font = font.font_variant(size=font_size)
        width, height = eval_metrics(mutable_message, font)
        if height > roi_height:
            font_size -= 0.75  # Reduce pointsize
            mutable_message = text  # Restore original text
        elif width > roi_width:
            columns = len(mutable_message)
            while columns > 0:
                columns -= 1
                if columns == 0:
                    break
                mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True)) 
                wrapped_width, _ = eval_metrics(mutable_message, font)
                if wrapped_width <= roi_width:
                    break
            if columns < 1:
                font_size -= 0.75  # Reduce pointsize
                mutable_message = text  # Restore original text
        else:
            break

    if font_size <= min_font_size:
        font_size = min_font_size
        mutable_message = text
        font = font.font_variant(size=font_size)

        # Wrap text to fit within as much as possible
        # Minimize cost function: (width - roi_width)^2 + (height - roi_height)^2
        # This is a brute force approach, but it works well enough
        min_cost = 1e9
        min_text = text
        for columns in range(1, len(text)):
            wrapped_text = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True))
            wrapped_width, wrapped_height = eval_metrics(wrapped_text, font)
            cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
            if cost < min_cost:
                min_cost = cost
                min_text = wrapped_text

        mutable_message = min_text
    
    return mutable_message, font_size

def get_best_render_area(
    blk_list: List[TextBlock],
    img,
    inpainted_img=None
):
    #if inpainted_img is None or inpainted_img.size == 0:
    #    return blk_list

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

        # Берём текущий bbox (его размер уже подогнан ранее)
        cur_x1, cur_y1, cur_x2, cur_y2 = blk.xyxy
        cur_w = cur_x2 - cur_x1
        cur_h = cur_y2 - cur_y1

        # Центр пузыря
        center_x = x1 + ((1.0 * box_w) // 2)
        center_y = y1 + ((0.9 * box_h) // 2)

        # Новый bbox — по центру
        new_x1 = int(center_x - cur_w // 2)
        new_y1 = int(center_y - cur_h // 2)
        new_x2 = new_x1 + cur_w
        new_y2 = new_y1 + cur_h

        blk.xyxy[:] = [new_x1, new_y1, new_x2, new_y2]


    if blk_list and blk_list[0].source_lang not in ['ko', 'zh', 'ja']:
        adjust_blks_size(blk_list, img, -5, -5)


    return blk_list

def pyside_word_wrap(
    blk_list: List["TextBlock"],
    text: str,
    font_input: str,
    roi_width: int,
    roi_height: int,
    line_spacing=1.0,
    outline_width=0.0,
    bold=False,
    italic=False,
    underline=False,
    alignment=Qt.AlignLeft,
    direction=Qt.LeftToRight,
    max_font_size: float = 40,
    min_font_size: float = 10,
    vertical: bool = False,
    no_space_language: bool = False,
    width_coef: float = 1.3,
    height_coef: float = 1.2,
) -> tuple[str, int]:

    from PySide6.QtGui import QFont, QFontMetrics

    if not text:
        return "", min_font_size

    text = str(text).strip()
    if not text:
        return "", min_font_size

    # --- адаптация под язык ---
    if blk_list and getattr(blk_list[0], "source_lang", None) in ['ko', 'zh', 'ja']:
        adjusted_width = roi_width * width_coef
        adjusted_height = roi_height * height_coef
    else:
        adjusted_width = roi_width
        adjusted_height = roi_height

    # --- font ---
    def prepare_font(size: int) -> QFont:
        family = font_input.strip() if isinstance(font_input, str) and font_input.strip() else QApplication.font().family()
        font = QFont(family, size)
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)
        return font

    def get_height(metrics: QFontMetrics, wrapped: str) -> int:
        lines = wrapped.split("\n")
        return int(metrics.height() * len(lines) * line_spacing)

    # --- split слова (ТОЛЬКО если нужно) ---
    def split_single_word(word: str, metrics: QFontMetrics) -> List[str]:
        if len(word) <= 6:
            return [word]
        
        best_i = 0
        best_diff = float("inf")
        

        for i in range(1, len(word)):
            left = word[:i]
            right = word[i:]

            diff = abs(metrics.horizontalAdvance(left) - metrics.horizontalAdvance(right))
            if diff < best_diff:
                best_diff = diff
                best_i = i

        if best_i == 0:
            for i in range(len(word), 0, -1):
                if metrics.horizontalAdvance(word[:i]) <= adjusted_width:
                    best_i = i
                    break

        return [word[:best_i] + "-", word[best_i:]]

    # --- wrap ---
    def wrap_text(src: str, font: QFont, allow_split: bool) -> str:
        metrics = QFontMetrics(font)

        words_raw = src.split()
        words = []

        for w in words_raw:
            if allow_split and metrics.horizontalAdvance(w) > adjusted_width:
                words.extend(split_single_word(w, metrics))
            else:
                words.append(w)

        if not words:
            return ""

        word_widths = [metrics.horizontalAdvance(w) for w in words]
        max_word_width = max(word_widths)        

        target_width = max(max_word_width, adjusted_width)

        lines = []
        current = ""

        flag = 0

        for word in words:
            if not current:
                current = word
                continue
            
            test = current + " " + word

            if flag <3:
                condition = target_width
            else:
                condition = roi_width

            if metrics.horizontalAdvance(test) <= condition:#target_width:
                current = test                
                flag += 1 
            else:
                lines.append(current)
                current = word
                flag = 0

        if current:
            lines.append(current)

        return "\n".join(lines)

    # =========================
    # 1. Подбор шрифта по ширине (БЕЗ split)
    # =========================
    best_size = min_font_size
    allow_split = False

    step = 0.1
    size = max_font_size

    # for size in range(max_font_size, min_font_size - 1, -1):
    while size>=min_font_size:
        font = prepare_font(size)
        metrics = QFontMetrics(font)

        words = text.split()
        max_word_width = max(metrics.horizontalAdvance(w) for w in words)

        if max_word_width <= adjusted_width:
            best_size = size
            wrapped = text
            break
        size-=step    

    else:
        # даже минимальный не влез → разрешаем split
        best_size = min_font_size
        allow_split = True
        font = prepare_font(best_size)
        wrapped = wrap_text(text, font, allow_split)
        allow_split = False
        # for size in range(min_font_size, max_font_size):
        while best_size<=max_font_size:            
            font = prepare_font(best_size)
            metrics = QFontMetrics(font)
            words = wrapped.split()
            max_word_width = max(metrics.horizontalAdvance(w) for w in words)
            if max_word_width >= adjusted_width:
                break
            best_size+=step    

    # =========================
    # 2. Теперь учитываем высоту
    # =========================
    # for size in range(best_size, min_font_size^ - 1, -1):
    while best_size > min_font_size:
        font = prepare_font(best_size)
        metrics = QFontMetrics(font)

        wrapped = wrap_text(wrapped, font, allow_split)
        height = get_height(metrics, wrapped)

        if height <= 1.1*adjusted_height:
            best_size = round(max(min_font_size, min(best_size, max_font_size)),1)
            return wrapped, best_size
        best_size -= step

    # fallback
    min_font_size = round(min_font_size,1)
    font = prepare_font(min_font_size)    
    return wrap_text(wrapped, font, allow_split), min_font_size

# ============================================================
# MANUAL MODE (БЕЗ ИЗМЕНЕНИЙ)
# ============================================================

def manual_wrap(
    main_page, 
    blk_list: List[TextBlock], 
    image_path: str,
    font_family: str, 
    line_spacing: float, 
    outline_width: float, 
    bold: bool, 
    italic: bool, 
    underline: bool, 
    alignment,#: Qt.AlignmentFlag, 
    direction,#: Qt.LayoutDirection, 
    init_font_size: float = 40, 
    min_font_size: float = 10
):
    target_lang = main_page.lang_mapping.get(main_page.t_combo.currentText(), None)
    trg_lng_cd = get_language_code(target_lang)                                                                                   
    for blk in blk_list:
        x1, y1, width, height = blk.xywh
        translation = blk.translation
        if not translation:
            continue
            
        vertical = is_vertical_block(blk, trg_lng_cd)    

        # 1️⃣ Подбираем текст и размер шрифта
        wrapped_text, font_size = pyside_word_wrap(
            blk_list,
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
            vertical,
            is_no_space_lang(trg_lng_cd)
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
        main_page.blk_rendered.emit(wrapped_text, font_size, blk, image_path)
