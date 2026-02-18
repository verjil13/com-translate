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
from modules.utils.language_utils import get_language_code

from dataclasses import dataclass



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
############################################
def get_best_render_area(
    blk_list: List[TextBlock],
    img,
    inpainted_img=None
):
    """
    Автоматический режим:
    - определяет область для рендера
    - ЦЕНТРИРУЕТ текст по вертикали и горизонтали внутри пузыря
    """


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
        center_y = y1 + ((0.9 * box_h) // 2) #1.2

        # Новый bbox — по центру
        new_x1 = int(center_x - cur_w // 2)
        new_y1 = int(center_y - cur_h // 2)
        new_x2 = new_x1 + cur_w
        new_y2 = new_y1 + cur_h

        blk.xyxy[:] = [new_x1, new_y1, new_x2, new_y2]


    if blk_list and blk_list[0].source_lang not in ['ko', 'zh']:
        adjust_blks_size(blk_list, img, -5, -5)


    return blk_list

# ============================================================
# PYSIDE WORD WRAP (исправленный)
# ============================================================

def pyside_word_wrap(
    text: str,
    font_input: str,
    roi_width: int,
    roi_height: int,
    line_spacing=1.0,
    outline_width=0,
    bold=False,
    italic=False,
    underline=False,
    alignment=Qt.AlignLeft,
    direction=Qt.LeftToRight,
    max_font_size: int = 40,
    min_font_size: int = 10,
    vertical: bool = False,
    width_coef: float = 1.2, # коэффициент по высоте ширине 1.25
    height_coef: float = 1.1,  # коэффициент по высоте 1.05
) -> tuple[str, int]:
    """
    Авто-перенос текста с подбором шрифта по ширине И ВЫСОТЕ блока.

    Новое:
    - Учитывается roi_height
    - adjusted_height = roi_height * height_coef
    - Если текст не помещается по высоте — шрифт уменьшается
    """

    from PySide6.QtGui import QFont, QFontMetrics

    if not text:
        return "", min_font_size

    text = str(text).strip()
    if not text:
        return "", min_font_size

    # --- коэффициенты удобства ---
    adjusted_width = roi_width * width_coef
    adjusted_height = roi_height * height_coef

    # --- подготовка шрифта ---
    def prepare_font(size: int) -> QFont:
        f = QFont(font_input.strip() or "Arial", size)
        f.setBold(bold)
        f.setItalic(italic)
        f.setUnderline(underline)
        return f

    # --- расчёт реальной высоты текста ---
    def get_text_height(metrics: QFontMetrics, wrapped: str) -> int:
        lines = wrapped.split("\n")
        if not lines:
            return 0
        # учитываем межстрочный интервал
        line_h = metrics.height()
        return int(line_h * len(lines) * line_spacing)

    # --- функция wrap текста ---
    # --- функция wrap текста по строгим правилам ---
    def wrap_text(src: str, font: QFont) -> tuple[str, bool]:
        metrics = QFontMetrics(font)
        lines: list[str] = []
        current_line = ""
        has_hyphen = False

        WWW_width = metrics.horizontalAdvance("WWWW")

        def split_half(word: str) -> tuple[str, str]:
            """Делим слово пополам: левая часть максимум на 1 символ длиннее"""
            n = len(word)
            left_len = (n + 1) // 2  # левая >= правой, разница ≤ 1
            return word[:left_len], word[left_len:]

        for word in src.split():
            space = " " if current_line else ""
            test_line = current_line + space + word

            # 1) Если слово помещается в текущую строку — пишем целиком
            if metrics.horizontalAdvance(test_line) <= adjusted_width:
                current_line = test_line
                continue

            # перенос строки перед обработкой слова
            if current_line:
                lines.append(current_line)
                current_line = ""

            word_width = metrics.horizontalAdvance(word)

            # 2) Если слово НЕ влезает, но короче WWW — пишем целиком
            if word_width <= WWW_width:
                current_line = word
                continue

            # 3) Если слово влезает в пустую строку — пишем целиком
            if word_width <= adjusted_width:
                current_line = word
                continue

            # 4) Слово длиннее WWW и не влезает — СТРОГО делим пополам (один раз)
            left_len = (len(word) + 1) // 2  # левая максимум на 1 символ длиннее
            left = word[:left_len]
            right = word[left_len:]

            left_with_hyphen = left + "-"
            left_width = metrics.horizontalAdvance(left_with_hyphen)

            # Если даже половина не влезает — НЕ режем дальше!
            # Пусть внешний цикл уменьшает шрифт (правило №4)
            if left_width > adjusted_width:
                # один единственный перенос без умных разрезов
                has_hyphen = True
                lines.append(left_with_hyphen)
                if right:
                    current_line = right
                continue

            # Нормальный перенос строго 50/50
            has_hyphen = True
            lines.append(left_with_hyphen)
            if right:
                current_line = right

        if current_line:
            lines.append(current_line)

        return "\n".join(lines), has_hyphen

    # --- подбор размера: теперь по ширине И высоте ---
    for size in range(max_font_size, min_font_size - 1, -1):
        font_for_measure = prepare_font(size)
        metrics = QFontMetrics(font_for_measure)

        wrapped_text, has_hyphen = wrap_text(text, font_for_measure)

        # вычисляем итоговую высоту текста
        text_height = get_text_height(metrics, wrapped_text)

        # ❗ НОВОЕ: проверка переполнения по высоте
        if text_height > adjusted_height:
            continue  # уменьшаем шрифт

        # если влез по высоте и не было переноса части слова — идеально
        if not has_hyphen:
            return wrapped_text, size

        # если был перенос части слова, но всё влезает по высоте —
        # всё равно допускаем (лучше чем переполнение)
        if text_height <= adjusted_height:
            return wrapped_text, size

    # --- fallback: минимальный размер ---
    font_for_measure = prepare_font(min_font_size)
    wrapped_text, _ = wrap_text(text, font_for_measure)
    return wrapped_text, min_font_size


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
    init_font_size: int = 40, 
    min_font_size: int = 10
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
        main_page.blk_rendered.emit(wrapped_text, font_size, blk, image_path)



