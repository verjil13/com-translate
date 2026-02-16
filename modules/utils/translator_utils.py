import base64
import json
import re
import jieba
import numpy as np
from pythainlp.tokenize import word_tokenize
from .textblock import TextBlock
import imkit as imk
import unicodedata
from pathlib import Path


MODEL_MAP = {
    "Custom": "",  
    "Deepseek-v3": "deepseek-chat", 
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "Claude-4.5-Sonnet": "claude-sonnet-4-5-20250929",
    "Claude-4.5-Haiku": "claude-haiku-4-5-20251001",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-3.0-Flash": "gemini-3-flash-preview",
    "Gemini-2.5-Pro": "gemini-2.5-pro"
}


# --- кэш словарей и regex ---
_SYMBOL_DICTS: dict[str, dict[str, str]] = {}
_CENSOR_PATTERN = re.compile(r"[●○◯☉〇•]+")

def normalize_censored(text: str) -> str:
    # заменяем любой символ цензуры на стандартный ●
    return _CENSOR_PATTERN.sub("●", text)

# --------------------------
# Загрузка словаря
# --------------------------
def load_symbol_dict(name: str) -> dict[str, str]:
    if name in _SYMBOL_DICTS:
        return _SYMBOL_DICTS[name]

    path = Path(__file__).parent / f"{name}_symbols.json"
    if not path.exists():
        _SYMBOL_DICTS[name] = {}
        return {}

    with open(path, "r", encoding="utf-8") as f:
        _SYMBOL_DICTS[name] = json.load(f)

    return _SYMBOL_DICTS[name]


# --------------------------
# Точная замена для цензуры
# --------------------------
def apply_censored_dict(text: str) -> str:
    symbol_dict = load_symbol_dict("censored")
    if not symbol_dict:
        return text

    text = normalize_censored(text)
    
    # прямой перебор ключей → точное совпадение
    for key, value in symbol_dict.items():
        text = text.replace(key, value)
    return text


def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')


def extract_translations_from_llm(content: str) -> dict[int, str]:
    result = {}
    for match in re.finditer(r'"block_(\d+)"\s*:\s*"([^"]*)"', content):
        idx = int(match.group(1))
        text = match.group(2)
        result[idx] = text
    return result


def apply_translations_to_blocks(blk_list: list[TextBlock], translations: dict[int, str]):
    for idx, blk in enumerate(blk_list):
        if idx in translations:
            blk.translation = translations[idx]
        else:
            print(f"Warning: block_{idx} not found in LLM response.")


def normalize_repeating_chars_advanced(text: str) -> str:

    if not text:
        return text
        
    # --- 0) Удаление мусора в начале строки ---
    text = re.sub(
        r'^[\s!！?？\.．…‥・,，。`~\-—–]+',
        '',
        text
    )    

    # --- 4) Конструкции, которые удаляем полностью ---
    patterns_to_remove = ["$/#", "$/#/$/#/"]
    for pat in patterns_to_remove:
        text = text.replace(pat, "")

    # --- 1) Особые символы, оставляем по 1 ---
    special_one = "~!@#$%^&*"
    if special_one:
        pattern = rf"([{re.escape(special_one)}])\1+"
        text = re.sub(pattern, r"\1", text)

    # --- 2) Особые символы, оставляем по 2 ---
    
    special_two = "あいうえおアイウエオ"
    if special_two:
        pattern = rf"([{re.escape(special_two)}])\1{{2,}}"
        text = re.sub(pattern, lambda m: m.group(1) * 2, text)

    # --- 3) Все остальные символы, оставляем максимум 3 повторов ---
    pattern = r"(.)\1{3,}"
    text = re.sub(pattern, lambda m: m.group(1) * 3, text)   

    # 5) цензура — ПЕРВОЙ
    text = apply_censored_dict(text)    
    
    return text


def get_raw_text(blk_list: list[TextBlock]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        text = blk.text
        text = normalize_repeating_chars_advanced(text)  # исправлено
        rw_txts_dict[block_key] = text

    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    print(raw_texts_json)
    return raw_texts_json


def post_process_translation(text: str) -> str:
    if not text:
        return text

    # 0) Удаляем шум в начале строки
    text = re.sub(
        r'^[\s!！?？\.．…‥・,，。`~\-—–]+',
        '',
        text
    )

    # 1) Ограничение повторов всех символов до 3
    text = re.sub(r"(.)\1{3,}", lambda m: m.group(1) * 3, text)

    # 2) Замена сердечек ♥  на ♡
    text = re.sub(r"[♥](?:️)?", "♡", text)

    return text

def get_raw_translation(blk_list: list[TextBlock]) -> str:
    rw_translations_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        if blk.translation:
            rw_translations_dict[block_key] = post_process_translation(blk.translation)
        else:
            rw_translations_dict[block_key] = ""

    return json.dumps(rw_translations_dict, ensure_ascii=False, indent=4)

def fix_llm_json_structure(s: str) -> str:
    if not s:
        return s

    # --- 1. Убираем лишние внешние кавычки (если LLM вернул строку как текст) ---
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    # --- 2. Нормализуем экранированные переводы строк ---
    # \\r\\n -> \\n
    s = s.replace("\\r\\n", "\\n")

    # Иногда LLM даёт реальные переносы вместо \n — переводим их в \n
    s = s.replace("\r\n", "\n")

    # --- 3. Исправляем "голый" n вместо \n (например: ", n    "block_1") ---
    s = re.sub(
        r'([",}\]])\s*n(\s*"block_\d+"\s*:)',
        r'\1\\n\2',
        s
    )

    # Также если просто n\n или n "block"
    s = re.sub(
        r'\bn(\s*"block_\d+"\s*:)',
        r'\\n\1',
        s
    )

    # --- 4. Гарантируем кавычки у ключей block_N ---
    # block_0: -> "block_0":
    s = re.sub(r'(?<!")\b(block_\d+)\b\s*:', r'"\1":', s)

    # --- 5. Нормализуем реальные переносы строк между блоками в \n ---
    # "text"
    # "block_1": -> "text"\n"block_1":
    s = re.sub(
        r'(")\s*\n\s*(?="block_\d+"\s*:)',
        r'\1\\n',
        s
    )

    # --- 6. Исправляем . \n ; \n : \n вместо ,\n между блоками ---
    # "1". \n "block_1":  ->  "1",\n "block_1":
    s = re.sub(
        r'"\s*[.;:]\s*(\\n|\n)\s*(?="block_\d+"\s*:)',
        r'",\\n',
        s
    )

    # --- 7. Если вообще нет разделителя между полями ---
    # "text"\n"block_1": -> "text",\n"block_1":
    s = re.sub(
        r'(")\s*(\\n|\n)\s*(?="block_\d+"\s*:)',
        r'",\\n',
        s
    )

    # --- 8. Если LLM вставил пробелы без \n ---
    # "text"    "block_2": -> "text",\n    "block_2":
    s = re.sub(
        r'(")\s+(?="block_\d+"\s*:)',
        r'",\\n',
        s
    )

    # --- 9. Чиним случай: "text"\nblock_1: (ключ без кавычек + без запятой) ---
    s = re.sub(
        r'(")\s*(\\n|\n)\s*(block_\d+\s*:)',
        r'",\\n"\3',
        s
    )

    # --- 10. Исправляем случай с лишними запятыми перед новым блоком ---
    # ",\n,"block_1" -> ",\n"block_1"
    s = re.sub(
        r',\s*(\\n)\s*,\s*(?="block_\d+")',
        r',\1',
        s
    )

    # --- 11. Убираем запятую перед последним блоком (перед }) ---
    # ,\n}
    s = re.sub(r',\s*(\\n)\s*}', r'\1}', s)
    s = re.sub(r',\s*}', r'}', s)

    # --- 12. Чиним незакрытые кавычки в значениях (частый LLM баг) ---
    # Если строка оборвалась: "block_1": "текст
    s = re.sub(
        r'("block_\d+"\s*:\s*"[^"\n]*)(\n|\\n)',
        r'\1"\2',
        s
    )

    # --- 13. Финальная проверка парности кавычек ---
    quote_count = s.count('"')
    if quote_count % 2 != 0:
        # Мягкая починка: закрываем последнюю строку значений
        if s.rstrip().endswith("}"):
            s = s.rstrip()[:-1] + '"}'
        else:
            s += '"'

    return s


def set_texts_from_json(blk_list: list[TextBlock], json_string: str):
    if not json_string:
        print("Empty LLM response.")
        return

    try:
        match = re.search(r"\{[\s\S]*\}", json_string)
        if match:
            raw_json = match.group(0)
        else:
            raise json.JSONDecodeError("No JSON object", json_string, 0)

        print(raw_json)
        raw_json = fix_llm_json_structure(raw_json) #проверка структуры
        #print(raw_json)
        translation_dict = json.loads(raw_json)
        #print(translation_dict)
        
        for idx, blk in enumerate(blk_list):
            key = f"block_{idx}"
            if key in translation_dict:
                blk.translation = translation_dict[key]
            else:
                print(f"Warning: {key} not found in JSON.")
        return

    except json.JSONDecodeError:
        pass

    translations = extract_translations_from_llm(json_string)   
    
    if not translations:
        print("❌ Failed to extract any translations from LLM response.")
        return

    apply_translations_to_blocks(blk_list, translations)


def set_upper_case(blk_list: list[TextBlock], upper_case: bool):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = translation.upper()
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation


def get_chinese_tokens(text):
    return list(jieba.cut(text, cut_all=False))


def get_japanese_tokens(text):
    tokenizer = janome.tokenizer.Tokenizer()
    return [token.surface for token in tokenizer.tokenize(text)]


def format_translations(blk_list: list[TextBlock], trg_lng_cd: str, upper_case: bool = True):
    for blk in blk_list:
        translation = blk.translation
        trg_lng_code_lower = trg_lng_cd.lower()
        seg_result = []

        if 'zh' in trg_lng_code_lower:
            seg_result = get_chinese_tokens(translation)
        elif 'ja' in trg_lng_code_lower:
            seg_result = get_japanese_tokens(translation)
        elif 'th' in trg_lng_code_lower:
            seg_result = word_tokenize(translation)

        if seg_result:
            blk.translation = ''.join(word if word in ['.', ','] else f' {word}' for word in seg_result).lstrip()
        else:
            if translation is None:
                continue
            if upper_case and not translation.isupper():
                blk.translation = translation.upper()
            elif not upper_case and translation.isupper():
                blk.translation = translation.lower().capitalize()
            else:
                blk.translation = translation


def is_there_text(blk_list: list[TextBlock]) -> bool:
    return any(blk.text for blk in blk_list)
