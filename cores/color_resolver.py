# -*- coding: utf-8 -*-
import json
import unicodedata
import re
import colorsys
import os
import atexit
import logging
import webcolors
from difflib import get_close_matches
from functools import lru_cache
from llama_cpp import Llama

# ─────────────────────────────────────────────────────────────
# 📦 1. 設定 & データロード
# ─────────────────────────────────────────────────────────────
GGUF_PATH = "./checkpoints/qwen2.5-1.5b-instruct-q4_k_m.gguf"
N_CTX = 2048
N_THREADS = os.cpu_count() or 4
DEBUG_LLM = True

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_data.json")
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r',(\s*[}\]])', r'\1', text)  # trailing-comma tolerance
    RAW = json.loads(text)
except FileNotFoundError:
    logging.error("色辞書ファイルが見つかりません: %s", DATA_PATH)
    raise
except json.JSONDecodeError as e:
    logging.error("JSONのパース失敗: %s", e)
    raise RuntimeError(f"色辞書ファイルのパースに失敗しました: {DATA_PATH}") from e


def _norm(t):
    return unicodedata.normalize("NFKC", re.sub(r'\s+', '', str(t))).lower()


def _get_raw_db(key):
    target = _norm(key)
    for k, v in RAW.items():
        if _norm(k) == target:
            return v
    return {}


COLOR_DB = {_norm(k): tuple(v) for k, v in _get_raw_db("COLOR_DB").items()}
MODIFIER_DB = {_norm(k): tuple(v) for k, v in _get_raw_db("MODIFIER_DB").items()}
DEGREE_SCALE_RAW = {_norm(k): v for k, v in _get_raw_db("DEGREE_SCALE").items()}
METAPHOR_DB = {_norm(k): _norm(v) for k, v in _get_raw_db("METAPHOR_DB").items()}

# 長い順にソートして最長一致を保証
_DEGREE_SORTED = sorted(DEGREE_SCALE_RAW.items(), key=lambda kv: -len(kv[0]))
_MODIFIER_SORTED = sorted(MODIFIER_DB.items(), key=lambda kv: -len(kv[0]))


def _build_unified():
    """METAPHOR_DB を起動時に RGB へ解決し COLOR_DB と統合した dict を返す。
    リンク切れがあれば stderr に警告。"""
    u = dict(COLOR_DB)
    broken = []
    for k, color_name in METAPHOR_DB.items():
        rgb = COLOR_DB.get(color_name)
        if rgb is None:
            broken.append((k, color_name))
            continue
        if k not in u:
            u[k] = rgb
    if broken:
        logging.warning("%s unresolved metaphors (first 5): %s", len(broken), broken[:5])
    return u


UNIFIED_DB = _build_unified()
_UNIFIED_KEYS_SORTED = sorted(UNIFIED_DB.keys(), key=len, reverse=True)

logging.info(
    "DB Load: COLOR=%s, META=%s, UNIFIED=%s, MOD=%s",
    len(COLOR_DB),
    len(METAPHOR_DB),
    len(UNIFIED_DB),
    len(MODIFIER_DB),
)


# ─────────────────────────────────────────────────────────────
# 🛠️ 2. ヘルパー & 抽出ロジック
# ─────────────────────────────────────────────────────────────
def rgb_to_hex(r, g, b): return f"#{r:02X}{g:02X}{b:02X}"
def get_ansi_block(r, g, b): return f"\033[48;2;{r};{g};{b}m  \033[0m"


def _clean_llm(raw):
    cleaned = re.sub(r'(?is)<think>.*?(?:</think>|$)', '', raw)
    cleaned = cleaned.replace("色名:", "").replace("出力:", "")
    return cleaned.strip("「」『』\"' ,.。、:：\n\t ")


def _longest_match_in(text, sorted_keys):
    for k in sorted_keys:
        if k and k in text:
            return k
    return None


def _strip_modifier_phrases(norm_text):
    """色名探索の前に MODIFIER/DEGREE 由来の語句を抜く。
    例: "やや暗いグレイッシュな青" → "  な青" にして、
    色名抽出時に "グレイ"(=MODIFIER "グレイッシュ" の頭) が
    "青" に勝ってしまう問題を防ぐ。"""
    s = norm_text
    for kw, _ in _MODIFIER_SORTED:
        if kw and kw in s:
            s = s.replace(kw, " ")
    for kw, _ in _DEGREE_SORTED:
        if kw and kw in s:
            s = s.replace(kw, " ")
    return s


def _longest_match(text, candidates):
    for k in sorted(candidates.keys(), key=len, reverse=True):
        if k and k in text:
            return k
    return None


def _resolve_color_name(name):
    norm = _norm(name)
    if not norm:
        return (128, 128, 128)
    if norm in COLOR_DB:
        return COLOR_DB[norm]
    if norm in UNIFIED_DB:
        return UNIFIED_DB[norm]
    m = get_close_matches(norm, COLOR_DB.keys(), n=1, cutoff=0.75)
    if m:
        return COLOR_DB[m[0]]
    try:
        return tuple(webcolors.hex_to_rgb(webcolors.name_to_hex(norm)))
    except (ValueError, AttributeError):
        return (128, 128, 128)


def parse_weighted_mix(text):
    norm_text = _norm(text)
    matches = re.findall(
        r'([一-龥a-zA-Zぁ-んァ-ヶー・]+?)\s*(\d+)\s*%',
        norm_text,
    )
    if not matches:
        return None
    ws, tw = [0, 0, 0], 0
    for name, w_str in matches:
        w = int(w_str)
        rgb = _resolve_color_name(name)
        if rgb == (128, 128, 128):
            continue
        for i in range(3):
            ws[i] += rgb[i] * w
        tw += w
    return tuple(int(round(v / tw)) for v in ws) if tw > 0 else None


def parse_multi_mix(norm_text):
    """3 色以上の "AとBとCの中間/を混ぜた/のブレンド" にも対応。"""
    if not any(tail in norm_text for tail in
               ("の中間", "を混ぜた", "のミックス", "のブレンド", "をブレンド")):
        return None
    tail_match = re.search(
        r'([一-龥a-zA-Zぁ-んァ-ヶー・]+?)'
        r'(?:の中間|を混ぜた|のミックス|のブレンド|をブレンド)',
        norm_text,
    )
    if not tail_match:
        return None
    head = norm_text[:tail_match.start(1) + len(tail_match.group(1))]
    raw_parts = re.split(r'(?:と|＆|&|＋|\+|、|,)', head)
    rgbs = []
    for p in raw_parts:
        p = p.strip("のな ")
        if not p:
            continue
        rgb = _resolve_color_name(p)
        if rgb != (128, 128, 128):
            rgbs.append(rgb)
    if len(rgbs) < 2:
        return None
    return tuple(sum(c) // len(rgbs) for c in zip(*rgbs))


def apply_modifiers(rgb, text):
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    norm_text = _norm(text)

    deg = 1.0
    for kw, sc in _DEGREE_SORTED:
        if kw and kw in norm_text:
            deg = sc
            break

    sa, va, hs = 0.0, 0.0, 0.0
    for kw, vals in _MODIFIER_SORTED:
        if kw and kw in norm_text:
            sa += vals[0]
            va += vals[1]
            hs += vals[2] if len(vals) > 2 else 0.0

    sa = max(-0.6, min(0.6, sa))
    va = max(-0.6, min(0.6, va))
    hs = max(-0.2, min(0.2, hs))

    s = max(0.05, min(0.95, s + sa * deg))
    v = max(0.15, min(0.95, v + va * deg))
    h = (h + hs * deg) % 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(round(c * 255)) for c in (r, g, b))


# ─────────────────────────────────────────────────────────────
# 🎨 3. 色解決エンジン
# ─────────────────────────────────────────────────────────────
def resolve_color(text):
    norm = _norm(text)

    mix = parse_weighted_mix(text)
    if mix:
        return apply_modifiers(mix, text)

    multi = parse_multi_mix(norm)
    if multi:
        return apply_modifiers(multi, text)

    # 修飾語を抜いた文字列で色名探索（MODIFIER 頭が COLOR 名と衝突するのを回避）
    stripped = _strip_modifier_phrases(norm)
    kw = _longest_match_in(stripped, _UNIFIED_KEYS_SORTED)
    if kw is None:
        # 修飾語剥ぎで色名が消えた場合は元文でフォールバック
        kw = _longest_match_in(norm, _UNIFIED_KEYS_SORTED)
    if kw:
        return apply_modifiers(UNIFIED_DB[kw], text)

    try:
        rgb = tuple(webcolors.hex_to_rgb(webcolors.name_to_hex(norm)))
        return apply_modifiers(rgb, text)
    except (ValueError, AttributeError):
        pass

    return _llm_fallback(norm)


# ─────────────────────────────────────────────────────────────
# 🤖 4. LLM
# ─────────────────────────────────────────────────────────────
_llm = None
def _get_llm():
    global _llm
    if _llm is None:
        if not os.path.exists(GGUF_PATH):
            raise FileNotFoundError(f"❌ GGUF未発見: {GGUF_PATH}")
        logging.info("モデル読み込み: %s", GGUF_PATH)
        _llm = Llama(model_path=GGUF_PATH, n_ctx=N_CTX, n_gpu_layers=0,
                     n_threads=N_THREADS, n_batch=256, verbose=False, logits_all=False)
        logging.info("LLM準備完了")
    return _llm


atexit.register(lambda: _llm.close() if _llm else None)


_LLM_SYSTEM = (
    "You map any phrase to ONE color word. "
    "Respond with ONLY a single color name (Japanese kanji/katakana, or English). "
    "No reasoning, no explanation, no punctuation."
)
_LLM_FEWSHOT = [
    {"role": "user",      "content": "夕焼けのオレンジ"},
    {"role": "assistant", "content": "橙"},
    {"role": "user",      "content": "深い海の色"},
    {"role": "assistant", "content": "マリンブルー"},
    {"role": "user",      "content": "深夜のコンビニ"},
    {"role": "assistant", "content": "蛍光白"},
    {"role": "user",      "content": "ancient ruins"},
    {"role": "assistant", "content": "黄土色"},
]


def _extract_color_from_llm(raw):
    if not raw:
        return None
    s = _clean_llm(raw)
    if not s:
        return None
    n = _norm(s)
    if n in UNIFIED_DB:
        return UNIFIED_DB[n]
    kw = _longest_match_in(n, _UNIFIED_KEYS_SORTED)
    if kw:
        return UNIFIED_DB[kw]
    for tok in re.split(r'[\s,、。\.・/／()（）「」\[\]【】]+', s):
        rgb = _resolve_color_name(tok)
        if rgb != (128, 128, 128):
            return rgb
    return None


def _semantic_neutral(text):
    """LLM が無効を返した時の文脈依存中立色 (旧 #64646E sentinel の代替)。"""
    n = _norm(text)
    warm = any(w in n for w in ("暖", "熱", "warm", "hot", "愛", "怒", "炎", "fire",
                                "情熱", "passion", "夕", "陽", "太陽"))
    cool = any(w in n for w in ("冷", "寒", "cool", "cold", "悲", "海", "sea", "sky",
                                "氷", "ice", "river", "川", "湖", "lake", "雨", "rain"))
    dark = any(w in n for w in ("暗", "闇", "dark", "深", "重", "沈", "heavy", "night",
                                "夜", "黒", "black", "絶望", "despair"))
    light = any(w in n for w in ("明", "軽", "light", "bright", "輝", "白", "white",
                                 "希望", "hope", "joy", "喜"))
    h = 0.06 if warm else (0.6 if cool else 0.1)
    s = 0.30
    v = 0.35 if dark else (0.85 if light else 0.6)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(round(c * 255)) for c in (r, g, b))


@lru_cache(maxsize=128)
def _llm_fallback(text):
    llm = _get_llm()
    res = llm.create_chat_completion(
        messages=[{"role": "system", "content": _LLM_SYSTEM},
                  *_LLM_FEWSHOT,
                  {"role": "user", "content": text}],
        temperature=0.0,
        max_tokens=16,
        stop=["\n", "。", "<think>", "</think>"],
    )
    raw = res.get("choices", [{}])[0].get("message", {}).get("content", "")
    rgb = _extract_color_from_llm(raw)
    raw2 = ""

    if rgb is None:
        res2 = llm.create_chat_completion(
            messages=[
                {"role": "system",
                 "content": _LLM_SYSTEM + " Output one Japanese kanji color noun only."},
                {"role": "user", "content": text},
            ],
            temperature=0.4,
            max_tokens=8,
            stop=["\n", "。"],
        )
        raw2 = res2.get("choices", [{}])[0].get("message", {}).get("content", "")
        rgb = _extract_color_from_llm(raw2)

    if DEBUG_LLM:
        logging.debug("[LLM] %r -> raw1=%r raw2=%r -> %s", text, raw[:40], raw2[:40], rgb)

    if rgb is not None:
        return apply_modifiers(rgb, text)
    return _semantic_neutral(text)


# ─────────────────────────────────────────────────────────────
# 🧪 テスト実行
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.environ.get("COLORTERM") != "truecolor":
        logging.warning("⚠️  警告: ターミナルがTrueColor非対応の場合、色ブロックが白くなります。")

    tests = [
        "橙色と白と青を混ぜた色",
        "桜色", "少し派手な赤", "抹茶と桜色の中間",
        "穏やかに流れる川のような青", "夕焼けのようなオレンジ",
        "星空のような紺", "とても明るいオレンジ",
        "やや暗いグレイッシュな青", "透明な水色",
        "爆発的な赤", "クールな青緑", "怒りが混ざった赤",
        "そうだ、京都に行こう", "北海道の雪景色",
        "宇宙の果ての色", "未来都市のネオンの色",
        "古代遺跡の石の色", "草原を歩く冒険者",
        "渋い秋の紅葉", "重い鉛のような灰色",
        "切ない思春期の恋心",
        "濡れた黒猫の毛色", "錆びた鉄柵の色", "雨上がりのアスファルト",
        "深夜のコンビニ蛍光灯", "待ち合わせに遅れそうな時の空気",
        "沈黙の重さを色にすると", "暖かくて少し濁った黄色",
        "透明だけど冷たい青", "青森の春はまだ遠い", "沖縄の海はどんな色？",
        "東京の夜は銀色に輝く", "未来の都市はシアンに光る",
        "古代の遺跡は黄土色に朽ちる", "冒険の始まりは茶色い土の匂い",
        "草原の風は緑色の息吹",
        "渋い渋谷の夜", "地味なオフィスの壁",
        "筋肉ムキムキ", "超絶かわいいピンク", "春を感じる優しい風",
        "黄金の太陽", "息が詰まりそうな深海の暗さ",
        "小豆色", "珊瑚色", "翡翠色", "琥珀色", "マリンブルー", "ライラック",
        "青10%、赤20%、緑30%を混ぜた色",
        "red and blue mix", "green blend yellow", "middle of pink and white",
        "鮮やかなピンク", "くすんだグレー",
        "宇宙の果てのような銀河の色",
        "ガラスの輝き", "ルビーの赤", "サファイアの青",
        "我こそは永遠なり", "悲しみの色", "喜びの色",
        "暗闇の中の一筋の光", "燃えるような赤", "冷たい氷の青",
        "そうです、私が変なおじさんです", "無の境地の色", "希望の光の色",
        "面白いを表す色", "怖いを表す色", "美しいを表す色",
    ]
    logging.info("🔵 色解決テスト (Qwen2.5 / UNIFIED最長一致 / sentinel撤廃)")
    logging.info("-" * 70)
    hex_counts = {}
    for t in tests:
        rgb = resolve_color(t)
        h = rgb_to_hex(*rgb)
        hex_counts[h] = hex_counts.get(h, 0) + 1
        logging.info(f"{t:38} → {h} {get_ansi_block(*rgb)}")
    logging.info("\n" + "-" * 70)
    dup = {k: v for k, v in hex_counts.items() if v > 1}
    logging.info("重複色: %s", dup or 'なし')
    if "#64646E" in hex_counts:
        logging.info("❌ #64646E が %s 件残っています", hex_counts['#64646E'])
    else:
        logging.info("✅ #64646E はゼロ")
