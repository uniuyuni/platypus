
import os
import logging
import numpy as np
import json
import math
from datetime import datetime as dt

import cores.core as core
from cores import pmck_store
import config
import define
import effects
import utils.utils as utils
import macos as device


_MESH_DEBUG = os.getenv("PLATYPUS_DEBUG_MESH_WARP", "0").strip().lower() in ("1", "true", "yes", "on")
from enums import ImageFidelity
from utils import rating_utils
from utils.rating_io import PMCK_RAW_RATING_KEY, merge_raw_pmck_rating

# pmck 内の「重い」primary_param キー（フル解像時のみシリアライズ／プレビュー時は読み飛ばし）
HEAVY_PRIMARY_PARAM_KEYS = (
    'ai_noise_reduction_result',
    'ai_noise_reduction_content_key',
    'ai_noise_reduction_source_signature',
    'inpaint_diff_list',
    'patchmatch_inpaint_diff_list',
    'color_match_source_image',
    'heavy_saved_at_fidelity',
)


def _param2_has_substantive_heavy_payload(param2: dict) -> bool:
    """本当に重いペイロードがあるときだけ（マーカー単体は False）。"""
    v = param2.get("ai_noise_reduction_result", None)
    if v is not None and not (isinstance(v, (list, tuple)) and len(v) == 0):
        return True
    for k in ("inpaint_diff_list", "patchmatch_inpaint_diff_list"):
        w = param2.get(k, None)
        if w and isinstance(w, (list, tuple)) and len(w) > 0:
            return True
    v = param2.get("color_match_source_image", None)
    if v is not None and not (isinstance(v, (list, tuple)) and len(v) == 0):
        return True
    return False

# レンズ3: lensfun 実効（color/subpixel/geometry）の書き戻しは SPECIAL、.pmck の primary には出さない。
# lensfun_user は「ユーザー3つ」だけ。RAW の TCA は LibRaw 側を優先するため、Subpixel はデフォルト OFF。
# .pmck に積むのは「ユーザー指定がデフォルトと異なる」場合。
LENSFUN_USER_KEY = "lensfun_user"
DEFAULT_LENSFUN_USER = (True, False, True)
# pmck には含めない内部状態（capability）
LENSFUN_STATE_KEY = "_lensfun_state"
_LENSFUN_CAPABILITY_KEY = "lensfun_capability"
_LENSFUN_EFFECTIVE_KEY = "lensfun_effective"


def normalize_lensfun_user(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return (bool(val[0]), bool(val[1]), bool(val[2]))
    return None


def get_lensfun_user_tuple(param):
    raw = param.get(LENSFUN_USER_KEY)
    n = normalize_lensfun_user(raw)
    if n is not None:
        return n
    return DEFAULT_LENSFUN_USER


def _ensure_lensfun_state(param) -> dict:
    st = param.get(LENSFUN_STATE_KEY)
    if not isinstance(st, dict):
        st = {}
        param[LENSFUN_STATE_KEY] = st
    return st


def clear_lensfun_capability(param) -> None:
    st = _ensure_lensfun_state(param)
    st.pop(_LENSFUN_CAPABILITY_KEY, None)


def set_lensfun_effective_tuple(param, effective) -> None:
    st = _ensure_lensfun_state(param)
    n = normalize_lensfun_user(effective)
    if n is None:
        st.pop(_LENSFUN_EFFECTIVE_KEY, None)
        return
    st[_LENSFUN_EFFECTIVE_KEY] = n


def get_lensfun_effective_tuple(param):
    st = param.get(LENSFUN_STATE_KEY)
    if isinstance(st, dict):
        n = normalize_lensfun_user(st.get(_LENSFUN_EFFECTIVE_KEY))
        if n is not None:
            return n
    return get_lensfun_user_tuple(param)


def set_lensfun_capability(param, capability) -> None:
    st = _ensure_lensfun_state(param)
    n = normalize_lensfun_user(capability)
    if n is None:
        st.pop(_LENSFUN_CAPABILITY_KEY, None)
        return
    st[_LENSFUN_CAPABILITY_KEY] = n


def _get_lensfun_capability(param):
    st = param.get(LENSFUN_STATE_KEY)
    if not isinstance(st, dict):
        return None
    return normalize_lensfun_user(st.get(_LENSFUN_CAPABILITY_KEY))


def should_persist_lensfun_in_pmck(param) -> bool:
    """
    ルール（ユーザー指定）:
    - デフォルト (Color=True, Subpixel=False, Geometry=True) と同じなら保存しない
    - ユーザーが Subpixel ON などデフォルトから変えた場合だけ保存する
    """
    t = get_lensfun_user_tuple(param)
    return t != DEFAULT_LENSFUN_USER


def _strip_default_lensfun_from_pmck_primary_param(param2: dict) -> None:
    """
    primary_param の lensfun_user を正規化し、デフォルト（不正値含む扱い）ならキーを落とす。
    has_user_lens / delete_special 以外の経路で混入した DEFAULT_LENSFUN_USER も消す最終防衛。
    """
    k = LENSFUN_USER_KEY
    if k not in param2:
        return
    n = normalize_lensfun_user(param2.get(k))
    if n is None or n == DEFAULT_LENSFUN_USER:
        param2.pop(k, None)


def collapse_default_lensfun_user(param: dict) -> None:
    """
    ユーザーレンズ3がデフォルトまたは正規化不能ならキーを落とす。
    Kivy/numpy の 0/1 や list/msgpack 由来で == 比較だけでは delete_default から抜けないのを防ぐ。
    """
    if LENSFUN_USER_KEY not in param:
        return
    n = normalize_lensfun_user(param.get(LENSFUN_USER_KEY))
    if n is None or n == DEFAULT_LENSFUN_USER:
        param.pop(LENSFUN_USER_KEY, None)


def _sync_lensfun_from_loaded_primary(param):
    """
    .pmck primary を param に入れた直後、lensfun 関連をタプル表現へ正規化する。
    """
    param.pop("_lens_trinity_wrote_from_lensfun", None)  # 旧内部フラグ
    clear_lensfun_capability(param)
    set_lensfun_effective_tuple(param, None)
    if LENSFUN_USER_KEY in param and param[LENSFUN_USER_KEY] is not None:
        n = normalize_lensfun_user(param[LENSFUN_USER_KEY])
        if n is not None:
            param[LENSFUN_USER_KEY] = n
    collapse_default_lensfun_user(param)
    set_lensfun_effective_tuple(param, get_lensfun_user_tuple(param))


# スイッチだけは従来どおり param に（pmck に残り得る＝ユーザーがオフにした場合）
_LENS_LIKE_DEFAULT_TRUE_KEYS = (
    "switch_lens_modifier",
)


def _param2_strips_nonsubstance_mutable(p: dict) -> None:
    """
    実質の編集に含めない付箋（解像度マーカー・未伴う heavy 等）を除去。
    `serialize` の「空判定用コピー」だけでなく、書き込み直前の param2 実体に必ず同じ操作を掛ける。
    """
    p.pop("rating", None)
    p.pop("image_fidelity", None)
    # lensfun_user は SPECIAL のため param2 に乗らない。pmck へは serialize が直接積む。
    for k in _LENS_LIKE_DEFAULT_TRUE_KEYS:
        v = p.get(k, True)
        if v == False:  # noqa: E712
            continue
        p.pop(k, None)
    v = p.get("ai_noise_reduction_result", None)
    if v is None or (isinstance(v, (list, tuple)) and len(v) == 0):
        p.pop("ai_noise_reduction_result", None)
    for k in ("inpaint_diff_list", "patchmatch_inpaint_diff_list"):
        w = p.get(k, None)
        if w is None or (isinstance(w, (list, tuple)) and len(w) == 0):
            p.pop(k, None)
    v = p.get("color_match_source_image", None)
    if v is None or (isinstance(v, (list, tuple)) and len(v) == 0):
        p.pop("color_match_source_image", None)
    if not _param2_has_substantive_heavy_payload(p):
        p.pop("heavy_saved_at_fidelity", None)
        p.pop("ai_noise_reduction_content_key", None)
        p.pop("ai_noise_reduction_source_signature", None)


def _param2_effective_user_substance_remaining(param2: dict) -> dict:
    """
    delete_special 後の primary 相当辞書から、パイプライン付帯だけを除いた「実質の編集」残り。
    ここが空でマスクも無いときは .pmck を作らない。
    """
    if not param2:
        return {}
    p = param2.copy()
    _param2_strips_nonsubstance_mutable(p)
    return p

# 重要だがセーブしないパラメータ
SPECIAL_PARAM = [
    # for set_image_param
    'original_img_size',
    'img_size',
    #'crop_rect',
    'disp_info',
    '_mask1_restore_view_after_submit',
    '_source_file_path',
    '_ai_noise_reduction_result_deferred',
    # セッション中のみ（フル/プレビュー・_serialize 重い判定用）。.pmck primary には出さない
    "image_fidelity",
    # レンズ3のユーザー意図。メモリ・copy_special 用。pmck へは serialize が (T,T,T) 以外のときだけ明示的に書く
    LENSFUN_USER_KEY,
    LENSFUN_STATE_KEY,
    # for effects.LensModifierEffect: 実効3値は永続に含めない
    'lens_modifier',
    'exif_data',
    # for imageset._set_temperature
    'color_temperature_reset',
    'color_tint_reset',
    'color_Y',
    # for effects.CropEffect
    'matrix',
    'crop_enable',
    # for core.apply_zero_wrap: ジオメトリ変換後のコンテンツ四辺形と変換キャンバス一辺（ランタイム専用）
    '_zero_wrap_content_quad',
    '_zero_wrap_canvas_size',
    # for effecs.Inpaint
    'inpaint',
    'inpaint_predict',
    # for effects.PatchMatchInpaint
    'patchmatch_inpaint',
    'patchmatch_inpaint_predict',
    # for effects.LUTEffect
    'lut_path',
    # for effects.LUTEffect (input stage auto exposure)
    'rgb_or_raw',
    'auto_exposure',
]

DO_NOT_COPY_SPECIAL_PARAM = {
    # Geometryタブ表示中かどうかは実行時状態。履歴やReset復元には混ぜない。
    'crop_enable',
}

# セーブするが、初期化時にリセットするパラメータ
REMAIN_PARAM = [
    "crop_rect",
]

# 正規化込みの読み出し、設定
def get_crop_rect(param, none_value=None):
    crop_rect = param.get('crop_rect', none_value)
    if crop_rect is not None:
        maxsize = max(param['original_img_size'])
        if crop_rect is none_value:
            crop_rect = (
                crop_rect[0] / maxsize,
                crop_rect[1] / maxsize,
                crop_rect[2] / maxsize,
                crop_rect[3] / maxsize,
            )
        
        crop_rect2 = (
            int(crop_rect[0] * maxsize),
            int(crop_rect[1] * maxsize),
            int(crop_rect[2] * maxsize),
            int(crop_rect[3] * maxsize),
        )
    else:
        crop_rect2 = None
    
    return crop_rect2

def set_crop_rect(param, crop_rect):
    if crop_rect is not None:
        maxsize = max(param['original_img_size'])
        crop_rect2 = (
            crop_rect[0] / maxsize,
            crop_rect[1] / maxsize,
            crop_rect[2] / maxsize,
            crop_rect[3] / maxsize,
        )
        param['crop_rect'] = crop_rect2

def get_disp_info(param, none_value=None):
    disp_info = param.get('disp_info', none_value)
    if disp_info is not None:
        maxsize = max(param['original_img_size'])
        if disp_info is none_value:
            disp_info = (
                disp_info[0] / maxsize,
                disp_info[1] / maxsize,
                disp_info[2] / maxsize,
                disp_info[3] / maxsize,
                disp_info[4],
            )

        disp_info2 = (
            int(disp_info[0] * maxsize),
            int(disp_info[1] * maxsize),
            int(disp_info[2] * maxsize),
            int(disp_info[3] * maxsize),
            disp_info[4],
        )
    else:
        disp_info2 = None

    return disp_info2

def set_disp_info(param, disp_info):
    if disp_info is not None:
        maxsize = max(param['original_img_size'])
        disp_info2 = (
            disp_info[0] / maxsize,
            disp_info[1] / maxsize,
            disp_info[2] / maxsize,
            disp_info[3] / maxsize,
            disp_info[4],
        )
        param['disp_info'] = disp_info2

def denorm_param(param, val):
    if val is not None:
        if type(val) == tuple or type(val) == list:
            x = val[0] * param['original_img_size'][0]
            y = val[1] * param['original_img_size'][1]
            return (x, y)
        
        return val * max(param['original_img_size'])

    return None

def norm_param(param, val):
    if val is not None:
        if type(val) == tuple or type(val) == list:
            x = val[0] / param['original_img_size'][0]
            y = val[1] / param['original_img_size'][1]
            return (x, y)
        
        return val / max(param['original_img_size'])
    return None

#-------------------------------------------------
def has_original_img_size(param):
    """幾何・クロップ・パイプラインが前提とする。SPECIAL_PARAM のため pmck 単体では欠ける。"""
    return param.get('original_img_size') is not None


def apply_original_geometry_if_missing(param, img):
    """
    デコード済み画像があるときだけ、欠けている original_img_size / img_size を実画像の寸法で埋める。
    crop_rect / disp_info は変更しない（pmck 由来の編集を壊さないため）。
    """
    if img is None or has_original_img_size(param):
        return
    h, w = img.shape[:2]
    param['original_img_size'] = (w, h)
    param['img_size'] = (w, h)


def ensure_initial_crop_rect(param):
    """original_img_size があるなら、未初期化の crop_rect だけ全体表示で補完する。"""
    if not has_original_img_size(param) or get_crop_rect(param) is not None:
        return False
    width, height = param['original_img_size']
    set_crop_rect(param, core.get_initial_crop_rect(width, height))
    return True


# 画像の初期設定を設定する
def set_image_param(param, img):
    height, width = img.shape[:2]

    # イメージサイズをパラメータに入れる
    param['original_img_size'] = (width, height)
    param['img_size'] = (width, height)
    set_crop_rect(param, get_crop_rect(param, core.get_initial_crop_rect(width, height)))
    set_disp_info(param, core.convert_rect_to_info(get_crop_rect(param), config.get_preview_texture_side()/max(param['original_img_size'])))
    # 新規デコード: pmck 未読込
    clear_lensfun_capability(param)

    return (width, height)

def set_image_param_for_mask2(param, size):
    width, height = size
    param['original_img_size'] = (width, height)

def set_temperature_to_param(param, temp, tint, Y):
    param['color_temperature_reset'] = temp
    param['color_temperature'] = temp
    param['color_tint_reset'] = tint
    param['color_tint'] = tint
    param['color_Y'] = Y

#-------------------------------------------------

def delete_special_param(param):
    p = param.copy()

    for key in SPECIAL_PARAM:
        try:
            del p[key]
        except KeyError:
            # 元々存在しないキーは削除不要（無害）
            pass

    return p

def delete_not_special_param(param):
    p = param.copy()

    for key in param.keys():
        if key not in SPECIAL_PARAM and key not in REMAIN_PARAM:
            try:
                del p[key]
            except KeyError:
                # 元々存在しないキーは削除不要（無害）
                pass

    return p

def copy_special_param(tar, src):
    for key in SPECIAL_PARAM:
        if key in DO_NOT_COPY_SPECIAL_PARAM:
            continue
        try:
            val = src[key]
            tar[key] = val
        except KeyError:
            # src側に無いキーはコピー対象外として無視（無害）
            pass

def copy_remain_param(tar, src):
    for key in REMAIN_PARAM:
        try:
            val = src[key]
            tar[key] = val
        except KeyError:
            # src側に無いキーはコピー対象外として無視（無害）
            pass

def _inpaint_dump(param, list_name='inpaint_diff_list'):
    inpaint_diff_list = param.get(list_name, None)
    if inpaint_diff_list is not None:
        inpaint_diff_list_dumps = []
        for inpaint_diff in inpaint_diff_list:
            inpaint_diff_list_dumps.append((inpaint_diff.type, inpaint_diff.disp_info, utils.convert_image_to_list(inpaint_diff.image)))
        param[list_name] = inpaint_diff_list_dumps

def _inpaint_load(param, list_name='inpaint_diff_list'):
    inpaint_diff_list_dumps = param.get(list_name, None)
    if inpaint_diff_list_dumps is not None:
        inpaint_diff_list = []
        for inpaint_diff_dump in inpaint_diff_list_dumps:
            inpaint_diff = effects.InpaintDiff(type=inpaint_diff_dump[0], disp_info=inpaint_diff_dump[1], image=utils.convert_image_from_list(inpaint_diff_dump[2]))
            inpaint_diff_list.append(inpaint_diff)
        param[list_name] = inpaint_diff_list

def _ai_noise_reduction_dump(param):
    ai_noise_reduction_result = param.get('ai_noise_reduction_result', None)
    if ai_noise_reduction_result is not None:
        ai_noise_reduction_result = utils.convert_image_to_list(ai_noise_reduction_result)
        param['ai_noise_reduction_result'] = ai_noise_reduction_result

def _ai_noise_reduction_load(param):
    ai_noise_reduction_result = param.get('ai_noise_reduction_result', None)
    if ai_noise_reduction_result is not None:
        ai_noise_reduction_result = utils.convert_image_from_list(ai_noise_reduction_result)
        param['ai_noise_reduction_result'] = ai_noise_reduction_result

def _color_match_source_dump(param):
    src = param.get('color_match_source_image', None)
    if src is not None:
        param['color_match_source_image'] = utils.convert_image_to_list(src)

def _color_match_source_load(param):
    src = param.get('color_match_source_image', None)
    if src is not None and not isinstance(src, np.ndarray):
        param['color_match_source_image'] = utils.convert_image_from_list(src)

def _serialize_param(param, include_heavy=True):
    if include_heavy:
        _inpaint_dump(param, 'inpaint_diff_list')
        _inpaint_dump(param, 'patchmatch_inpaint_diff_list')
        _ai_noise_reduction_dump(param)
        _color_match_source_dump(param)
        param['heavy_saved_at_fidelity'] = ImageFidelity.FULL.value
    else:
        for k in HEAVY_PRIMARY_PARAM_KEYS:
            param.pop(k, None)

def _msgpack_safe_key(key):
    if isinstance(key, np.generic):
        key = key.item()
    if isinstance(key, (str, bytes, int, float, bool)) or key is None:
        return key
    if isinstance(key, tuple):
        return ",".join(str(_msgpack_safe_key(v)) for v in key)
    return str(key)

def _msgpack_safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            _msgpack_safe_key(k): _msgpack_safe_value(v)
            for k, v in value.items()
        }
    if isinstance(value, tuple):
        return [_msgpack_safe_value(v) for v in value]
    if isinstance(value, list):
        return [_msgpack_safe_value(v) for v in value]
    return value

def _deserialize_param(param, load_heavy=True):
    param['disp_info'] = core.convert_rect_to_info(param['crop_rect'], config.get_preview_texture_side()/max(param['original_img_size']))
    if load_heavy:
        _inpaint_load(param, 'inpaint_diff_list')
        _inpaint_load(param, 'patchmatch_inpaint_diff_list')
        _ai_noise_reduction_load(param)
        _color_match_source_load(param)
    else:
        param.pop('ai_noise_reduction_result', None)
        param.pop('color_match_source_image', None)
        param['inpaint_diff_list'] = []
        param['patchmatch_inpaint_diff_list'] = []
        param.pop('heavy_saved_at_fidelity', None)

def _pmck_shell_empty_primary() -> dict:
    return pmck_store.empty_pmck()


def _iter_mask2_dicts(mask2_list):
    """mask_editor2.serialize() が返す mask2 ツリー(コンポジットの mask_list を含む)を
    再帰的に平坦化する。"""
    for m in mask2_list or []:
        yield m
        for child, _maskop in m.get('mask_list', []) or []:
            yield from _iter_mask2_dicts([child])


def _mask2_bitmap_key_to_str(cache_key):
    """cache_keys の msgpack セーフなキー(list)を mask2_bitmaps の辞書キー用文字列に変換する。
    _msgpack_safe_key の comma-join は TargetText の自由入力テキスト等を含むキーでは非可逆
    なため使わず、json でロスレスに往復できる形にする。"""
    return json.dumps(cache_key, sort_keys=True, ensure_ascii=True)


def _mask2_bitmap_key_from_str(key_str):
    return json.loads(key_str)


def _collect_mask2_bitmaps(mask2_list, mask_editor2):
    """シリアライズ済みマスクツリーが参照する image_mask_cache_key を集め、AIImageCache
    共有ストアから圧縮済みビットマップを取得する(保存時点で参照されているキーのみ =自然GC)。
    同じキーを複数マスクが共有していても1回だけ含める。"""
    store = getattr(mask_editor2, "ai_image_cache", None)
    if store is None or not hasattr(store, "get_serialized_mask_bitmap"):
        return None
    result = {}
    for m in _iter_mask2_dicts(mask2_list):
        key = m.get('image_mask_cache_key')
        if key is None:
            continue
        key_str = _mask2_bitmap_key_to_str(key)
        if key_str in result:
            continue
        compressed = store.get_serialized_mask_bitmap(key)
        if compressed is None:
            continue
        result[key_str] = compressed
    return result or None


def _merge_mask2_bitmaps(ser, mask_editor2):
    """mask2_bitmaps を AIImageCache 共有ストアへマージ put する。
    mask_editor2.deserialize(ser) の前に呼ぶこと(マスク側の get_mask_bitmap が参照できるように)。"""
    bitmaps = ser.get("mask2_bitmaps") if isinstance(ser, dict) else None
    logging.warning("[DIAG-REINFER] _merge_mask2_bitmaps called n_bitmaps=%s store_id=%s suspend=%s",
                    None if bitmaps is None else len(bitmaps),
                    id(getattr(mask_editor2, "ai_image_cache", None)),
                    getattr(mask_editor2, "_bitmap_sweep_suspend_depth", "n/a"))
    if not bitmaps:
        return
    store = getattr(mask_editor2, "ai_image_cache", None)
    if store is None or not hasattr(store, "put_mask_bitmap"):
        return
    for key_str, compressed in bitmaps.items():
        try:
            cache_key = _mask2_bitmap_key_from_str(key_str)
            # キー→内容は不変(キーは推論入力由来)なので、既にストアにあれば解凍も put も
            # 省略する。undo/redo のたびに再解凍・圧縮キャッシュ破棄が起きるのを防ぐ。
            if store.get_mask_bitmap(cache_key) is not None:
                continue
            image = utils.convert_image_from_list(compressed)
            store.put_mask_bitmap(cache_key, image)
        except Exception:
            logging.exception("mask2_bitmaps のマージに失敗しました key=%s", key_str)


def serialize(param, mask_editor2, file_path=None):
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y/%m/%d')
    mask_dict = mask_editor2.serialize()
    ai_image_cache = None
    serialize_ai_image_cache = getattr(mask_editor2, "serialize_ai_image_cache", None)
    if callable(serialize_ai_image_cache):
        ai_image_cache = serialize_ai_image_cache()

    # セーブしないパラメータを削除
    param2 = delete_special_param(param)
    if not isinstance(param2, dict):
        param2 = {}
    else:
        param2 = param2.copy()
    # 編集用 primary_param には星を入れない。永続化は (1)RAW… .pmck トップの platypus_raw_rating
    # (2)RGB… 画像ファイルの XMP。ここは pmck 内 primary のシリアライズのため必ず排除する。
    param2.pop("rating", None)
    _param2_strips_nonsubstance_mutable(param2)

    include_heavy = param.get("image_fidelity") == ImageFidelity.FULL.value
    has_mask = bool(mask_dict)
    eff = _param2_effective_user_substance_remaining(param2)
    # pmck に載せるのは should_persist_lensfun_in_pmck が True のときだけ。
    has_user_lens = should_persist_lensfun_in_pmck(param)

    if not eff and not has_mask and not has_user_lens:
        return None
    if not eff and has_mask:
        param2 = {}
    elif eff:
        _serialize_param(param2, include_heavy=include_heavy)

    _param2_strips_nonsubstance_mutable(param2)
    if has_user_lens:
        t = get_lensfun_user_tuple(param)
        param2[LENSFUN_USER_KEY] = (bool(t[0]), bool(t[1]), bool(t[2]))

    _strip_default_lensfun_from_pmck_primary_param(param2)
    if not has_mask and not param2:
        return None
    if not param2 and has_mask:
        param2 = {}

    ser = {
        'make': "Platypus",
        'date': tstr,
        'version': define.VERSION,
        'primary_param': param2,
    }
    if mask_dict is not None:
        ser.update(mask_dict)
        mask2_bitmaps = _collect_mask2_bitmaps(mask_dict.get("mask2"), mask_editor2)
        if mask2_bitmaps is not None:
            ser["mask2_bitmaps"] = mask2_bitmaps
    if ai_image_cache is not None:
        ser["ai_image_cache"] = ai_image_cache

    return _msgpack_safe_value(ser)

def deserialize(ser, param, mask_editor2, load_heavy=True):
    set_ai_image_cache = getattr(mask_editor2, "set_serialized_ai_image_cache", None)
    if callable(set_ai_image_cache):
        set_ai_image_cache(ser.get("ai_image_cache"))

    pp = ser.get("primary_param")
    if not isinstance(pp, dict):
        pp = {}
    else:
        pp = pp.copy()
    pp.pop("rating", None)  # 旧形式・レーティングは primary に還元しない
    if not load_heavy:
        if pp.get("ai_noise_reduction_result") is not None:
            param["_ai_noise_reduction_result_deferred"] = True
        for k in HEAVY_PRIMARY_PARAM_KEYS:
            pp.pop(k, None)
    param.update(pp)
    _sync_lensfun_from_loaded_primary(param)

    # 色々処理変換
    _deserialize_param(param, load_heavy=load_heavy)

    # AI マスクビットマップ共有ストアの復元とマスク生成は sweep 抑止下で行う。
    # clear_mask() は on_structure_change 経由で mark-and-sweep を走らせるため、先に
    # _merge_mask2_bitmaps を呼ぶと mask_list が空の状態で sweep され、復元したばかりの
    # ビットマップが即座に掃除されてしまう(→ 再読み込み時に AI 再推論)。そこで
    #   1. clear_mask (sweep はここでは無害)
    #   2. _merge_mask2_bitmaps でストアへ復元(各マスクの deserialize が参照する)
    #   3. mask_editor2.deserialize でマスク生成
    # の順に並べ、全体を suspend_bitmap_sweep で囲って中間 sweep を止め、最後に一度だけ
    # 掃除する(複数 AI マスクが別コンポジットにいても互いのビットマップを消さないため)。
    suspend = getattr(mask_editor2, "suspend_bitmap_sweep", None)
    resume = getattr(mask_editor2, "resume_bitmap_sweep", None)
    if callable(suspend):
        suspend()
    try:
        mask_editor2.clear_mask()
        _merge_mask2_bitmaps(ser, mask_editor2)
        mask_dict = ser.get("mask2", None)
        if mask_dict is not None:
            mask_editor2.deserialize(ser)
    finally:
        if callable(resume):
            resume()


def merge_heavy_from_pmck(file_path, param, mask_editor2, cached_dict=None):
    """
    RAW がプレビュー→フルに遷移したとき等、既に軽い内容だけ読み込んだ後に
    pmck から重いペイロードだけをマージする。

    cached_dict: load_json で取得済みの msgpack デコード結果。渡されればファイル再読込・
    再パースをスキップする。
    """
    if file_path is None:
        return
    if cached_dict is not None:
        d = cached_dict
    else:
        d = pmck_store.read_image(file_path)
        if d is None:
            return
    pp = d.get('primary_param')
    if not pp or pp.get('heavy_saved_at_fidelity') != ImageFidelity.FULL.value:
        return
    for k in (
        'ai_noise_reduction_result',
        'ai_noise_reduction_content_key',
        'ai_noise_reduction_source_signature',
        'inpaint_diff_list',
        'patchmatch_inpaint_diff_list',
        'color_match_source_image',
    ):
        if k in pp:
            param[k] = pp[k]
    param.pop('_ai_noise_reduction_result_deferred', None)
    _ai_noise_reduction_load(param)
    _inpaint_load(param, 'inpaint_diff_list')
    _inpaint_load(param, 'patchmatch_inpaint_diff_list')
    _color_match_source_load(param)

def save_json(file_path, param, mask_editor2, raw_sidecar_rating: int = 0):
    if file_path is None:
        return False
    is_raw = bool(file_path) and rating_utils.is_raw_path(file_path)
    raw_r = int(raw_sidecar_rating or 0) if is_raw else 0
    empty = is_empty_param(param, mask_editor2)
    if empty:
        if is_raw and raw_r > 0:
            # RAW で実質編集なしの場合は、古い primary/mask を残さず rating 専用の最小 .pmck を再構築する。
            ser = _pmck_shell_empty_primary()
            ser[PMCK_RAW_RATING_KEY] = raw_r
            return pmck_store.write_image(file_path, ser)
        pmck_store.delete_image(file_path)
        return False
    ser = serialize(param, mask_editor2, file_path=file_path)
    if is_raw:
        if raw_r > 0:
            if ser is None:
                ser = _pmck_shell_empty_primary()
            ser[PMCK_RAW_RATING_KEY] = raw_r
        else:
            if ser is not None:
                ser.pop(PMCK_RAW_RATING_KEY, None)
    else:
        if ser is not None:
            ser.pop(PMCK_RAW_RATING_KEY, None)  # RGB に RAW 専用キーが乗るのを防ぐ
    if ser is not None:
        return pmck_store.write_image(file_path, ser)
    pmck_store.delete_image(file_path)
    return False

def load_json(file_path, param, mask_editor2, load_heavy=True):
    if file_path is None:
        return None
    dict_ = pmck_store.read_image(file_path)
    if dict_ is None:
        set_ai_image_cache = getattr(mask_editor2, "set_serialized_ai_image_cache", None)
        if callable(set_ai_image_cache):
            set_ai_image_cache(None)
        return None
    set_ai_image_cache = getattr(mask_editor2, "set_serialized_ai_image_cache", None)
    if callable(set_ai_image_cache):
        set_ai_image_cache(dict_.get("ai_image_cache"))

    pp = dict_.get("primary_param") or {}
    has_geometry_or_mask = (
        "crop_rect" in pp
        or "original_img_size" in pp
        or bool(dict_.get("mask2"))
    )
    ppx = {k: v for k, v in pp.items() if k != "rating"}
    if not has_geometry_or_mask and not ppx:
        return dict_
    # tupleがlistになってしまうのでtupleに戻す
    try:
        if dict_.get('primary_param') and 'crop_rect' in dict_['primary_param']:
            dict_['primary_param']['crop_rect'] = tuple(dict_['primary_param']['crop_rect'])
    except (KeyError, TypeError):
        # tuple化に失敗しても後続の deserialize は元の形式のまま処理を続行できる
        pass
    deserialize(dict_, param, mask_editor2, load_heavy=load_heavy)
    param.pop("rating", None)  # 旧 pmck
    return dict_

def is_empty_param(param, mask_editor2):
    param2 = delete_special_param(param)
    if not isinstance(param2, dict):
        param2 = {}
    else:
        param2 = param2.copy()
    param2.pop("rating", None)
    _param2_strips_nonsubstance_mutable(param2)
    if should_persist_lensfun_in_pmck(param):
        return False
    if _param2_effective_user_substance_remaining(param2):
        return False
    mask_list = mask_editor2.get_mask_list()
    if mask_list is None or len(mask_list) == 0:
        return True
    return False


def delete_empty_param_json(file_path):
    if file_path is not None:
        deleted = pmck_store.delete_image(file_path)
        legacy_json_path = file_path + '.json'

        if os.path.exists(legacy_json_path):
            os.remove(legacy_json_path)
            return True

        if deleted:
            return True

    return False


#-------------------------------------------------

def param_to_tcg_info(param):
    """
    パラメータをTCGパラメータに変換

    param: パラメータ辞書(Noneなら全て0に設定)
    戻り値: 情報辞書
    """
    tcg_info = {}
    preview_side = config.get_preview_texture_side()
    tcg_info['original_img_size'] = param.get('original_img_size', (preview_side, preview_side))
    tcg_info['disp_info'] = param.get('disp_info', (0, 0, 1.0, 1.0, 1.0))
    tcg_info['rotation'] = math.radians(param.get('rotation', 0.0))
    tcg_info['rotation2'] = math.radians(param.get('rotation2', 0.0))
    tcg_info['flip_mode'] = param.get('flip_mode', 0)
    tcg_info['matrix'] = param.get('matrix', np.eye(3))

    return tcg_info

def window_to_tcg(cx, cy, widget, texture_size, tcg_info, normalize=True):
    """
    ウインドウ座標からTCG座標に変換
    cx, cy: TCG座標
    widget: 表示するウィジェット
    texture_size: テクスチャサイズ
    ref_image: 参照イメージ
    tcg_info: 回転情報
    normalize: 正規化するかどうか
    戻り値: TCG座標
    """
    disp_info = get_disp_info(tcg_info)
    imax = max(tcg_info['original_img_size'][0] / 2, tcg_info['original_img_size'][1] / 2)
    wx, wy = widget.to_window(*widget.pos)
    cx, cy = cx - wx, cy - wy
    cx, cy = cx / device.dpi_scale(), cy / device.dpi_scale()
    margin_x, margin_y = (widget.size[0] / device.dpi_scale() - texture_size[0])/2, (widget.size[1] / device.dpi_scale() - texture_size[1])/2
    cx, cy = cx - margin_x, cy - margin_y
    cx, cy = cx, texture_size[1] - cy
    _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*texture_size, disp_info)
    cx, cy = cx - offset_x, cy - offset_y
    cx, cy = cx / disp_info[4], cy / disp_info[4]
    cx, cy = cx + disp_info[0], cy + disp_info[1]
    cx, cy = cx - imax, cy - imax # ここで - (imax - self.current_image.shape[0] / 2)の分の計算もやってる
    cx, cy = center_rotate_invert(cx, cy, tcg_info)
    if normalize:
        cx, cy = norm_param(tcg_info, (cx, cy))
    return (cx, cy)

def tcg_to_window(cx, cy, widget, texture_size, tcg_info, normalize=True):
    """
    TCG座標をウインドウ座標に変換
    cx, cy: TCG座標
    widget: 表示するウィジェット
    texture_size: テクスチャサイズ
    ref_image: 参照イメージ
    tcg_info: 回転情報
    normalize: 正規化するかどうか
    戻り値: ウインドウ座標
    """
    disp_info = get_disp_info(tcg_info)
    imax = max(tcg_info['original_img_size'][0] / 2, tcg_info['original_img_size'][1] / 2)

    # 環境変数 PLATYPUS_DEBUG_MESH_WARP=1 のときだけ debug log。
    # 通常は _MESH_DEBUG=False で早期 skip し、hot path (秒間数百回) の overhead を避ける。
    _is_mesh_dbg = (
        _MESH_DEBUG
        and normalize and abs(cx) < 1e-9 and abs(cy) < 1e-9
        and type(widget).__name__ == 'MeshWarpWidget'
    )
    if _is_mesh_dbg:
        try:
            logging.warning(
                "[T2W_IN] widget_id=%s widget.size=%s widget.pos=%s texture_size=%s "
                "disp_info=%s orig=%s dpi=%s",
                id(widget),
                tuple(widget.size), tuple(widget.pos), tuple(texture_size),
                disp_info, tcg_info['original_img_size'], device.dpi_scale(),
            )
        except Exception:
            # デバッグログ自体の失敗でホットパスを止めないためのベストエフォート
            pass

    if normalize:
        cx, cy = denorm_param(tcg_info, (cx, cy))
    cx, cy = center_rotate(cx, cy, tcg_info)
    cx, cy = cx + imax, cy + imax
    cx, cy = cx - disp_info[0], cy - disp_info[1]
    cx, cy = cx * disp_info[4], cy * disp_info[4]
    _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*texture_size, disp_info)
    cx, cy = cx + offset_x, cy + offset_y
    cx, cy = cx, texture_size[1] - cy
    margin_x, margin_y = (widget.size[0] / device.dpi_scale() - texture_size[0])/2, (widget.size[1] / device.dpi_scale() - texture_size[1])/2
    cx, cy = cx + margin_x, cy + margin_y
    cx, cy = cx * device.dpi_scale(), cy * device.dpi_scale()
    wx, wy = widget.to_window(*widget.pos)
    cx, cy = cx + wx, cy + wy

    if _is_mesh_dbg:
        try:
            logging.warning(
                "[T2W_OUT] widget_id=%s margin=(%s,%s) widget.to_window(pos)=(%s,%s) "
                "final_cx_cy=(%s, %s)",
                id(widget), margin_x, margin_y, wx, wy, cx, cy,
            )
        except Exception:
            # デバッグログ自体の失敗でホットパスを止めないためのベストエフォート
            pass

    return (cx, cy)

def window_to_tcg_scale(x, tcg_info):
    # ワールド座標にスケーリングだけ適用する
    if isinstance(x, (tuple, list)):
        return (x[0] / tcg_info['disp_info'][4] / device.dpi_scale(), x[1] / tcg_info['disp_info'][4] / device.dpi_scale())
    return x / tcg_info['disp_info'][4] / device.dpi_scale()

def tcg_to_window_scale(x, tcg_info):
    # TCG座標にスケーリングだけ適用する
    if isinstance(x, (tuple, list)):
        return (x[0] * tcg_info['disp_info'][4] * device.dpi_scale(), x[1] * tcg_info['disp_info'][4] * device.dpi_scale())
    return x * tcg_info['disp_info'][4] * device.dpi_scale()

def tcg_to_image_scale(x, tcg_info):
    # TCG座標にスケーリングだけ適用する
    if isinstance(x, (tuple, list)):
        return (x[0] * tcg_info['disp_info'][4], x[1] * tcg_info['disp_info'][4])
    return x * tcg_info['disp_info'][4]

def tcg_to_ref_image(cx, cy, ref_img, tcg_info, apply_disp_info=False, apply_ref_img_divide=False):
    """
    TCGから参照イメージの座標を得る

    cx, cy: TCG座標
    ref_img: 参照イメージ
    tcg_info: 回転情報
    apply_disp_info: disp_infoを適用するかどうか
    戻り値: 参照イメージ座標
    """
    cx, cy = denorm_param(tcg_info, (cx, cy))
    cx, cy = center_rotate(cx, cy, tcg_info)
    imax = max(tcg_info['original_img_size'][0] / 2, tcg_info['original_img_size'][1] / 2)
    cx, cy = cx + imax, cy + imax
    if apply_disp_info:
        disp_info = get_disp_info(tcg_info)
        if (   np.isclose(disp_info[2], disp_info[3])
            or not (    np.isclose(disp_info[2], tcg_info['original_img_size'][0])
                    and np.isclose(disp_info[3], tcg_info['original_img_size'][1]))
            or np.isclose(disp_info[4], 1.0)
#            or (ref_img.shape[1] == tcg_info['original_img_size'][0] and ref_img.shape[0] == tcg_info['original_img_size'][1])
        ):
            # Geometryモード時、クロップ時または拡大表示時とエクスポート時
            cx, cy = cx - disp_info[0], cy - disp_info[1]
            # クロップ時の表示空白
            if apply_ref_img_divide == False:
                cx = cx + (ref_img.shape[1] - disp_info[2]) / 2
                cy = cy + (ref_img.shape[0] - disp_info[3]) / 2
            else:
                # 縮小画像用（一旦大きくしてから計算）
                cx = cx + (ref_img.shape[1] / disp_info[4] - disp_info[2]) / 2
                cy = cy + (ref_img.shape[0] / disp_info[4] - disp_info[3]) / 2        
        cx, cy = cx * disp_info[4], cy * disp_info[4]
    return (cx, cy)

def ref_image_to_tcg(cx, cy, ref_img, tcg_info, apply_disp_info=False):
    """
    参照イメージの座標からTCGを得る

    cx, cy: 参照イメージ座標
    ref_img: 参照イメージ
    tcg_info: 回転情報
    apply_disp_info: disp_infoを適用するかどうか
    戻り値: TCG座標
    """
    if apply_disp_info:
        disp_info = get_disp_info(tcg_info)
        cx, cy = cx / disp_info[4], cy / disp_info[4]
        if (   np.isclose(disp_info[2], disp_info[3])
            or not (    np.isclose(disp_info[2], tcg_info['original_img_size'][0])
                    and np.isclose(disp_info[3], tcg_info['original_img_size'][1]))
            or np.isclose(disp_info[4], 1.0)
           ):
            # クロップ時の表示空白
            cx = cx - (ref_img.shape[1] / disp_info[4] - disp_info[2]) / 2
            cy = cy - (ref_img.shape[0] / disp_info[4] - disp_info[3]) / 2        
            # Geometryモード時、クロップ時または拡大表示時
            cx, cy = cx + disp_info[0], cy + disp_info[1]
    imax = max(tcg_info['original_img_size'][0] / 2, tcg_info['original_img_size'][1] / 2)
    cx, cy = cx - imax, cy - imax
    cx, cy = center_rotate_invert(cx, cy, tcg_info)
    cx, cy = norm_param(tcg_info, (cx, cy))
    return (cx, cy)
    

def apply_orientation(cx, cy, tcg_info):
    rad, flip = tcg_info['rotation2'], tcg_info['flip_mode']

    if (flip & 1) == 1:
        cx = -cx
    if (flip & 2) == 2:
        cy = -cy

    return cx, cy, rad

def center_rotate(cx, cy, tcg_info):
    cx, cy, rad = apply_orientation(cx, cy, tcg_info)
    rad = tcg_info['rotation'] + rad
    rad = -rad

    new_cx = cx * math.cos(rad) - cy * math.sin(rad)
    new_cy = cx * math.sin(rad) + cy * math.cos(rad)

    new_cx, new_cy = apply_matrix(new_cx, new_cy, tcg_info['matrix'])

    return (new_cx, new_cy)

def center_rotate_invert(cx, cy, tcg_info):
    rad = tcg_info['rotation'] + tcg_info['rotation2']
    rad = -rad

    cx, cy = apply_matrix_inverse(cx, cy, tcg_info['matrix'])

    new_cx = cx * math.cos(rad) + cy * math.sin(rad)
    new_cy = -cx * math.sin(rad) + cy * math.cos(rad)

    new_cx, new_cy, _ = apply_orientation(new_cx, new_cy, tcg_info)

    return (new_cx, new_cy)

def set_matrix(param, matrix):
    if matrix is None:
        param['matrix'] = np.eye(3)
    else:
        param['matrix'] = matrix.copy()

def add_matrix(param, matrix, offset=(0, 0)):
    if offset != (0, 0):
        # 座標系変換
        T = np.array([
            [1, 0, offset[0]],
            [0, 1, offset[1]],
            [0, 0, 1]
        ])
        T_inv = np.linalg.inv(T)
        matrix = T_inv @ matrix @ T
    
    if 'matrix' not in param:
        param['matrix'] = matrix.copy()
    else:
        param['matrix'] = np.dot(matrix, param['matrix'])

def apply_matrix(px, py, matrix):
    pt = np.array([px, py, 1.0])
    pt_transformed = matrix @ pt
    px_new = pt_transformed[0] / pt_transformed[2]
    py_new = pt_transformed[1] / pt_transformed[2]
    return (float(px_new), float(py_new))

def apply_matrix_inverse(px, py, matrix):
    H_inv = np.linalg.inv(matrix)
    return apply_matrix(px, py, H_inv)
