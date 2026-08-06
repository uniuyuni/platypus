import gc
import logging
import os
import resource
import subprocess
from typing import Any

import numpy as np

from utils.envutils import env_flag

try:
    import psutil
except Exception:
    psutil = None


def debug_enabled() -> bool:
    return env_flag("PLATYPUS_MEMORY_DEBUG")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip())
    except ValueError:
        return default


def current_rss_bytes() -> int | None:
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            # psutil側の取得失敗時はresource.getrusageへフォールバック
            pass
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if os.uname().sysname == "Darwin":
        return int(usage)
    return int(usage) * 1024


def available_memory_bytes() -> int | None:
    if psutil is None:
        if os.uname().sysname != "Darwin":
            return None
        try:
            output = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=1.0)
        except Exception:
            return None
        page_size = 16384
        first = output.splitlines()[0] if output else ""
        if "page size of" in first:
            try:
                page_size = int(first.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                page_size = 16384
        pages = 0
        for line in output.splitlines():
            name, _, value = line.partition(":")
            if name.strip() not in {"Pages free", "Pages inactive", "Pages speculative"}:
                continue
            try:
                pages += int(value.strip().rstrip(".").replace(".", ""))
            except ValueError:
                # vm_stat の出力形式が想定外の行は無視して集計を継続
                pass
        return pages * page_size if pages else None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _bytes_of(obj: Any, seen: set[int]) -> int:
    if obj is None:
        return 0
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_bytes_of(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(_bytes_of(v, seen) for v in obj)
    return 0


def bytes_of(obj: Any) -> int:
    return _bytes_of(obj, set())


def copy_image_for_cache(image):
    arr = np.asarray(image)
    contiguous = np.ascontiguousarray(arr)
    # ascontiguousarray が非連続入力に対して新規配列を作った場合はそれ自体が
    # 独立コピーなので、二重コピーを避けるため同一オブジェクトの場合のみ .copy() する。
    # 戻り値が呼び出し元の入力と絶対にエイリアスしない契約は維持する。
    if contiguous is arr:
        contiguous = contiguous.copy()
    return contiguous


def format_bytes(num: int | None) -> str:
    if num is None:
        return "unknown"
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def effect_cache_bytes(effects) -> int:
    total = 0
    seen: set[int] = set()
    for layer in effects or []:
        for effect in getattr(layer, "values", lambda: [])():
            total += _bytes_of(getattr(effect, "diff", None), seen)
            total += _bytes_of(getattr(effect, "_cached_predict", None), seen)
    return total


def processor_cache_bytes(processor) -> int:
    total = 0
    seen: set[int] = set()
    cache = getattr(processor, "cache", None)
    if not isinstance(cache, dict):
        return 0
    for entry in cache.values():
        if isinstance(entry, dict):
            total += _bytes_of(entry.get("result"), seen)
    return total


def clear_primary_param_ai_caches(primary_param) -> dict:
    removed = 0
    removed_bytes = 0
    if not isinstance(primary_param, dict):
        return {"primary_param_entries": removed, "primary_param_bytes": removed_bytes}

    for key in ("ai_noise_reduction_result",):
        value = primary_param.pop(key, None)
        if value is not None:
            removed += 1
            removed_bytes += bytes_of(value)
    primary_param.pop("ai_noise_reduction_content_key", None)
    return {"primary_param_entries": removed, "primary_param_bytes": removed_bytes}


def clear_mask2_ai_caches(mask_editor2) -> dict:
    clear = getattr(mask_editor2, "clear_ai_intermediate_caches", None)
    if clear is None:
        return {"mask2_entries": 0, "mask2_bytes": 0}
    try:
        result = clear()
    except Exception:
        logging.exception("memory_manager: failed to clear Mask2 AI cache")
        return {"mask2_entries": 0, "mask2_bytes": 0}
    if not isinstance(result, dict):
        return {"mask2_entries": 0, "mask2_bytes": 0}
    return {
        "mask2_entries": int(result.get("mask2_entries", 0) or 0),
        "mask2_bytes": int(result.get("mask2_bytes", 0) or 0),
    }


def _empty_ai_release_result() -> dict:
    # release_ai_model_runtimes の各失敗パスで返す 0 埋めフォールバック。
    # 呼び出し側で誤って書き換えられても影響が漏れないよう、共有定数ではなく
    # 呼び出しごとに新しい dict を返す。
    return {
        "sam3_processor_released": 0,
        "sam3_model_released": 0,
        "depth_model_released": 0,
        "face_runtime_released": 0,
    }


def release_ai_model_runtimes() -> dict:
    try:
        from cores.mask2 import inference_runtime
    except Exception:
        logging.exception("memory_manager: failed to import Mask2 inference runtime")
        return _empty_ai_release_result()
    release = getattr(inference_runtime, "release_ai_model_runtimes", None)
    if release is None:
        return _empty_ai_release_result()
    try:
        result = release()
    except Exception:
        logging.exception("memory_manager: failed to release AI model runtimes")
        return _empty_ai_release_result()
    if not isinstance(result, dict):
        return _empty_ai_release_result()
    return {
        "sam3_processor_released": int(result.get("sam3_processor_released", 0) or 0),
        "sam3_model_released": int(result.get("sam3_model_released", 0) or 0),
        "depth_model_released": int(result.get("depth_model_released", 0) or 0),
        "face_runtime_released": int(result.get("face_runtime_released", 0) or 0),
    }


def clear_effect_intermediate_caches(
    effects=None,
    processor=None,
    primary_param=None,
    mask_editor2=None,
    *,
    reason: str = "memory_pressure",
    clear_mask2_results: bool = False,
    release_ai_models: bool = True,
) -> dict:
    effect_count = 0
    for layer in effects or []:
        for effect in getattr(layer, "values", lambda: [])():
            try:
                effect.reeffect()
                effect_count += 1
            except Exception:
                try:
                    effect.diff = None
                    effect.hash = None
                    effect_count += 1
                except Exception:
                    logging.exception("memory_manager: failed to clear effect cache")
            if hasattr(effect, "_cached_predict"):
                effect._cached_predict = None
            if hasattr(effect, "_cached_predict_key"):
                effect._cached_predict_key = None

    processor_entries = 0
    if processor is not None:
        clear = getattr(processor, "clear_completed_cache", None)
        if clear is not None:
            try:
                processor_entries = int(clear())
            except Exception:
                logging.exception("memory_manager: failed to clear processor cache")

    primary_param_result = clear_primary_param_ai_caches(primary_param)
    if clear_mask2_results:
        mask2_result = clear_mask2_ai_caches(mask_editor2)
    else:
        mask2_result = {"mask2_entries": 0, "mask2_bytes": 0}
    ai_model_result = (
        release_ai_model_runtimes()
        if release_ai_models
        else {
            "sam3_processor_released": 0,
            "sam3_model_released": 0,
            "depth_model_released": 0,
            "face_runtime_released": 0,
        }
    )
    gc.collect()
    logging.info(
        "memory_manager cleared effect intermediates reason=%s effects=%d processor_entries=%d primary_param_entries=%d primary_param_bytes=%s mask2_entries=%d mask2_bytes=%s sam3_processor_released=%d sam3_model_released=%d depth_model_released=%d face_runtime_released=%d",
        reason,
        effect_count,
        processor_entries,
        primary_param_result["primary_param_entries"],
        format_bytes(primary_param_result["primary_param_bytes"]),
        mask2_result["mask2_entries"],
        format_bytes(mask2_result["mask2_bytes"]),
        ai_model_result["sam3_processor_released"],
        ai_model_result["sam3_model_released"],
        ai_model_result["depth_model_released"],
        ai_model_result["face_runtime_released"],
    )
    return {
        "effects": effect_count,
        "processor_entries": processor_entries,
        **primary_param_result,
        **mask2_result,
        **ai_model_result,
    }


def memory_pressure() -> tuple[bool, str]:
    available_min_mb = _env_float("PLATYPUS_MEMORY_AVAILABLE_MIN_MB", 1024.0)
    rss_limit_mb = _env_float("PLATYPUS_MEMORY_RSS_LIMIT_MB", 0.0)

    available = available_memory_bytes()
    if available is not None and available < available_min_mb * 1024 * 1024:
        return True, f"available<{available_min_mb:g}MB"

    rss = current_rss_bytes()
    if rss_limit_mb > 0 and rss is not None and rss > rss_limit_mb * 1024 * 1024:
        return True, f"rss>{rss_limit_mb:g}MB"

    return False, "ok"


def enforce_memory_policy(effects=None, processor=None, primary_param=None, mask_editor2=None, *, reason: str = "check") -> dict:
    pressured, pressure_reason = memory_pressure()
    if not pressured:
        return {"cleared": False, "reason": pressure_reason}
    cleared = clear_effect_intermediate_caches(
        effects,
        processor,
        primary_param,
        mask_editor2,
        reason=f"{reason}:{pressure_reason}",
    )
    cleared.update({"cleared": True, "reason": pressure_reason})
    return cleared


def build_memory_report(file_path=None, stage=None, cache_system=None, effects=None, processor=None, extra=None) -> dict:
    cache_bytes = 0
    final_display_cache_bytes = 0
    if cache_system is not None:
        cache_bytes = cache_system.cache_memory_bytes()
        final_display_cache_bytes = cache_system.final_display_cache_memory_bytes()
    report = {
        "file_path": file_path,
        "stage": str(stage) if stage is not None else None,
        "rss_bytes": current_rss_bytes(),
        "available_bytes": available_memory_bytes(),
        "fcs_cache_bytes": cache_bytes,
        "final_display_cache_bytes": final_display_cache_bytes,
        "effect_cache_bytes": effect_cache_bytes(effects),
        "processor_cache_bytes": processor_cache_bytes(processor),
    }
    if extra:
        report.update(extra)
    return report


def log_memory_report(label: str, report: dict, *, force: bool = False) -> None:
    if not force and not debug_enabled():
        return
    logging.debug(
        "[MEMORY] %s file=%s stage=%s rss=%s available=%s fcs_cache=%s final_display_cache=%s effect_cache=%s processor_cache=%s extra=%s",
        label,
        report.get("file_path"),
        report.get("stage"),
        format_bytes(report.get("rss_bytes")),
        format_bytes(report.get("available_bytes")),
        format_bytes(report.get("fcs_cache_bytes")),
        format_bytes(report.get("final_display_cache_bytes")),
        format_bytes(report.get("effect_cache_bytes")),
        format_bytes(report.get("processor_cache_bytes")),
        {
            k: v for k, v in report.items()
            if k not in {
                "file_path",
                "stage",
                "rss_bytes",
                "available_bytes",
                "fcs_cache_bytes",
                "final_display_cache_bytes",
                "effect_cache_bytes",
                "processor_cache_bytes",
            }
        },
    )
