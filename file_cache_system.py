
import threading
import time
from typing import Dict, Any
import os
import concurrent.futures
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
import logging
import sys
import multiprocessing
from collections import OrderedDict

import imageset
import config
import memory_manager
import utils.utils as utils
from utils import perf_trace
from enums import LoadStage, ImageFidelity


def _load_stall_warn_seconds():
    try:
        return max(0.0, float(os.getenv("PLATYPUS_LOAD_STALL_WARN_SECONDS", "15")))
    except ValueError:
        return 15.0


# warm-up用のダミー関数（pickle化可能なトップレベル関数）
def _warmup_worker():
    """ProcessPoolExecutor warm-up用の空関数"""
    pass


def _load_pool_context():
    start_method = os.getenv("PLATYPUS_LOAD_POOL_START_METHOD", "").strip().lower()
    if not start_method and getattr(sys, "frozen", False) and sys.platform == "darwin":
        # PyInstaller/macOS の frozen app では spawn 子プロセスがアプリ本体を再初期化して
        # 落ちることがあるため、重いRAWロードだけ fork でUIプロセスから隔離する。
        start_method = "fork"
    if not start_method:
        return None
    try:
        return multiprocessing.get_context(start_method)
    except ValueError:
        logging.warning("FCS: invalid PLATYPUS_LOAD_POOL_START_METHOD=%r", start_method)
        return None

# メインプロセスで実行されるコールバック関数
def _task_callback(file_callbacks, shared_resources, future):
    try:
        # タスクの結果を取得
        if isinstance(future, Future):
            # サブプロセス実行なら共有メモリから取得
            file_path, shm, exif_data, param, stage = future.result()
            imgset = imageset.shared_memory_to_imageset(*shm)
        else:
            # メインプロセス実行ならメモリから取得
            file_path, imgset, exif_data, param, stage = future
            # 一部ワーカーは共有メモリタプルを返す (file_path, shm_name, shape, dtype, fidelity)
            # その場合は ImageSet に復元する
            try:
                if isinstance(imgset, tuple) and len(imgset) >= 4:
                    # imageset.shared_memory_to_imageset は (file_path, shm_name, shape, dtype, ...)
                    imgset = imageset.shared_memory_to_imageset(*imgset)
            except Exception:
                # 復元失敗しても進めてエラーを記録する
                logging.exception("FCS: failed to restore ImageSet from shared-memory tuple")

        # Memmap化 (キャッシュ投入前)
        # RAW プレビュー段階は短命（直後にフルデコードで置換される）なので
        # memmap 化に伴う temp ファイル書き出しコストを払う価値がなくスキップする。
        # しきい値はディスク I/O コストとメモリ常駐コストのトレードオフから 32MB。
        # 24MP データを多数キャッシュする実運用を想定（max_cache_size=100、float32 換算で
        # 1 枚 288MB のため、閾値を高くすると数 GB 級の RAM を消費し得る）。
        if (
            imgset is not None
            and imgset.img is not None
            and getattr(imgset, "fidelity", None) != ImageFidelity.PREVIEW
            and imgset.img.nbytes > 32 * 1024 * 1024
        ):
            perf_trace.event("fcs.memmap_begin", nbytes=int(imgset.img.nbytes))
            mm, backing = utils.array_to_memmap(imgset.img)
            imgset.img = mm
            imgset.backing = backing
            logging.info(f"FCS: Converted {file_path} to memmap. Backing: {backing}")
            perf_trace.event("fcs.memmap_done")

        # キャッシュに追加
        # 既存のキャッシュがある場合はHistoryを維持する
        # このコールバックはThreadPoolExecutorのワーカースレッドから呼ばれるため、
        # 既存Historyの読み取り→上書きを cache_lock でアトミックにする
        # （delete_cache/clear_cache 等の別スレッド操作との競合を防ぐ）
        current_history = None
        with shared_resources['cache_lock']:
            if file_path in shared_resources['cache']:
                try:
                    # 既存のキャッシュからHistoryを取得
                    # 形式: (imgset, exif_data, param, history)
                    current_history = shared_resources['cache'][file_path][3]
                except Exception:
                    # Historyを引き継げないと当該ファイルの undo/redo 履歴が失われるため記録する
                    logging.exception("FCS: failed to carry over existing history for %s", file_path)

            shared_resources['cache'][file_path] = (imgset, exif_data, param.copy(), current_history)

        # コールバックを実行
        # 注意: このコールバックは executor のワーカースレッド上で直接呼び出される。
        # UI更新が必要な場合、呼び出し側（main.py）で @kvmainthread 等により
        # メインスレッドへマーシャリングする責任を持つ（file_cache_system は Kivy 非依存）。
        # ロックは保持しない: コールバックは重い処理を行いうるため critical section の外で実行する。
        callback = file_callbacks.get(file_path, None)
        if callback:
            callback(file_path, imgset, exif_data, param, current_history, stage)
            logging.info(f"FCS Callback executed for {file_path}, stage={stage}")

    except Exception as e:
        logging.error(f"FCS: {str(e)}")

# インスタンスメソッド用ラッパー
def run_method(obj, method_name, copy_config, *args, **kwargs):
    if copy_config:
        config._config = copy_config

    return getattr(obj, method_name)(*args, **kwargs)

def _notify_load_failed(
    file_path: str,
    file_callbacks: dict,
    exif_data,
    param,
    reason: str,
):
    """preload 失敗・未対応形式・例外時に UI の loading を解除するためコールバックを必ず呼ぶ。"""
    logging.error(f"FCS load failed ({file_path}): {reason}")
    callback = file_callbacks.get(file_path)
    if not callback:
        return
    imgset = imageset.ImageSet()
    imgset.file_path = file_path
    imgset.img = None
    exif_data = exif_data if exif_data is not None else {}
    param = param if param is not None else {}
    try:
        callback(file_path, imgset, exif_data, param, None, LoadStage.FULL_DECODE)
    except Exception as e:
        logging.error(f"FCS failure callback error: {e}")


# ヘルパー関数（スレッド間で共有）
def _load_file_thread(shared_resources, file_path, exif_data, param, imgset, file_callbacks):
    """
    ファイル読み込みスレッド
    
    Args:
        shared_resources: 共有リソース辞書
        file_path: ファイル名
        exif_data: EXIFデータ
        param: 追加パラメータ
    """
    # 共有リソースを解凍
    cache = shared_resources['cache']
    preload_registry = shared_resources['preload_registry']
    active_processes = shared_resources['active_processes']
    
    logging.info(f"Loading thread started for {file_path}")
    if imgset is None:
        imgset = imageset.ImageSet()

    try:
        # ファイル読み込み準備？
        result = imgset.preload(file_path, exif_data, param)
        if result is None:
            _notify_load_failed(
                file_path,
                file_callbacks,
                exif_data,
                param,
                "unsupported extension or preload returned None",
            )
        elif result is not None:
            # まずプレビューを単独で作って UI に渡す。ThreadPool 時にフル RAW デコードを
            # 先に投げると同一プロセス内で競合し、プレビュー表示自体が数秒遅れる。
            executor = shared_resources['executor']
            tasks = result
            first_result = run_method(imgset, tasks[0].worker, config._config, None, file_path, exif_data, param)
            _task_callback(file_callbacks, shared_resources, first_result)

            # 続きの読み込みがある
            futures = []
            for task in tasks[1:]:
                try:
                    task_imgset = type(imgset)()
                except Exception:
                    task_imgset = imageset.ImageSet()
                task_imgset.file_path = file_path
                future = executor.submit(
                    run_method,
                    task_imgset,
                    task.worker,
                    config._config,
                    None,
                    file_path,
                    exif_data.copy() if isinstance(exif_data, dict) else exif_data,
                    param.copy() if isinstance(param, dict) else param,
                )
                futures.append(future)
            
            # 完了待ち。RAW フルデコードなどが詰まった時に無音で待ち続けると
            # UI 側では「ロード中で固まった」ようにしか見えないため、周期的に状況を出す。
            if futures:
                warn_seconds = _load_stall_warn_seconds()
                if warn_seconds <= 0:
                    concurrent.futures.wait(futures)
                else:
                    pending = set(futures)
                    while pending:
                        _, pending = concurrent.futures.wait(
                            pending,
                            timeout=warn_seconds,
                            return_when=concurrent.futures.ALL_COMPLETED,
                        )
                        if pending:
                            started_at = active_processes.get(file_path, time.time())
                            logging.warning(
                                "FCS load still waiting: file=%s elapsed=%.1fs pending_subtasks=%d "
                                "active=%d preload=%d cache=%d",
                                file_path,
                                time.time() - started_at,
                                len(pending),
                                len(active_processes),
                                len(preload_registry),
                                len(cache),
                            )
                for future in futures:
                    _task_callback(file_callbacks, shared_resources, future)

            # キャッシュに登録（すでにキャンセルされていたらスキップ）
            #if file_path in active_processes:
            #    cache[file_path] = (imgset, exif_data, param.copy())
        
        # 先行読み込み登録から削除
        if file_path in preload_registry:
            del preload_registry[file_path]

        # スレッド終了
        elapsed_time = time.time() - active_processes[file_path]
        logging.info(f"FCS Finish loading {file_path} 経過時間 {elapsed_time:.3f} 秒.")

        # 進行中のスレッドから削除
        if file_path in active_processes:
            del active_processes[file_path]

    except Exception as e:
        logging.error(f"FCS Error preloading {file_path}: {e}")
        _notify_load_failed(
            file_path,
            file_callbacks,
            exif_data,
            param,
            str(e),
        )
    finally:
        if file_path in preload_registry:
            del preload_registry[file_path]
        if file_path in active_processes:
            del active_processes[file_path]
        # 処理キューフラグを設定
        shared_resources['process_queue_flag'] = True

class FileCacheSystem:
    def __init__(self, max_cache_size: int = 10, max_concurrent_loads: int = 4):
        # 共有リソースを初期化
        # 重いRAWフルデコードはUIプロセスから隔離する。frozenでも main.py の
        # freeze_support() によりProcessPool子プロセスとして起動できる。
        force_thread_pool = os.getenv("PLATYPUS_FORCE_THREAD_LOAD_POOL", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if force_thread_pool:
            self.ppe = ThreadPoolExecutor(max_workers=2)
            self._use_process_pool = False
            logging.info("FCS load executor: ThreadPoolExecutor forced by PLATYPUS_FORCE_THREAD_LOAD_POOL")
        else:
            mp_context = _load_pool_context()
            ppe_kwargs = {"mp_context": mp_context} if mp_context is not None else {}
            self.ppe = ProcessPoolExecutor(max_workers=2, **ppe_kwargs)
            self._use_process_pool = True
            if mp_context is not None:
                start_method = mp_context.get_start_method()
            else:
                start_method = multiprocessing.get_start_method(allow_none=True) or "default"
            logging.info(
                "FCS load executor: ProcessPoolExecutor start_method=%s frozen=%s",
                start_method,
                getattr(sys, "frozen", False),
            )

        self.shared_resources = {
            'cache': {},
            'preload_registry': {},
            'active_processes': {},
            'process_queue_flag': False,
            'executor': self.ppe,
            # 'cache' 辞書はThreadPoolExecutorの完了コールバックスレッドと
            # メインスレッドの双方からread-modify-writeされるため専用ロックで保護する
            'cache_lock': threading.Lock()
        }
        self.final_display_cache = OrderedDict()
        # ダミーを走らせる（プロセスのwarm-up）
        # アプリケーション起動時にワーカープロセスを起動しておくことで、
        # 最初の画像読み込み時のプロセス起動コストを回避
        if self._use_process_pool:
            for _ in range(self.ppe._max_workers):
                self.ppe.submit(_warmup_worker)
        
        # 各共有リソースへの参照を設定
        self.cache = self.shared_resources['cache']
        self.cache_lock = self.shared_resources['cache_lock']
        self.preload_registry = self.shared_resources['preload_registry']
        self.active_processes = self.shared_resources['active_processes']
        self.file_callbacks = {}  # コールバックはpickle化できないので共有しない
        
        # その他の設定
        self.max_cache_size = max_cache_size
        self.max_concurrent_loads = max_concurrent_loads
        try:
            self.max_final_display_cache = int(os.getenv("PLATYPUS_FINAL_DISPLAY_CACHE_MAX", "8"))
        except ValueError:
            self.max_final_display_cache = 8
        
        # 監視スレッドの開始
        #self.monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        #self.monitor_thread.start()

        # ThreadPool
        self.p = ThreadPoolExecutor(max_workers=max_concurrent_loads)
     
     
    def get_file(self, file_path: str, callback=None):
        """
        ファイルを取得する関数
        
        Args:
            file_path: ファイル名
            callback: ファイルが読み込まれた際に呼び出すコールバック関数
            
        Returns:
            Tuple[Dict[str, Any], Optional[Imgset]]: (exif_data, imgset)のタプル
            
        Raises:
            FileNotFoundError: キャッシュにも先行読み込み登録もされていない場合
        """
        result = (None, None)

        # 先行読み込み登録がある場合
        if file_path in self.preload_registry:
            # コールバックが指定された場合、登録
            if callback:
                self.file_callbacks.clear()     # 登録できるのは一つだけ
                self.file_callbacks[file_path] = callback

            logging.info(f"FCS Preload registry hit: {file_path}")
            exif_data, param, imgset, history = self.preload_registry[file_path]
            
            # まだ読み込みスレッドが開始されていなければ開始
            if file_path not in self.active_processes:
                self._start_loading_thread(file_path, exif_data, param, imgset)

            result = (exif_data, imgset)  # imgsetはまだ利用不可

        # キャッシュにある場合
        # 存在チェックと取得を1回のロックでアトミックに行う（check-then-actレース回避）
        with self.cache_lock:
            cache_entry = self.cache.get(file_path)
        if cache_entry is not None:
            logging.info(f"FCS Cache hit: {file_path}")
            imgset, exif_data, param, history = cache_entry

            # コールバックが指定されていればすぐに呼び出す
            if callback:
                if file_path not in self.file_callbacks:
                    self.file_callbacks.clear() # 他のファイルなら消す
                
                callback(file_path, imgset, exif_data, param.copy(), history, LoadStage.FIRST_PAINTABLE)

                # フル解像がキャッシュに乗っているときは、旧 -1 相当の再コールバック（重い pmck マージ用）
                if getattr(imgset, 'fidelity', None) == ImageFidelity.FULL:
                    callback(file_path, imgset, exif_data, param.copy(), history, LoadStage.FULL_DECODE)
                
            result = (exif_data, imgset)
                        
        # どちらにもない場合
        if result == (None, None):
            raise FileNotFoundError(f"File {file_path} is not in cache or preload registry")

        return result
    
    def register_for_preload(self, file_path: str, exif_data: Dict[str, Any], param: Dict[str, Any] = None, 
                        high_priority: bool = False):
        """
        先行読み込み登録関数
        
        Args:
            file_path: ファイル名
            exif_data: EXIFデータ
            param: 追加パラメータ
            high_priority: 優先度が高いかどうか
        """
        if param is None:
            param = {}
            
        # すでにキャッシュにある場合は何もしない
        with self.cache_lock:
            already_cached = file_path in self.cache
        if already_cached:
            return
        
        # すでに先行読み込み登録されている場合は何もしない
        if file_path in self.preload_registry:
            return
        
        # 高速化のためここで作っとく
        imgset = imageset.ImageSet()
        imgset.file_path = file_path
        imgset.param = param
        
        # 先行読み込み登録
        self.preload_registry[file_path] = (exif_data, param, imgset, None)
        logging.info(f"Registered {file_path} for preload")
        
        # 優先度が高い場合はすぐに読み込みを開始
        if high_priority:
            self._start_loading_thread(file_path, exif_data, param, imgset)
        else:
            # 優先度が低い場合は自動的にキューを処理
            self.process_preload_queue(max_concurrent_loads=self.max_concurrent_loads)

    def set_history(self, file_path, history):
        # 読み取り→上書きをアトミックに（_task_callback 等との競合防止）
        with self.cache_lock:
            entry = self.cache.get(file_path)
            if entry is not None:
                imgset, exif_data, param, _ = entry
                self.cache[file_path] = (imgset, exif_data, param, history)

    def _entry_memory_bytes(self, entry):
        try:
            imgset, exif_data, param, history = entry
        except Exception:
            return 0
        total = 0
        total += memory_manager.bytes_of(getattr(imgset, "img", None))
        total += memory_manager.bytes_of(param)
        return total

    def cache_memory_bytes(self):
        total = 0
        # values() のイテレーション中に別スレッドが辞書を変更すると
        # RuntimeError になり得るため、スナップショットを取ってからロック外で集計する
        with self.cache_lock:
            cache_entries = list(self.cache.values())
        for entry in cache_entries:
            total += self._entry_memory_bytes(entry)
        for entry in self.preload_registry.values():
            total += self._entry_memory_bytes((entry[2], entry[0], entry[1], entry[3]))
        total += self.final_display_cache_memory_bytes()
        return total

    def final_display_cache_memory_bytes(self):
        total = 0
        for entry in self.final_display_cache.values():
            total += memory_manager.bytes_of(entry.get("image"))
        return total

    def remember_final_display_image(self, file_path, image, *, stage=None, frame_version=None):
        if not file_path or image is None or self.max_final_display_cache <= 0:
            return False
        try:
            cached_image = memory_manager.copy_image_for_cache(image)
        except Exception:
            logging.exception("FCS memory: failed to cache final display image for %s", file_path)
            return False
        self.final_display_cache[file_path] = {
            "image": cached_image,
            "stage": stage,
            "frame_version": frame_version,
            "created_at": time.time(),
        }
        self.final_display_cache.move_to_end(file_path)
        while len(self.final_display_cache) > self.max_final_display_cache:
            evicted, _ = self.final_display_cache.popitem(last=False)
            logging.info("FCS memory: evicted final display cache by count %s", evicted)
        return True

    def get_final_display_image(self, file_path):
        entry = self.final_display_cache.get(file_path)
        if entry is None:
            return None
        self.final_display_cache.move_to_end(file_path)
        return entry.get("image")

    def clear_final_display_cache(self, keep_file_path=None):
        removed = 0
        for file_path in list(self.final_display_cache.keys()):
            if keep_file_path is not None and file_path == keep_file_path:
                continue
            del self.final_display_cache[file_path]
            removed += 1
        return removed

    def evict_final_display_cache_for_memory(self, keep_file_path=None):
        removed = 0
        for file_path in list(self.final_display_cache.keys()):
            if keep_file_path is not None and file_path == keep_file_path and len(self.final_display_cache) > 1:
                continue
            del self.final_display_cache[file_path]
            removed += 1
            pressured, _ = memory_manager.memory_pressure()
            if not pressured:
                break
        if removed:
            logging.info("FCS memory: evicted final display caches for memory pressure count=%d", removed)
        return removed

    def release_pmck_payload(self, owner=None, *, reason="image_selection_changed"):
        if owner is not None and hasattr(owner, "_last_pmck_dict"):
            owner._last_pmck_dict = None
        logging.info("FCS memory: released pmck payload reason=%s", reason)

    def on_image_selection_changed(self, owner=None, previous_file_path=None, current_file_path=None):
        if previous_file_path != current_file_path:
            self.release_pmck_payload(owner, reason="image_selection_changed")
        self.enforce_memory_policy(owner=owner, reason="image_selection_changed")

    def enforce_memory_policy(self, owner=None, *, reason="check"):
        effects = getattr(owner, "primary_effects", None) if owner is not None else None
        processor = getattr(owner, "processor", None) if owner is not None else None
        primary_param = getattr(owner, "primary_param", None) if owner is not None else None
        mask_editor2 = None
        ids = getattr(owner, "ids", None) if owner is not None else None
        if ids is not None:
            try:
                mask_editor2 = ids.get("mask_editor2")
            except Exception:
                try:
                    mask_editor2 = ids["mask_editor2"]
                except Exception:
                    mask_editor2 = None
        result = memory_manager.enforce_memory_policy(effects, processor, primary_param, mask_editor2, reason=reason)
        if result.get("cleared"):
            pressured, pressure_reason = memory_manager.memory_pressure()
            if pressured:
                keep_file_path = None
                imgset = getattr(owner, "imgset", None) if owner is not None else None
                if imgset is not None:
                    keep_file_path = getattr(imgset, "file_path", None)
                result["final_display_evicted"] = self.evict_final_display_cache_for_memory(
                    keep_file_path=keep_file_path,
                )
                result["final_display_reason"] = pressure_reason
        return result

    def log_display_ready_memory(self, owner=None, *, file_path=None, stage=None, extra=None):
        effects = getattr(owner, "primary_effects", None) if owner is not None else None
        processor = getattr(owner, "processor", None) if owner is not None else None
        report = memory_manager.build_memory_report(
            file_path=file_path,
            stage=stage,
            cache_system=self,
            effects=effects,
            processor=processor,
            extra=extra,
        )
        memory_manager.log_memory_report("display_ready", report)
        return report

    def _start_loading_thread(self, file_path: str, exif_data: Dict[str, Any], param: Dict[str, Any] = None, imgset=None):
        """読み込みスレッドを開始する内部関数"""
        if param is None:
            param = {}
        
        if file_path in self.active_processes:
            return
        
        self.active_processes[file_path] = time.time()  # プロセスIDのみ保存

        # プロセスを起動（self自体は渡さない）
        if False:
            """
            process = multiprocessing.Process(
                target=_load_file_process,
                args=(self.shared_resources, file_path, exif_data, param)
            )
            process.start()
            """
            thread = threading.Thread(target=_load_file_thread, args=[self.shared_resources, file_path, exif_data, param, imgset], daemon=True)
            thread.start()
        else:
            future = self.p.submit(_load_file_thread, self.shared_resources, file_path, exif_data, param, imgset, self.file_callbacks)

        logging.info(f"Started loading process for {file_path}")

    def delete_cache(self, dict, file_path):
        """
        キャッシュからファイルを削除する関数
        """
        if file_path in self.file_callbacks:
            del self.file_callbacks[file_path]

        if file_path in self.active_processes:
            del self.active_processes[file_path]

        if file_path in self.preload_registry:
            del self.preload_registry[file_path]

        # 存在チェックと削除をアトミックに（_task_callback 等との競合防止）
        with self.cache_lock:
            if file_path in self.cache:
                del self.cache[file_path]


    def delete_file(self, file_path):
        """
        キャッシュからファイルを削除する関数

        Args:
            file_path: ファイル名
        """
        self.delete_cache(self.cache, file_path)
        self.delete_cache(self.preload_registry, file_path)
        self.final_display_cache.pop(file_path, None)
            
    def clear_cache(self, keep_files=None):
        """
        キャッシュをクリアする関数
        
        Args:
            keep_files: キャッシュに残すファイル名のリスト
        """
        if keep_files is None:
            keep_files = []
        
        # キャッシュからkeep_files以外の全てのアイテムを削除
        # キー一覧はロック内でスナップショットし、削除自体は delete_cache 側でロックする
        with self.cache_lock:
            keys_snapshot = list(self.cache.keys())
        for file_path in keys_snapshot:
            if file_path not in keep_files:
                self.delete_cache(self.cache, file_path)
    
    def get_cache_status(self):
        """
        キャッシュの状態を取得する関数
        
        Returns:
            Dict: キャッシュの状態
        """
        with self.cache_lock:
            cache_size = len(self.cache)
            cached_files = list(self.cache.keys())
        return {
            "cache_size": cache_size,
            "preload_registry_size": len(self.preload_registry),
            "active_processes": len(self.active_processes),
            "max_cache_size": self.max_cache_size,
            "cached_files": cached_files,
            "preload_registered_files": list(self.preload_registry.keys()),
            "active_process_files": list(self.active_processes.keys())
        }
    
    def process_preload_queue(self, max_concurrent_loads=None):
        """
        先行読み込み登録されているファイルの読み込みプロセスを開始する関数
        
        Args:
            max_concurrent_loads: 同時に実行する最大読み込みプロセス数（Noneの場合は制限なし）
        """
        # 現在の進行中プロセス数
        current_processes = len(self.active_processes)
        
        # 同時実行数の制限
        if max_concurrent_loads is not None and current_processes >= max_concurrent_loads:
            #logging.info(f"Maximum concurrent loads ({max_concurrent_loads}) reached. Waiting for processes to complete.")
            return
        
        # 利用可能なスロット数を計算
        available_slots = float('inf') if max_concurrent_loads is None else max_concurrent_loads - current_processes
        
        # キャッシュの空き容量を確認
        with self.cache_lock:
            cache_len = len(self.cache)
        available_cache_slots = self.max_cache_size - (cache_len + current_processes)

        # いっぱいなら古いものから削除
        while available_cache_slots <= 0:
            with self.cache_lock:
                file_to_delete = list(self.cache)[:1]
            if not file_to_delete:
                break
            file_to_delete = file_to_delete[0]
            self.delete_cache(self.cache, file_to_delete)
            available_cache_slots += 1

        # 実際に開始できるプロセス数（キャッシュ容量と同時実行数の少ない方）
        processes_to_start = min(available_slots, available_cache_slots, len(self.preload_registry))
        
        if processes_to_start <= 0:
            # print("No available slots for new loading processes")
            return
        
        # 先行読み込み登録から指定数のファイルを取得して読み込みを開始
        _keys = list(self.preload_registry.keys())
        files_to_load = _keys[: int(processes_to_start)]
        
        for file_path in files_to_load:
            if file_path not in self.active_processes:  # 既に進行中でないことを確認
                exif_data, param, imgset, _ = self.preload_registry[file_path]
                self._start_loading_thread(file_path, exif_data, param, imgset)
                logging.info(f"FCS Starting loading processes for {file_path}")
        
    def shutdown(self):
        """
        システムをシャットダウンする関数
        """
        # 全ての進行中のプロセスを終了
        #for process in self.active_processes.values():
        #    process.terminate()
        
        self.p.shutdown()
        self.ppe.shutdown()
        
        logging.info("FCS shutdown complete")
