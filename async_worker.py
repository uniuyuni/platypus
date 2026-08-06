
import multiprocessing
from multiprocessing import Process, Queue, Event, shared_memory
import numpy as np
import time
from queue import Empty
import logging
import traceback
import sys
import copy
import pickle
import threading

import effects
import config
import waitinfo

EFFECT_TIMEOUT_SECONDS = {
    "InpaintEffect": 600.0,
}

_AI_NOISE_WORKER_ONLY_KEYS = (
    "ai_noise_reduction_result",
    "ai_noise_reduction_content_key",
    "_ai_noise_reduction_result_deferred",
)


def _task_params_for_worker(effect_name, params):
    if effect_name == "AINoiseReductonEffect" or not isinstance(params, dict):
        return params

    worker_params = params.copy()
    for key in _AI_NOISE_WORKER_ONLY_KEYS:
        worker_params.pop(key, None)
    return worker_params


def _worker_param_summary(params):
    if not isinstance(params, dict):
        return f"type={type(params).__name__}"
    heavy_keys = []
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            heavy_keys.append(f"{key}:{value.shape}/{value.dtype}/{value.nbytes}")
    return f"keys={len(params)} heavy=[{', '.join(heavy_keys[:6])}]"


def _worker_result_image(effect_name, params, target_effect, input_image, diff):
    if effect_name == "AINoiseReductonEffect":
        raw_nr = params.get("ai_noise_reduction_result")
        if raw_nr is not None and isinstance(raw_nr, np.ndarray):
            return np.ascontiguousarray(raw_nr, dtype=np.float32)
    if diff is not None:
        return target_effect.apply_diff(input_image)
    return input_image


def worker_process(input_queue, result_queue, msg_queue, stop_event, config_dict, latest_tasks):    
    """
    Background worker process.
    Continuously pulls tasks from input_queue and processes them.
    """
    # 子プロセスで読み込まれているモジュールを確認
    loaded_modules = list(sys.modules.keys())
    print(f"子プロセス {multiprocessing.current_process().name} で読み込まれているモジュール:")
    
    # 特定のモジュールをチェック
    check_modules = ['matplotlib', 'kivy']
    for module in check_modules:
        if any(module in m for m in loaded_modules):
            print(f"  ⚠️ {module} が読み込まれています")
            # 具体的にどのサブモジュールか表示
            related = [m for m in loaded_modules if module in m]
            print(f"     {related[:5]}")  # 最初の5つを表示

    # Restore configuration in the worker process
    config._config = config_dict

    # Configure logging for the worker process
    logging.basicConfig(level=logging.INFO, format='[%(levelname)-7s] %(message)s')

    # Initialize waitinfo with IPC queue
    waitinfo.init(msg_queue)
    
    # Create independent effect instances for the worker
    worker_effects = effects.create_effects()
    
    while not stop_event.is_set():
        try:
            # Get task with timeout to allow checking stop_event
            task = input_queue.get(timeout=0.1)
            
            if task is None:
                continue
                
            task_id, effect_name, shm_name, shape, dtype_str, params, efconfig = task
            task_started_at = time.monotonic()
            logging.info(
                "AsyncWorker task received: task_id=%s effect=%s shape=%s dtype=%s params=%s",
                task_id,
                effect_name,
                shape,
                dtype_str,
                _worker_param_summary(params),
            )
            
            # Check if this task is already cancelled
            if effect_name in latest_tasks:
                if task_id < latest_tasks[effect_name]:
                    # Cancelled
                    # Need to cleanup input SHM? 
                    # Main process cleans up when it receives result. 
                    # If we silently drop, Main process waits forever? 
                    # No, Main tracks SHM. We should probably signal "Cancelled" 
                    # or Main should rely on "Latest Task ID" logic too?
                    # Main polls. If Main sees new task submitted, it knows old one is dead.
                    # But SHM cleanup needs to happen.
                    
                    # Simplest: Send "cancelled" result so Main can cleanup SHM
                    logging.info(
                        "AsyncWorker task cancelled before run: task_id=%s effect=%s latest=%s",
                        task_id,
                        effect_name,
                        latest_tasks.get(effect_name),
                    )
                    result_queue.put({'task_id': task_id, 'status': 'cancelled'})
                    
                    # Close input SHM
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        shm.close()
                    except (FileNotFoundError, OSError):
                        # Main側が既に解放済みならそれで良い（ベストエフォートのクリーンアップ）
                        pass
                    continue
            
            existing_shm = None
            result_shm = None
            result_shm_queued = False
            try:
                # Access shared memory
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                # Create numpy array from shared memory (no copy yet)
                input_image = np.ndarray(shape, dtype=dtype_str, buffer=existing_shm.buf)
                
                # We need a copy to process because we shouldn't modify the source in place 
                # if it might be used by others, but here strictly 1-to-1.
                # However, many effects return a new array.
                
                # Find the effect instance
                # Currently we assume effects are in the first group (lv0~lv4 flattened? No, effects structure is complex)
                # We need a way to locate the effect. 
                # For simplicity, let's assume we pass enough info to reconstruct or find it.
                # Actually, `effects.create_effects()` returns a structure.
                # We need to find the specific effect by name.
                
                target_effect = None
                for layer in worker_effects:
                    # Search by key match first (fast)
                    if effect_name in layer:
                         target_effect = layer[effect_name]
                         break
                    # Fallback: Search by Class Name
                    for inst in layer.values():
                        if inst.__class__.__name__ == effect_name:
                            target_effect = inst
                            break
                    if target_effect:
                        break
                
                if target_effect:
                    # Sync parameters
                    # We assume params is a dictionary of parameters for the effect
                    # But wait, `make_diff` takes `param` (global param) and `efconfig`.
                    # The `params` passed in task should be the global param dictionary.
                    
                    # Execute heavy processing
                    # Note: We are not calling make_diff/apply_diff logic directly because we want to force the heavy logic.
                    # But usually `make_diff` IS the heavy logic.
                    
                    # Update effect internal state if needed (some effects might need set2param logic equivalent?)
                    # Generally parameters are passed to make_diff.
                    
                    diff = target_effect.make_diff(input_image, params, efconfig)
                    result_image = _worker_result_image(
                        effect_name,
                        params,
                        target_effect,
                        input_image,
                        diff,
                    )
                    
                    # Write result to NEW shared memory
                    # (We cannot reuse input SHM as it might be read by others or main process?)
                    # Actually, for efficiency, can we? 
                    # Use a new SHM for the result to avoid race conditions.
                    
                    result_shm = shared_memory.SharedMemory(create=True, size=result_image.nbytes)
                    result_array = np.ndarray(result_image.shape, dtype=result_image.dtype, buffer=result_shm.buf)
                    result_array[:] = result_image[:]
                    
                    # Send result back
                    result_queue.put({
                        'task_id': task_id,
                        'status': 'success',
                        'shm_name': result_shm.name,
                        'shape': result_image.shape,
                        'dtype': str(result_image.dtype)
                    })
                    # Ownership of result_shm now transfers to the main process (it
                    # opens the SHM by name and unlinks it in poll_results). Only mark
                    # it as queued AFTER put() succeeds, so a failed put still falls
                    # through to our own close+unlink in the finally block below.
                    result_shm_queued = True
                    logging.info(
                        "AsyncWorker task success: task_id=%s effect=%s elapsed=%.3fs result_shape=%s result_dtype=%s",
                        task_id,
                        effect_name,
                        time.monotonic() - task_started_at,
                        result_image.shape,
                        result_image.dtype,
                    )

                else:
                    logging.error(f"Worker: Effect {effect_name} not found.")
                    result_queue.put({'task_id': task_id, 'status': 'error', 'message': 'Effect not found'})

            except Exception as e:
                logging.error(
                    "Worker processing error: task_id=%s effect=%s elapsed=%.3fs error=%s",
                    task_id,
                    effect_name,
                    time.monotonic() - task_started_at,
                    e,
                )
                traceback.print_exc()
                result_queue.put({'task_id': task_id, 'status': 'error', 'message': str(e)})

            finally:
                # Always release the input SHM handle, whether processing succeeded,
                # failed, or the effect wasn't found.
                if existing_shm is not None:
                    existing_shm.close()
                # result_shm is only still open/unlinked-pending here if something
                # raised between its creation and the 'success' message reaching
                # result_queue (make_diff, _worker_result_image, or result_queue.put
                # itself). In that case the main process never learns its name, so we
                # still own the segment and must unlink it ourselves or it leaks
                # permanently. If the 'success' message was queued, ownership passed
                # to the main process, which unlinks it itself in poll_results.
                if result_shm is not None:
                    result_shm.close()
                    if not result_shm_queued:
                        try:
                            result_shm.unlink()
                        except Exception as unlink_err:
                            logging.warning(
                                "AsyncWorker failed to unlink orphaned result SHM: task_id=%s error=%s",
                                task_id,
                                unlink_err,
                            )

        except Empty:
            continue
        except Exception as e:
            logging.error(f"Worker loop error: {e}")

class AsyncWorker:
    def __init__(self):
        self.process = None
        self.thread_mode = getattr(sys, "frozen", False)
        self.input_queue = Queue()
        self.result_queue = Queue()
        self.msg_queue = Queue()
        self.stop_event = Event()
        self.active_shms = set() # Track SHMs to unlink them if needed
        self.active_effects = {} # task_id -> effect_name
        self.active_started_at = {} # task_id -> monotonic start time
        self.task_counter = 0
        if self.thread_mode:
            self.latest_tasks = {}
        else:
            self.latest_tasks = multiprocessing.Manager().dict() # effect_name -> task_id

    def _cleanup_active_task(self, task_id):
        self.active_effects.pop(task_id, None)
        self.active_started_at.pop(task_id, None)

        to_remove = None
        for tid, shm in self.active_shms:
            if tid == task_id:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logging.warning("AsyncWorker failed to cleanup input SHM: task_id=%s error=%s", task_id, e)
                to_remove = (tid, shm)
                break
        if to_remove:
            self.active_shms.remove(to_remove)

    def reap_dead_worker(self):
        process = getattr(self, "process", None)
        if getattr(self, "thread_mode", False) or process is None or process.is_alive():
            return False
        if not self.active_effects and not self.active_shms:
            return False

        pending = dict(self.active_effects)
        logging.error(
            "AsyncWorker process exited with pending tasks: exitcode=%s pending=%s active_shms=%s",
            getattr(process, "exitcode", None),
            pending,
            len(self.active_shms),
        )
        for task_id in list({task_id for task_id, _ in self.active_shms} | set(self.active_effects)):
            self._cleanup_active_task(task_id)
        self.process = None
        return True

    def start(self):
        if self.process is None or not self.process.is_alive():
            self.stop_event.clear()
            if self.thread_mode:
                self.process = threading.Thread(
                    target=worker_process,
                    name="ASyncWorkerThread",
                    args=(self.input_queue, self.result_queue, self.msg_queue, self.stop_event, config._config, self.latest_tasks),
                    daemon=True,
                )
                self.process.start()
            else:
                # Pass current config to worker
                self.process = Process(
                    target=worker_process,
                    name="ASyncWorker", 
                    args=(self.input_queue, self.result_queue, self.msg_queue, self.stop_event, config._config, self.latest_tasks)
                )
                self.process.daemon = True
                self.process.start()
            logging.info("AsyncWorker started.")

    def stop(self):
        if self.process:
            self.stop_event.set()
            self.process.join(timeout=0.2) # Short wait

            if self.thread_mode:
                self.process = None
                logging.info("AsyncWorker stopped.")
            else:
                if self.process.is_alive():
                    logging.warning(f"Terminating worker process {self.process.pid}...")
                    self.process.terminate()
                    self.process.join(timeout=0.1)
                    
                if self.process.is_alive():
                    logging.warning(f"Killing worker process {self.process.pid}...")
                    try:
                        self.process.kill() # Force kill
                        self.process.join()
                    except AttributeError:
                        # In case python version is old or process object issues
                        import os
                        import signal
                        try:
                            os.kill(self.process.pid, signal.SIGKILL)
                        except OSError:
                            # プロセスが既に終了している等のレース（ベストエフォートの強制終了）
                            pass
                            
                self.process = None
                logging.info("AsyncWorker stopped.")
            
        # Clean up queues (drain)
        try:
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
        except (Empty, OSError, ValueError):
            # Empty: empty()判定後にキューが空になるレース
            # OSError/ValueError: 停止済みプロセスのキューが既に閉じられている場合
            # （いずれも停止処理のベストエフォートなドレイン）
            pass
            
    def restart(self):
        """
        Forcefully restart the worker.
        Useful for cancelling running heavy tasks or recovering from errors.
        SAFELY recreates queues to avoid corruption from terminated process.
        """
        self.stop()
        
        # Clear active SHMs (tasks are cancelled)
        for task_id, shm in list(self.active_shms):
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, OSError):
                # 既に close/unlink 済みのSHMのレース（再起動時のベストエフォートな後始末）
                pass
        self.active_shms.clear()
        self.active_effects.clear()
        self.active_started_at.clear()
        
        # Recreate queues to avoid corruption
        self.input_queue = Queue()
        self.result_queue = Queue()
        self.msg_queue = Queue()
        self.stop_event = Event()
        
        self.start()
        
        # We might need to handle leftover SHMs here, but it's tricky.

    def submit_task(self, effect_name, image, params, efconfig):
        """
        Submit a task to the background worker.
        Returns task_id.
        """
        self.start() # Ensure started
        
        self.task_counter += 1
        task_id = self.task_counter
        
        # Create SharedMemory for input image
        shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        shm_array = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
        shm_array[:] = image[:]
        
        # We keep track of this SHM to unlink it later if needed (or rely on worker to close and we unlink?)
        # Strategy: Main creates, Worker opens & closes. Main unlinks after Worker is done?
        # Actually, simpler: Main creates, puts name in queue. Worker opens, reads, closes.
        # Main MUST unlink it. But when? 
        # If we just put it in queue, we lose reference. 
        # Ideally: Main creates -> Worker reads -> Worker signals done -> Main unlinks.
        # But for input, we can probably unlink immediately after putting to queue IF we trust the OS keeps it alive 
        # as long as it's open? No, in POSIX shm_unlink removes the name but keeps the segment if open. 
        # But in Python shared_memory, strict lifecycle is better.
        # Let's let the worker be responsible for closing its handle, but unlinking must happen 
        # after worker is done reading. 
        
        # To simplify: We won't strictly optimize SHM lifecycle perfectly in this V1.
        # We will let the "inputs" be managed by a simple mechanism: 
        # The worker copies the data immediately then closes. 
        # So multiple heavy tasks might consume RAM. 
        # For now, let's just assume we unlink in `poll_results` or similar? 
        # No, `input` SHM is transient. 
        # Use a strategy: Main creates SHM, sends to Worker. Worker reads. 
        # Main can unlink after some time? No.
        # Correct approach: Worker sends back an acknowledgement? Too complex.
        # Alternative: We use the same SHM for return? No, shape might change.
        
        # Let's track input SHMs associated with tasks.
        
        # pickle 不可な属性(get_ai_depth_map 等のクロージャ / processor)の除去は
        # EffectConfig.__getstate__ が自動で行う（callable と processor を None 化）。
        safe_efconfig = copy.copy(efconfig)
        safe_efconfig.processor = None

        task_params = _task_params_for_worker(effect_name, params)
        task = (task_id, effect_name, shm.name, image.shape, str(image.dtype), task_params, safe_efconfig)
        try:
            pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                # pickle不可と判明した後の後始末失敗は無視（本エラーは直後にlogging.exceptionで記録）
                pass
            logging.exception(
                "AsyncWorker task is not picklable: task_id=%s effect=%s params=%s error=%s",
                task_id,
                effect_name,
                _worker_param_summary(task_params),
                e,
            )
            raise
        self.input_queue.put(task)
        logging.info(
            "AsyncWorker task submitted: task_id=%s effect=%s shape=%s dtype=%s params=%s active_before=%s",
            task_id,
            effect_name,
            image.shape,
            image.dtype,
            _worker_param_summary(task_params),
            len(self.active_effects),
        )
        
        # Update latest task ID for this effect
        self.latest_tasks[effect_name] = task_id
        
        # We must keep shm open in this process until worker picks it up?
        # Actually, if we close it here, and unlink, it might disappear before worker opens it.
        # If we don't unlink, it leaks.
        # Hack: Unlink immediately? 
        # "On Windows, looking up the name... fails... if it has been marked for deletion."
        # On Mac/Linux, unlink removes the name. New opens fail.
        # So we must NOT unlink until worker has opened it.
        # We will add it to a list and check in poll logic? 
        # Or cleaner: Worker sends an event "Input Read Complete"?
        # Let's stick to a simpler approach: 
        # Input SHMs are tracked in `self.pending_inputs`. 
        # When result comes back (or error), we unlink the input SHM.
        
        self.active_shms.add((task_id, shm))
        self.active_effects[task_id] = effect_name
        self.active_started_at[task_id] = time.monotonic()
        
        return task_id

    def cancel_effect(self, effect_name):
        """
        Cancel pending tasks for a specific effect.
        """
        # Set latest task to a future ID (or current max + 1) to invalidate pending
        # But we don't know pending IDs easily.
        # Just ensure any task currently in queue with id <= current is skipped.
        # We can just increment counter? No, counter is global.
        # We set latest_tasks[effect_name] = self.task_counter + 1
        # So any task (id <= task_counter) will be seen as old.
        self.latest_tasks[effect_name] = self.task_counter + 1

        # Only pay the worker-restart cost (terminate/kill -> join -> spawn) when this
        # effect actually has in-flight work. The logical cancel above already makes
        # the worker skip stale tasks it hasn't started yet, so a restart would just
        # be wasted spawn cost on e.g. frequent slider-driven cancels.
        if self.has_pending_effect(effect_name):
            self.restart()

    def cancel_all(self):
        """
        Cancel all pending tasks and terminate any currently running tasks.
        """
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except Empty:
                break

        # Only restart if something is actually in flight; otherwise there is no
        # running task for a restart to stop.
        if self.has_pending_tasks():
            self.restart()

    def has_pending_tasks(self):
        """
        処理中のタスクがあるかどうかを判定
        Returns:
            bool: 処理中のタスクがある場合はTrue
        """
        self.reap_dead_worker()
        # multiprocessing.Queue.empty() can report stale state after a worker
        # consumes a task, so use only the task ids tracked by this process.
        return len(self.active_shms) > 0 or bool(self.active_effects)

    def has_pending_effect(self, effect_name):
        # reap_dead_worker() is only otherwise invoked via has_pending_tasks(); call
        # it here too so a crashed process doesn't leave stale active_effects entries
        # making this report in-flight work that no longer exists.
        self.reap_dead_worker()
        return any(name == effect_name for name in self.active_effects.values())

    def effect_elapsed_seconds(self, effect_name):
        starts = [
            self.active_started_at.get(task_id)
            for task_id, name in self.active_effects.items()
            if name == effect_name
        ]
        starts = [started_at for started_at in starts if started_at is not None]
        if not starts:
            return None
        return time.monotonic() - min(starts)

    def cancel_timed_out_effects(self):
        cancelled = []
        for effect_name, timeout_seconds in EFFECT_TIMEOUT_SECONDS.items():
            elapsed = self.effect_elapsed_seconds(effect_name)
            if elapsed is not None and elapsed > timeout_seconds:
                logging.error(
                    "Async task timed out: %s elapsed=%.1fs timeout=%.1fs",
                    effect_name,
                    elapsed,
                    timeout_seconds,
                )
                self.cancel_effect(effect_name)
                cancelled.append(effect_name)
        return cancelled

    def poll_results(self):
        """
        Yields (task_id, result_image_or_None, error_msg).
        """
        results = []
        while True:
            try:
                res = self.result_queue.get_nowait()
                task_id = res['task_id']
                effect_name = self.active_effects.get(task_id)
                self._cleanup_active_task(task_id)

                if res['status'] == 'success':
                    # Get result from SHM
                    shm_name = res['shm_name']
                    shape = res['shape']
                    dtype_str = res['dtype']
                    
                    try:
                        r_shm = shared_memory.SharedMemory(name=shm_name)
                        r_array = np.ndarray(shape, dtype=dtype_str, buffer=r_shm.buf)
                        # Copy to local
                        img = r_array.copy()
                        r_shm.close()
                        r_shm.unlink() # We own the result SHM now
                        
                        logging.info(
                            "AsyncWorker result received: task_id=%s effect=%s shape=%s dtype=%s active_left=%s",
                            task_id,
                            effect_name,
                            shape,
                            dtype_str,
                            len(self.active_effects),
                        )
                        results.append((task_id, img, None))
                    except Exception as e:
                        logging.error(f"Failed to read result SHM: {e}")
                        results.append((task_id, None, str(e)))
                else:
                    logging.warning(
                        "AsyncWorker result status=%s: task_id=%s effect=%s message=%s active_left=%s",
                        res.get('status'),
                        task_id,
                        effect_name,
                        res.get('message', ''),
                        len(self.active_effects),
                    )
                    results.append((task_id, None, res.get('message', 'Unknown error')))
                    
            except Empty:
                break
        
        self.reap_dead_worker()
        
        return results

    def poll_messages(self):
        """
        Yields messages from the worker process (e.g. waitinfo updates).
        """
        while True:
            try:
                msg = self.msg_queue.get_nowait()
                yield msg
            except Empty:
                break
