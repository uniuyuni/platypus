
import multiprocessing

_msg_queue = None

def init(queue):
    global _msg_queue

    _msg_queue = queue

def set_text(tag, text, main_widget=None):
    global _msg_queue

    if _msg_queue is not None:
        try:
            _msg_queue.put({'type': 'waitinfo', 'tag': tag, 'text': text})
        except Exception:
            # キュークローズ等のレースでのステータス通知失敗は無視（表示更新の一手段に過ぎない）
            pass
        return

    # Fallback to direct UI update (Main Process only)
    try:
        widget = main_widget.ids["waitinfo_" + tag]

        widget.text = text + " "
        if text is None or text == "":
            widget.disabled = True
        else:
            widget.disabled = False
    except (KeyError, AttributeError):
        # 該当ウィジェットが存在しない/未初期化な場合のベストエフォート表示更新
        pass
