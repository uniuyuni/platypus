
import AppKit
import fcntl
from AppKit import NSScreen, NSApplication, NSEvent, NSBundle, NSObject
from Quartz import CGDisplayScreenSize, CGDisplayPixelsWide, CGDisplayPixelsHigh

import define

class FileChooser:
    '''A native implementation of file chooser dialogs using Apple's API
    through pyobjus.

    Not implemented features:
    * filters (partial, wildcards are converted to extensions if possible.
        Pass the Mac-specific "use_extensions" if you can provide
        Mac OS X-compatible to avoid automatic conversion)
    * multiple (only for save dialog. Available in open dialog)
    * icon
    * preview
    '''

    mode = "open"
    path = None
    multiple = False
    filters = []
    preview = False
    title = None
    icon = None
    show_hidden = False
    use_extensions = False

    def __init__(self, *args, **kwargs):
        self._handle_selection = kwargs.pop(
            'on_selection', self._handle_selection
        )

        # Simulate Kivy's behavior
        for i in kwargs:
            setattr(self, i, kwargs[i])

    @staticmethod
    def _handle_selection(selection):
        '''
        Dummy placeholder for returning selection from chooser.
        '''
        return selection

    def run(self):
        panel = None
        if self.mode in ("open", "dir", "dir_and_files"):
            panel = AppKit.NSOpenPanel.openPanel()

            panel.setCanChooseDirectories_(self.mode != "open")
            panel.setCanChooseFiles_(self.mode != "dir")

            if self.multiple:
                panel.setAllowsMultipleSelection_(True)
        elif self.mode == "save":
            panel = AppKit.NSSavePanel.savePanel()
        else:
            assert False, self.mode

        panel.setCanCreateDirectories_(True)
        panel.setShowsHiddenFiles_(self.show_hidden)

        if self.title:
            panel.setTitle_(AppKit.NSString.alloc().initWithString_(self.title))

        # Mac OS X does not support wildcards unlike the other platforms.
        # This tries to convert wildcards to "extensions" when possible,
        # ans sets the panel to also allow other file types, just to be safe.
        if self.filters:
            filthies = []
            for f in self.filters:
                if isinstance(f, str):
                    f = (None, f)
                for s in f[1:]:
                    if not self.use_extensions:
                        if s.strip().endswith("*"):
                            continue
                    pystr = s.strip().split("*")[-1].split(".")[-1]
                    filthies.append(AppKit.NSString.alloc().initWithString_(pystr))

            ftypes_arr = AppKit.NSArray.alloc().initWithArray_(filthies)
            # todo: switch to allowedContentTypes
            panel.setAllowedFileTypes_(ftypes_arr)
            panel.setAllowsOtherFileTypes_(not self.use_extensions)

        if self.path:
            url = AppKit.NSURL.fileURLWithPath_(self.path)
            panel.setDirectoryURL_(url)

        selection = None

        try:
            _center_window_on_app(panel)
            if panel.runModal():
                if self.mode == "save" or not self.multiple:
                    selection = [panel.filename().UTF8String()]
                else:
                    filename = panel.filenames()
                    selection = [
                        filename.objectAtIndex_(x).UTF8String()
                        for x in range(filename.count())]
        finally:
            _restore_app_window_focus()

        self._handle_selection(selection)

        return selection

def fadvice(file_path, use_cache=True):
    with open(file_path, "rb") as fd:
        # キャッシュの有効/無効を設定
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 0 if use_cache else 1)
        
        # シーケンシャルアクセスの場合、先読みを有効化
        fcntl.fcntl(fd, 45, 1 if use_cache else 0)  # F_RDAHEAD の値は macOS でのみ有効


def get_screens_info():
    """
    NSScreenを使用して全ディスプレイ情報を取得
    スクリーンID、解像度（ポイント/ピクセル）、表示位置、スケールファクタ、物理サイズ(mm)を含む
    """
    screens = NSScreen.screens()
    results = []
    
    for i, screen in enumerate(screens):
        # スクリーンID
        screen_number = screen.deviceDescription().get('NSScreenNumber', 0)
        display_id = int(screen_number)
        
        # 表示位置とサイズ（ポイント）
        frame = screen.frame()
        
        # 実際のピクセル解像度
        backing_frame = screen.convertRectToBacking_(frame)
        
        # スケールファクタ
        scale_factor = screen.backingScaleFactor()
        
        # 物理サイズ（mm）をQuartzから取得
        try:
            screen_size = CGDisplayScreenSize(display_id)
            width_mm = screen_size.width
            height_mm = screen_size.height
        except Exception:
            # 取得できない場合は0
            width_mm = 0
            height_mm = 0
        
        # 物理ピクセル解像度をQuartzから取得（より正確）
        try:
            phys_pixels_wide = CGDisplayPixelsWide(display_id)
            phys_pixels_high = CGDisplayPixelsHigh(display_id)
        except Exception:
            # 取得できない場合はバッキングフレームから計算
            phys_pixels_wide = backing_frame.size.width
            phys_pixels_high = backing_frame.size.height
        
        results.append({
            # 基本情報
            'id': display_id,
            'index': i,
            'is_primary': (i == 0),
            
            # 表示位置
            'x': int(frame.origin.x),
            'y': int(frame.origin.y),
                        
            # 物理解像度（ピクセル）
            'width_pixels': int(phys_pixels_wide),
            'height_pixels': int(phys_pixels_high),
            
            # 表示解像度（ポイント）
            'width_points': int(frame.size.width),
            'height_points': int(frame.size.height),

            # スケールファクタ
            'scale': float(scale_factor),
            
            # 物理サイズ（mm）
            'width_mm': float(width_mm),
            'height_mm': float(height_mm)
        })
    
    return results

def print_screens_info(screens):
    """ディスプレイ情報を見やすく表示"""
    print("=" * 100)
    print(f"{'No.':<3} {'ID':<10} {'位置':<10} {'解像度（ポイント）':<9} {'解像度（ピクセル）':<9} {'Scale':<7} {'物理サイズ(mm)':<16}")
    print("-" * 100)
    
    for screen in screens:
        pos = f"({screen['x']},{screen['y']})"
        points_res = f"{screen['width_points']}×{screen['height_points']}"
        pixels_res = f"{screen['width_pixels']}×{screen['height_pixels']}"
        scale = f"{screen['scale']}x"
        phys_size = f"{screen['width_mm']:.0f}×{screen['height_mm']:.0f}"
        
        print(f"{screen['index']:<3} "
              f"{screen['id']:<10} "
              f"{pos:<12} "
              f"{points_res:<18} "
              f"{pixels_res:<18} "
              f"{scale:<7} "
              f"{phys_size:<16}")

def calculate_ppi(width_px, height_px, width_mm, height_mm):
    """PPIを計算"""
    import math
    
    # 対角線のピクセル数
    diag_px = math.sqrt(width_px**2 + height_px**2)
    
    # 対角線のインチ数（mmをインチに変換: 1インチ = 25.4mm）
    diag_inch = math.sqrt((width_mm/25.4)**2 + (height_mm/25.4)**2)
    
    # PPI (対角線ベース)
    return diag_px / diag_inch if diag_inch > 0 else 0

_screens = get_screens_info()


def get_primary_display_size_points():
    """主ディスプレイ（または先頭）の表示サイズ (幅, 高さ) ポイント。フォールバック用。"""
    if not _screens:
        return None
    for m in _screens:
        if m.get("is_primary"):
            return int(m["width_points"]), int(m["height_points"])
    m0 = _screens[0]
    return int(m0["width_points"]), int(m0["height_points"])


def get_primary_display_backing_pixel_size():
    """
    主ディスプレイ（または先頭）の枠。ポイント * scale = バッキング画素。
    kvutils の ref*device.dpi_scale と同じ系の「レイアウト/バッキング」寄りの単位。
    """
    if not _screens:
        return None
    for m in _screens:
        if m.get("is_primary"):
            s = float(m.get("scale", 1.0) or 1.0)
            return int(round(m["width_points"] * s)), int(round(m["height_points"] * s))
    m0 = _screens[0]
    s = float(m0.get("scale", 1.0) or 1.0)
    return int(round(m0["width_points"] * s)), int(round(m0["height_points"] * s))


def get_self_window_position(app_name=None):
    """
    最もシンプルなバージョン - 座標のみ
    """
    if app_name:
        window = get_window(app_name)
    else:
        window = app.mainWindow()
    
    if window:
        frame = window.frame()
        screen_index = NSScreen.screens().index(window.screen())
        return (frame.origin.x, frame.origin.y, frame.size.width, frame.size.height, screen_index)

    # ウィンドウが見つからない場合はマウスカーソルの位置を返す
    mouse_location = NSEvent.mouseLocation()
    return (mouse_location.x, mouse_location.y, 0, 0, None)

def get_app_window_scale():
    try:
        window = get_window(define.APPNAME)
        if window is None:
            return None
        screen = window.screen()
        if screen is None:
            return None
        return float(screen.backingScaleFactor())
    except Exception:
        return None

def dpi_scale():
    scale = get_app_window_scale()
    if scale:
        return scale
    display = get_current_display()
    return _screens[display['display']]['scale']

def get_current_display(win_x=0, win_y=0, win_display=None):
    global _screens

    # 現在のウィンドウの左下座標
    if win_x == 0 and win_y == 0:
        win_x, win_y, _, _, win_display = get_self_window_position(define.APPNAME)
    
    # 所属しているディスプレイが取得できている場合は、それを返す
    if win_display is not None:
        return {"display": win_display, "width": _screens[win_display]['width_points'], "height": _screens[win_display]['height_points'], "is_primary": _screens[win_display]['is_primary']}

    for m in _screens:
        if m['is_primary'] == True:
            primary = m
            break

    for i, m in enumerate(_screens):
        # Yの反転
        my = 0
        if m['y'] != 0:
            my = primary['height_points'] if m['y'] > 0 else -m['height_points']

        if m['x'] <= win_x < m['x'] + m['width_points'] and my <= win_y < my + m['height_points']:
            return {"display": i, "width": m['width_points'], "height": m['height_points'], "is_primary": m['is_primary']}
    
    return {"display": primary['index'], "width": primary['width_points'], "height": primary['height_points'], "is_primary": primary['is_primary']}

def get_window(app_name):
    app = NSApplication.sharedApplication()

    # mainWindow() ではなく、windows() リストから探す
    windows = app.windows()

    if windows and windows.count() > 0:
        # より確実に特定したい場合（例：特定のタイトルを持つウィンドウ）
        for i in range(windows.count()):
            win = windows.objectAtIndex_(i)
            if win.title() == app_name:
                return win
    
    return None


def get_app_window_screen_size_points():
    """
    Platypus ウィンドウが乗っている NSScreen の表示サイズ (幅, 高さ) ポイント。
    マルチモニタで Kivy Window.system_size が仮想デスクトップ幅になるのを避ける。
    """
    try:
        w = get_window(define.APPNAME)
        if w is None:
            return None
        sc = w.screen()
        if sc is None:
            return None
        f = sc.frame()
        return int(f.size.width), int(f.size.height)
    except Exception:
        return None


def get_app_window_screen_backing_pixel_size():
    """
    Platypus ウィンドウが乗っている NSScreen の (幅, 高さ) バッキング画素。
    マルチディスプレイ各々の scale がその画面に反映される。Kivy 座標系には依存しない。
    """
    try:
        w = get_window(define.APPNAME)
        if w is None:
            return None
        sc = w.screen()
        if sc is None:
            return None
        f = sc.frame()
        s = float(sc.backingScaleFactor())
        return int(round(f.size.width * s)), int(round(f.size.height * s))
    except Exception:
        return None


def _objc_text(value):
    """NSString/bytes/str を Python の str に寄せる。"""
    if value is None:
        return None
    try:
        if hasattr(value, "UTF8String"):
            value = value.UTF8String()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    except Exception:
        return None


def _normalize_color_space_name(name):
    """
    macOS の表示名をアプリ内の表示色域名に寄せる。
    未知のカスタム ICC 名は、勝手に sRGB 扱いせず元の名前を返す。
    """
    text = _objc_text(name)
    if not text:
        return None

    text = text.strip()
    key = (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", "")
    )
    compact = "".join(ch for ch in key if ch.isalnum())

    if compact in {"srgb", "srgbiec6196621", "srgbcolorprofile"}:
        return "sRGB"
    if compact in {"adobergb", "adobergb1998", "adobergbcolorprofile"}:
        return "Adobe RGB (1998)"
    if "displayp3" in compact or "p3d65" in compact:
        return "Display P3"
    if compact in {"prophotorgb", "rommrgb"}:
        return "ProPhoto RGB"
    if "bt2020" in compact or "rec2020" in compact or "iturbt2020" in compact:
        return "Rec.2020"
    if "bt709" in compact or "rec709" in compact:
        return "Rec.709"

    return text


def _nsdata_bytes(data):
    if data is None:
        return None
    try:
        if isinstance(data, bytes):
            return data
        return memoryview(data).tobytes()
    except Exception:
        # memoryview非対応のNSDataラッパもあるため、次の変換方法にフォールバック
        pass
    try:
        return bytes(data)
    except Exception:
        # bytes()変換不可なら最後の手段（getBytes_length_）へフォールバック
        pass
    try:
        length = int(data.length())
        out = bytearray(length)
        data.getBytes_length_(out, length)
        return bytes(out)
    except Exception:
        return None


def _icc_tag_data(icc_data, signature):
    if not icc_data or len(icc_data) < 132:
        return None
    try:
        tag_count = int.from_bytes(icc_data[128:132], "big")
        table_start = 132
        for i in range(tag_count):
            entry = table_start + i * 12
            if entry + 12 > len(icc_data):
                return None
            sig = icc_data[entry:entry + 4]
            offset = int.from_bytes(icc_data[entry + 4:entry + 8], "big")
            size = int.from_bytes(icc_data[entry + 8:entry + 12], "big")
            if sig == signature and offset >= 0 and size >= 0 and offset + size <= len(icc_data):
                return icc_data[offset:offset + size]
    except Exception:
        return None
    return None


def _s15_fixed16_to_float(raw):
    value = int.from_bytes(raw, "big", signed=True)
    return value / 65536.0


def _icc_xyz_tag_xy(icc_data, signature):
    tag = _icc_tag_data(icc_data, signature)
    if not tag or len(tag) < 20 or tag[:4] != b"XYZ ":
        return None
    try:
        x_val = _s15_fixed16_to_float(tag[8:12])
        y_val = _s15_fixed16_to_float(tag[12:16])
        z_val = _s15_fixed16_to_float(tag[16:20])
        total = x_val + y_val + z_val
        if total <= 0:
            return None
        return (x_val / total, y_val / total)
    except Exception:
        return None


def _icc_rgb_primary_xy(icc_data):
    red = _icc_xyz_tag_xy(icc_data, b"rXYZ")
    green = _icc_xyz_tag_xy(icc_data, b"gXYZ")
    blue = _icc_xyz_tag_xy(icc_data, b"bXYZ")
    if red is None or green is None or blue is None:
        return None
    return (red, green, blue)


def _primary_distance(primary, target):
    return sum(
        (primary[i][0] - target[i][0]) ** 2 + (primary[i][1] - target[i][1]) ** 2
        for i in range(3)
    )


def _classify_rgb_primary_gamut(primary):
    if primary is None:
        return None

    # ICC RGB colorant tags are PCS D50 adapted, so compare against D50-adapted
    # xy positions rather than the familiar D65 spec primaries.
    targets = {
        "sRGB": ((0.6484, 0.3309), (0.3212, 0.5979), (0.1559, 0.0660)),
        "Display P3": ((0.6820, 0.3196), (0.2845, 0.6746), (0.1559, 0.0660)),
        "Adobe RGB (1998)": ((0.6484, 0.3309), (0.2310, 0.7040), (0.1559, 0.0660)),
        "Rec.2020": ((0.7080, 0.2920), (0.1700, 0.7970), (0.1310, 0.0460)),
    }
    best_name = None
    best_distance = None
    for name, target in targets.items():
        distance = _primary_distance(primary, target)
        if best_distance is None or distance < best_distance:
            best_name = name
            best_distance = distance

    return best_name if best_distance is not None and best_distance <= 0.015 else None


def _screen_icc_profile_data(screen):
    try:
        color_space = screen.colorSpace()
        if color_space is None:
            return None
        if hasattr(color_space, "ICCProfileData"):
            return _nsdata_bytes(color_space.ICCProfileData())
        if hasattr(color_space, "iccProfileData"):
            return _nsdata_bytes(color_space.iccProfileData())
    except Exception:
        return None
    return None


def _screen_display_gamut_from_capability(screen):
    try:
        can_represent = getattr(screen, "canRepresentDisplayGamut_", None)
        if can_represent is None:
            return None
        p3 = getattr(AppKit, "NSDisplayGamutP3", 2)
        srgb = getattr(AppKit, "NSDisplayGamutSRGB", 1)
        if bool(can_represent(p3)):
            return "Display P3"
        if bool(can_represent(srgb)):
            return "sRGB"
    except Exception:
        return None
    return None


def _screen_display_color_gamut_name(screen):
    icc_data = _screen_icc_profile_data(screen)
    gamut = _classify_rgb_primary_gamut(_icc_rgb_primary_xy(icc_data))
    if gamut:
        return gamut
    return _screen_display_gamut_from_capability(screen)


def _known_display_color_gamut_name(name):
    gamut = _normalize_color_space_name(name)
    if gamut in {
        "sRGB",
        "Display P3",
        "Adobe RGB (1998)",
        "ProPhoto RGB",
        "Rec.2020",
        "Rec.709",
    }:
        return gamut
    return None


def get_screen_color_space_name(screen=None, default="sRGB", normalize=True):
    """
    NSScreen の表示色域名を返す。
    normalize=True では ICC primaries / 表示能力から sRGB や Display P3 等へ寄せる。
    """
    try:
        if screen is None:
            screen = NSScreen.mainScreen()
        if screen is None:
            return default

        color_space = screen.colorSpace()
        name = None
        if color_space is not None:
            name = color_space.localizedName()
        if not name:
            name = screen.deviceDescription().get("NSDeviceColorSpaceName")

        if normalize:
            icc_data = _screen_icc_profile_data(screen)
            icc_gamut = _classify_rgb_primary_gamut(_icc_rgb_primary_xy(icc_data))
            name_gamut = _known_display_color_gamut_name(name)
            capability_gamut = _screen_display_gamut_from_capability(screen)
            return icc_gamut or name_gamut or capability_gamut or _normalize_color_space_name(name) or default
        return _objc_text(name) or default
    except Exception:
        return default


def _get_app_window_screen(app_name=None):
    try:
        window = get_window(app_name or define.APPNAME)
        if window is not None:
            screen = window.screen()
            if screen is not None:
                return screen
    except Exception:
        # ウィンドウ/スクリーン取得失敗時はmainScreen()にフォールバック
        pass

    try:
        return NSScreen.mainScreen()
    except Exception:
        return None


def get_app_window_screen_color_space_name(app_name=None, default="sRGB", normalize=True):
    """
    自分のアプリウィンドウが現在乗っている画面の色空間名を返す。
    例: "sRGB", "Display P3", "Adobe RGB (1998)"。
    """
    return get_screen_color_space_name(
        _get_app_window_screen(app_name),
        default=default,
        normalize=normalize,
    )


def set_window_autosave(app_name, window_name):

    target_window = get_window(app_name)
    if target_window:
        target_window.setFrameAutosaveName_(window_name)
        return True

    return False

"""
macos_dialog.py
---------------
macOS ネイティブダイアログを Python から呼び出すモジュール。
外部ライブラリ不要（osascript を使用）。PyObjC が入っていればより高機能。

対応ダイアログ:
  - alert()          : 警告/情報ダイアログ
  - confirm()        : OK/キャンセル 確認ダイアログ
  - prompt_native()  : テキスト入力ダイアログ（PyObjC NSAlert、要メインスレッド）
  - choose_from_list(): リスト選択ダイアログ
  - file_open()      : ファイルを開くダイアログ
  - file_save()      : ファイルを保存ダイアログ
  - folder_select()  : フォルダ選択ダイアログ
  - notify()         : 通知センター通知

使い方:
  import macos_dialog as dlg

  dlg.alert("エラーが発生しました", title="エラー", icon="stop")
  name = dlg.prompt_native("名前を入力してください", default="太郎")
  path = dlg.file_open(file_types=["txt", "csv"])
"""

import subprocess
import json
import re
import os
from typing import Any, Optional, Union

# ────────────────────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ────────────────────────────────────────────────────────────────────────────

def _run_applescript(script: str) -> str:
    """AppleScript を実行して stdout を返す。キャンセル時は '' を返す。"""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        # ユーザーがキャンセルした場合は空文字列を返す
        err = result.stderr.strip()
        if "User canceled" in err or "(-128)" in err:
            return ""
        raise RuntimeError(f"AppleScript エラー: {err}")
    return result.stdout.strip()


def _escape(s: Any) -> str:
    """AppleScript 文字列内でエスケープが必要な文字を処理する。"""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _alert_style(icon: str) -> int:
    styles = {
        "note": AppKit.NSAlertStyleInformational,
        "info": AppKit.NSAlertStyleInformational,
        "informational": AppKit.NSAlertStyleInformational,
        "caution": AppKit.NSAlertStyleWarning,
        "warning": AppKit.NSAlertStyleWarning,
        "stop": AppKit.NSAlertStyleCritical,
        "critical": AppKit.NSAlertStyleCritical,
    }
    return styles.get(str(icon or "").lower(), AppKit.NSAlertStyleInformational)


def _app_main_window():
    try:
        return get_window(define.APPNAME)
    except Exception:
        return None


def _center_window_on_app(win) -> bool:
    parent = _app_main_window()
    if parent is None or win is None:
        return False
    try:
        parent_frame = parent.frame()
        frame = win.frame()
        x = parent_frame.origin.x + (parent_frame.size.width - frame.size.width) / 2.0
        y = parent_frame.origin.y + (parent_frame.size.height - frame.size.height) / 2.0
        win.setFrameOrigin_(AppKit.NSMakePoint(x, y))
        return True
    except Exception:
        return False


def _restore_app_window_focus() -> bool:
    app = NSApplication.sharedApplication()
    try:
        if not app.isActive():
            return False
    except Exception:
        # isActive() 取得失敗時は非アクティブ扱いにせず、以降のフォーカス処理を続行
        pass
    win = _app_main_window()
    if win is None:
        return False
    try:
        win.makeMainWindow()
        win.makeKeyAndOrderFront_(None)
        return True
    except Exception:
        return False


class _AlertController(NSObject):
    """Button target for the native app-modal alert window."""

    def choose_(self, sender):
        NSApplication.sharedApplication().stopModalWithCode_(int(sender.tag()))


def _run_native_alert(
    message: str,
    title: str,
    icon: str,
    buttons: list[str],
) -> Optional[str]:
    app = NSApplication.sharedApplication()
    controller = _AlertController.alloc().init()
    clean_buttons = [str(button) for button in (buttons or ["OK"])]

    pad = 20.0
    right_pad = 10.0
    bottom_pad = 10.0
    center_gap = 10.0
    gap = 12.0
    btn_w, btn_h = 92.0, 32.0
    icon_w = 34.0
    content_w = 360.0
    title_h = 24.0 if title else 0.0
    message_h = max(44.0, min(180.0, 18.0 * (str(message or "").count("\n") + 2)))
    button_count = max(1, len(clean_buttons))
    buttons_w = button_count * btn_w + (button_count - 1) * gap
    win_w = max(content_w + icon_w + pad + center_gap + right_pad, buttons_w + pad + right_pad)
    y_btn = bottom_pad
    y_msg = y_btn + btn_h + gap
    y_title = y_msg + message_h
    win_h = y_title + title_h + pad

    win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(0.0, 0.0, win_w, win_h),
        AppKit.NSWindowStyleMaskTitled,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    win.setTitle_(str(title or ""))
    content = win.contentView()

    icon = AppKit.NSImageView.alloc().initWithFrame_(
        AppKit.NSMakeRect(pad, y_msg + max(0.0, message_h - icon_w) / 2.0, icon_w, icon_w)
    )
    icon.setImage_(AppKit.NSImage.imageNamed_(AppKit.NSImageNameCaution))
    content.addSubview_(icon)

    text_x = pad + icon_w + center_gap
    if title:
        title_label = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(text_x, y_title, win_w - text_x - right_pad, title_h)
        )
        title_label.setStringValue_(str(title))
        title_label.setBezeled_(False)
        title_label.setDrawsBackground_(False)
        title_label.setEditable_(False)
        title_label.setSelectable_(False)
        title_label.setFont_(AppKit.NSFont.boldSystemFontOfSize_(13.0))
        content.addSubview_(title_label)

    message_label = AppKit.NSTextField.alloc().initWithFrame_(
        AppKit.NSMakeRect(text_x, y_msg, win_w - text_x - right_pad, message_h)
    )
    message_label.setStringValue_(str(message or ""))
    message_label.setBezeled_(False)
    message_label.setDrawsBackground_(False)
    message_label.setEditable_(False)
    message_label.setSelectable_(False)
    content.addSubview_(message_label)

    start_x = win_w - right_pad - buttons_w
    for i, label in enumerate(reversed(clean_buttons)):
        button = AppKit.NSButton.alloc().initWithFrame_(
            AppKit.NSMakeRect(start_x + i * (btn_w + gap), y_btn, btn_w, btn_h)
        )
        button.setTitle_(label)
        button.setBezelStyle_(AppKit.NSBezelStyleRounded)
        button.setTarget_(controller)
        button.setAction_("choose:")
        index = clean_buttons.index(label)
        button.setTag_(index + 1)
        if index == 0:
            button.setKeyEquivalent_("\r")
        if label.lower() in {"cancel", "キャンセル"}:
            button.setKeyEquivalent_("\x1b")
        content.addSubview_(button)

    if not _center_window_on_app(win):
        win.center()
    win.makeKeyAndOrderFront_(None)
    try:
        code = int(app.runModalForWindow_(win))
    finally:
        win.orderOut_(None)
        _restore_app_window_focus()
    index = code - 1
    if 0 <= index < len(clean_buttons):
        return clean_buttons[index]
    return None


# ────────────────────────────────────────────────────────────────────────────
# 公開 API
# ────────────────────────────────────────────────────────────────────────────

def alert(
    message: str,
    title: str = "通知",
    subtitle: str = "",
    icon: str = "note",  # "note" | "caution" | "stop"
    buttons: list[str] = None,
    default_button: str = None,
) -> Optional[str]:
    """
    警告/情報ダイアログを表示する。

    Parameters
    ----------
    message       : 本文メッセージ
    title         : ウィンドウタイトル
    subtitle      : サブタイトル（任意）
    icon          : アイコン種別 "note"(ℹ️) / "caution"(⚠️) / "stop"(🛑)
    buttons       : ボタンラベルのリスト（最大3つ）
    default_button: デフォルトボタンのラベル

    Returns
    -------
    押されたボタンのラベル。キャンセル時は None。
    """
    buttons = buttons or ["OK"]
    text = f"{subtitle}\n\n{message}" if subtitle else message
    if default_button and default_button in buttons:
        buttons = [default_button] + [button for button in buttons if button != default_button]
    return _run_native_alert(text, title, icon, buttons)


def confirm(
    message: str,
    title: str = "確認",
    ok_label: str = "OK",
    cancel_label: str = "キャンセル",
    icon: str = "caution",
) -> bool:
    """
    OK / キャンセル 確認ダイアログ。

    Returns
    -------
    True: OK が押された, False: キャンセル
    """
    return _run_native_alert(
        message,
        title,
        icon,
        [ok_label, cancel_label],
    ) == ok_label


# prompt_native の ascii_only フラグ。ダイアログは常にメインスレッドで modal（同時に1つ）の
# ため、デリゲートからはモジュール変数で参照すれば十分。
_prompt_ascii_only = False


class _PromptController(NSObject):
    """prompt_native のボタンターゲット兼テキスト入力デリゲート。"""

    def ok_(self, sender):
        NSApplication.sharedApplication().stopModalWithCode_(1)

    def cancel_(self, sender):
        NSApplication.sharedApplication().stopModalWithCode_(0)

    def controlTextDidChange_(self, notification):
        # ascii_only 指定時は非 ASCII（日本語等）を打ち込んだ端から除去する。
        if not _prompt_ascii_only:
            return
        field = notification.object()
        s = str(field.stringValue())
        filtered = "".join(ch for ch in s if ord(ch) < 128)
        if filtered != s:
            field.setStringValue_(filtered)


def prompt_native(
    message: str = "",
    title: str = "入力",
    default: str = "",
    ok_label: str = "OK",
    show_cancel: bool = False,
    cancel_label: str = "キャンセル",
    ascii_only: bool = False,
    width: float = 360.0,
) -> Optional[str]:
    """
    PyObjC ネイティブのテキスト入力ダイアログ（自前 NSWindow）。

    NSAlert と違いアイコン枠が無く、OK ボタンは右下に置く。FileChooser(NSOpenPanel) 同様に
    アプリ本体プロセス内で modal 実行するため最前面でフォーカスを取れる
    （カーソル表示・IME 有効）。**必ずメインスレッドから呼ぶこと**
    （runModalForWindow_ がメインスレッドをブロックする）。

    Parameters
    ----------
    show_cancel : True でキャンセルボタンを表示。押下時は None を返す。
    ascii_only  : True で非 ASCII（日本語等）の入力を抑止する（処理側が日本語非対応な用途向け）。

    Returns
    -------
    入力文字列。show_cancel=True でキャンセルされた場合のみ None。
    """
    global _prompt_ascii_only
    _prompt_ascii_only = ascii_only

    app = NSApplication.sharedApplication()
    controller = _PromptController.alloc().init()

    # レイアウト定数（AppKit 座標は左下原点）。pad=左右余白 / pad_y=上下余白（やや詰める）
    pad = 20.0
    pad_y = 12.0
    field_h = 24.0
    btn_w, btn_h = 90.0, 32.0
    gap = 12.0
    win_w = float(width) + pad * 2

    has_msg = bool(message)
    msg_h = 36.0 if has_msg else 0.0

    y_btn = pad_y
    y_field = y_btn + btn_h + gap
    y_msg = y_field + field_h #+ gap
    win_h = (y_msg + msg_h if has_msg else y_field + field_h) + pad_y

    win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(0.0, 0.0, win_w, win_h),
        AppKit.NSWindowStyleMaskTitled,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    win.setTitle_(title)
    content = win.contentView()

    # メッセージ（任意）: ラベル（枠なし・背景なし・編集不可）
    if has_msg:
        label = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(pad, y_msg, win_w - pad * 2, msg_h)
        )
        label.setStringValue_(message)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        content.addSubview_(label)

    # 入力フィールド
    field = AppKit.NSTextField.alloc().initWithFrame_(
        AppKit.NSMakeRect(pad, y_field, win_w - pad * 2, field_h)
    )
    field.setStringValue_(default or "")
    field.setDelegate_(controller)
    content.addSubview_(field)

    # OK ボタン（右下、Enter がデフォルト）
    ok_x = win_w - pad / 2 - btn_w
    ok_btn = AppKit.NSButton.alloc().initWithFrame_(
        AppKit.NSMakeRect(ok_x, y_btn, btn_w, btn_h)
    )
    ok_btn.setTitle_(ok_label)
    ok_btn.setBezelStyle_(AppKit.NSBezelStyleRounded)
    ok_btn.setTarget_(controller)
    ok_btn.setAction_("ok:")
    ok_btn.setKeyEquivalent_("\r")
    content.addSubview_(ok_btn)

    # キャンセルボタン（任意、OK の左、Esc）
    if show_cancel:
        cancel_x = ok_x - btn_w - gap / 2
        cancel_btn = AppKit.NSButton.alloc().initWithFrame_(
            AppKit.NSMakeRect(cancel_x, y_btn, btn_w, btn_h)
        )
        cancel_btn.setTitle_(cancel_label)
        cancel_btn.setBezelStyle_(AppKit.NSBezelStyleRounded)
        cancel_btn.setTarget_(controller)
        cancel_btn.setAction_("cancel:")
        cancel_btn.setKeyEquivalent_("\x1b")  # Esc
        content.addSubview_(cancel_btn)

    # 表示 + テキストフィールドにフォーカス + modal
    if not _center_window_on_app(win):
        win.center()
    win.makeKeyAndOrderFront_(None)
    win.makeFirstResponder_(field)

    if ascii_only:
        # 入力コンテキストを Roman（アルファベット）入力ソースに制限し、日本語 IME 自体を抑止する。
        # フォーカス中の実体はフィールドエディタ(NSTextView)なので、その input context に設定する。
        # （取れない場合のフォールバックとして controlTextDidChange_ 側の非 ASCII 除去も残してある）
        responder = win.firstResponder()
        for obj in (responder, field):
            try:
                ic = obj.inputContext() if obj is not None else None
            except Exception:
                ic = None
            if ic is not None:
                ic.setAllowedInputSourceLocales_(
                    [AppKit.NSAllRomanInputSourcesLocaleIdentifier]
                )

    try:
        code = app.runModalForWindow_(win)
    finally:
        win.orderOut_(None)
        _restore_app_window_focus()

    if show_cancel and code == 0:
        return None
    return str(field.stringValue())


def choose_from_list(
    items: list[str],
    message: str = "項目を選択してください",
    title: str = "選択",
    multiple: bool = False,
    ok_label: str = "選択",
    cancel_label: str = "キャンセル",
) -> Optional[Union[str, list[str]]]:
    """
    リストから項目を選択するダイアログ。

    Parameters
    ----------
    multiple: True で複数選択可能

    Returns
    -------
    選択された文字列（or リスト）。キャンセル時は None。
    """
    items_str = "{" + ", ".join(f'"{_escape(i)}"' for i in items) + "}"
    multi_str = "with multiple selections allowed" if multiple else ""
    script = f"""
    set result to choose from list {items_str} ¬
        with title "{_escape(title)}" ¬
        with prompt "{_escape(message)}" ¬
        OK button name "{_escape(ok_label)}" ¬
        cancel button name "{_escape(cancel_label)}" ¬
        {multi_str}
    if result is false then return ""
    return result as string
    """
    raw = _run_applescript(script)
    if not raw:
        return None
    if multiple:
        return [s.strip() for s in raw.split(",")]
    return raw


def file_open(
    message: str = "ファイルを選択してください",
    file_types: list[str] = None,
    multiple: bool = False,
    start_folder: str = "~",
) -> Optional[Union[str, list[str]]]:
    """
    ファイルを開くダイアログ。

    Parameters
    ----------
    file_types  : 拡張子のリスト（例: ["txt", "csv"]）
    multiple    : True で複数選択可能
    start_folder: 初期表示フォルダ

    Returns
    -------
    選択されたパス文字列（or リスト）。キャンセル時は None。
    """
    type_str = ""
    if file_types:
        type_str = "of type {" + ", ".join(f'"{t}"' for t in file_types) + "}"

    multi_str = "with multiple selections allowed" if multiple else ""
    folder = os.path.expanduser(start_folder)

    script = f"""
    try
        set result to choose file ¬
            with prompt "{_escape(message)}" ¬
            {type_str} ¬
            default location POSIX file "{_escape(folder)}" ¬
            {multi_str}
        if class of result is list then
            set paths to ""
            repeat with f in result
                set paths to paths & POSIX path of f & linefeed
            end repeat
            return paths
        else
            return POSIX path of result
        end if
    on error
        return ""
    end try
    """
    raw = _run_applescript(script).strip()
    if not raw:
        return None
    paths = [p for p in raw.splitlines() if p]
    if multiple:
        return paths
    return paths[0] if paths else None


def file_save(
    message: str = "保存先を選択してください",
    default_name: str = "untitled",
    file_types: list[str] = None,
    start_folder: str = "~",
) -> Optional[str]:
    """
    ファイルを保存ダイアログ。

    Returns
    -------
    選択されたパス文字列。キャンセル時は None。
    """
    folder = os.path.expanduser(start_folder)
    script = f"""
    try
        set result to choose file name ¬
            with prompt "{_escape(message)}" ¬
            default name "{_escape(default_name)}" ¬
            default location POSIX file "{_escape(folder)}"
        return POSIX path of result
    on error
        return ""
    end try
    """
    raw = _run_applescript(script).strip()
    return raw if raw else None


def folder_select(
    message: str = "フォルダを選択してください",
    start_folder: str = "~",
    multiple: bool = False,
) -> Optional[Union[str, list[str]]]:
    """
    フォルダ選択ダイアログ。

    Returns
    -------
    選択されたフォルダパス（or リスト）。キャンセル時は None。
    """
    folder = os.path.expanduser(start_folder)
    multi_str = "with multiple selections allowed" if multiple else ""
    script = f"""
    try
        set result to choose folder ¬
            with prompt "{_escape(message)}" ¬
            default location POSIX file "{_escape(folder)}" ¬
            {multi_str}
        if class of result is list then
            set paths to ""
            repeat with f in result
                set paths to paths & POSIX path of f & linefeed
            end repeat
            return paths
        else
            return POSIX path of result
        end if
    on error
        return ""
    end try
    """
    raw = _run_applescript(script).strip()
    if not raw:
        return None
    paths = [p for p in raw.splitlines() if p]
    if multiple:
        return paths
    return paths[0] if paths else None


def notify(
    message: str,
    title: str = "通知",
    subtitle: str = "",
    sound: str = "default",
) -> None:
    """
    macOS 通知センターに通知を送る。

    Parameters
    ----------
    sound: サウンド名（"default", "Ping", "Glass", "" で無音）
    """
    sub_str = f'subtitle "{_escape(subtitle)}"' if subtitle else ""
    sound_str = f'sound name "{_escape(sound)}"' if sound else ""
    script = f"""
    display notification "{_escape(message)}" ¬
        with title "{_escape(title)}" ¬
        {sub_str} ¬
        {sound_str}
    """
    _run_applescript(script)

"""
macos_gif_dialog.py
--------------------
GIF アニメーション・テキスト・ボタンを持つ macOS ネイティブダイアログ。
PyObjC (AppKit) を使用。tkinter 不使用。

インストール:
    pip install pyobjc-framework-Cocoa

使い方:
    from macos_gif_dialog import gif_dialog, gif_progress_dialog

    # 確認ダイアログ
    clicked = gif_dialog(
        gif_path="spinner.gif",
        message="サーバーに接続しています...",
        title="接続中",
        buttons=["再試行", "キャンセル"],
        cancel_label="キャンセル",
    )
    print(clicked)  # "再試行" or "キャンセル" or None

    # バックグラウンド処理 + 自動で閉じる
    def heavy_task():
        import time; time.sleep(5)

    gif_progress_dialog(
        gif_path="loading.gif",
        message="データを処理しています...",
        task=heavy_task,
    )
"""

import os
import sys
import threading
from typing import Optional

try:
    import objc
    from AppKit import (
        NSApplication, NSApp,
        NSPanel, NSWindow,
        NSImageView, NSImage,
        NSTextField, NSButton,
        NSColor, NSFont,
        NSView, NSVisualEffectView,
        NSStackView,
        NSUserInterfaceLayoutOrientationVertical,
        NSUserInterfaceLayoutOrientationHorizontal,
        NSStackViewGravityTop,
        NSStackViewGravityCenter,
        NSStackViewGravityBottom,
        NSTextAlignmentCenter,
        NSTitledWindowMask,
        NSClosableWindowMask,
        NSResizableWindowMask,
        NSHUDWindowMask,
        NSUtilityWindowMask,
        NSFloatingWindowLevel,
        NSBackingStoreBuffered,
        NSMakeRect, NSMakeSize,
        NSRunLoop, NSDate,
        NSEvent,
        NSApplicationActivationPolicyAccessory,
        NSApplicationActivationPolicyRegular,
        NSObject,
        NSWindowStyleMaskTitled,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskResizable,
        NSWindowStyleMaskFullSizeContentView,
        NSWindowStyleMaskUtilityWindow,
        NSWindowStyleMaskHUDWindow,
        NSBezelStyleRounded,
        NSBezelStyleRegularSquare,
        NSMomentaryLightButton,
        NSLineBreakByWordWrapping,
        NSImageFrameNone,
        NSImageScaleProportionallyUpOrDown,
        NSLayoutAttributeCenterX,
        NSLayoutAttributeCenterY,
        NSLayoutAttributeWidth,
        NSLayoutAttributeHeight,
        NSLayoutAttributeTop,
        NSLayoutAttributeBottom,
        NSLayoutAttributeLeft,
        NSLayoutAttributeRight,
        NSLayoutRelationEqual,
        NSLayoutConstraint,
        NSWindowStyleMaskNonactivatingPanel,
        NSTextAlignmentLeft,
        NSTextAlignmentRight,
        NSCursor,
    )
    from Foundation import NSMakeRect, NSMakeSize, NSObject, NSRunLoop, NSDate
    HAS_PYOBJC = True
except ImportError as e:
    HAS_PYOBJC = False
    _IMPORT_ERROR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Kivy 処理中オーバーレイ（HUD パネル・メインウィンドウ追従）
# ─────────────────────────────────────────────────────────────────────────────

if HAS_PYOBJC:
    class MacOSProcessingOverlay:
        """
        標準に近いフローティング HUD。define.APPNAME と一致するタイトルのウィンドウ中央に重ね、
        update() ごとに位置を同期する（マルチディスプレイ／ウィンドウ移動に追従）。
        """

        # _pump_runloop で configuring 中も配送してよい「ハウスキーピング」イベント種別。
        # ここに無いイベント（マウス／キーボード／スクロール／ジェスチャ等、ユーザー入力全般）は
        # 宛先ウィンドウに関わらず配送しない＝処理中は裏の操作を受け付けない。
        _HOUSEKEEPING_EVENT_TYPES = frozenset(
            v for v in (
                getattr(AppKit, name, None)
                for name in (
                    "NSEventTypeAppKitDefined",
                    "NSEventTypeApplicationDefined",
                    "NSEventTypePeriodic",
                    "NSEventTypeSystemDefined",
                    "NSEventTypeCursorUpdate",
                )
            ) if v is not None
        )

        def __init__(
            self,
            gif_path: str,
            app_name: str,
            message: str = "Processing...",
        ):
            self._app_name = app_name
            self._gif_path = os.path.abspath(gif_path)
            self._message = message
            self._win = None
            self._gif_view = None
            self._main_label = None
            self._sub_label = None
            self._pending_sub = None
            self._parent_win = None
            self._panel_w = 268.0
            self._panel_h = 108.0
            self._built = False

        def _ensure_shared_app(self):
            app = NSApplication.sharedApplication()
            # NSApplicationActivationPolicyProhibited == 2
            if app.activationPolicy() == 2:
                app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        def _build(self):
            if self._built:
                return
            self._ensure_shared_app()
            style = NSWindowStyleMaskHUDWindow | NSWindowStyleMaskNonactivatingPanel
            self._win = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, self._panel_w, self._panel_h),
                style,
                NSBackingStoreBuffered,
                False,
            )
            self._win.setFloatingPanel_(False)
            self._win.setWorksWhenModal_(True)
            self._win.setLevel_(getattr(AppKit, "NSNormalWindowLevel", 0))
            self._win.setAlphaValue_(0.92)
            self._win.setTitle_("")
            self._win.setReleasedWhenClosed_(False)
            self._win.setHidesOnDeactivate_(True)
            self._win.setBecomesKeyOnlyIfNeeded_(True)
            cb = getattr(AppKit, "NSWindowCollectionBehaviorTransient", 0)
            if cb:
                try:
                    self._win.setCollectionBehavior_(cb)
                except Exception:
                    # Cocoa側のベストエフォート設定（失敗してもパネル自体は使える）
                    pass

            content = self._win.contentView()
            img = NSImage.alloc().initWithContentsOfFile_(self._gif_path)
            if img is None:
                img = NSImage.alloc().initWithSize_(NSMakeSize(100, 100))

            self._gif_view = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 100))
            self._gif_view.setImage_(img)
            self._gif_view.setImageFrameStyle_(NSImageFrameNone)
            self._gif_view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
            self._gif_view.setAnimates_(True)
            self._gif_view.setTranslatesAutoresizingMaskIntoConstraints_(False)

            self._main_label = NSTextField.labelWithString_(self._message)
            self._main_label.setAlignment_(NSTextAlignmentLeft)
            self._main_label.setFont_(NSFont.systemFontOfSize_weight_(14.0, 0.3))
            self._main_label.setLineBreakMode_(NSLineBreakByWordWrapping)
            self._main_label.setMaximumNumberOfLines_(2)
            self._main_label.setTranslatesAutoresizingMaskIntoConstraints_(False)

            self._sub_label = NSTextField.labelWithString_("")
            self._sub_label.setAlignment_(NSTextAlignmentRight)
            self._sub_label.setFont_(NSFont.systemFontOfSize_(10.0))
            self._sub_label.setTextColor_(NSColor.secondaryLabelColor())
            self._sub_label.setLineBreakMode_(NSLineBreakByWordWrapping)
            self._sub_label.setMaximumNumberOfLines_(2)
            self._sub_label.setTranslatesAutoresizingMaskIntoConstraints_(False)

            text_stack = NSStackView.stackViewWithViews_([self._main_label, self._sub_label])
            text_stack.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
            text_stack.setSpacing_(4.0)
            text_stack.setAlignment_(NSLayoutAttributeLeft)
            text_stack.setTranslatesAutoresizingMaskIntoConstraints_(False)

            h_stack = NSStackView.stackViewWithViews_([self._gif_view, text_stack])
            h_stack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
            h_stack.setSpacing_(12.0)
            h_stack.setAlignment_(NSLayoutAttributeCenterY)
            h_stack.setEdgeInsets_((12.0, 14.0, 12.0, 14.0))
            h_stack.setTranslatesAutoresizingMaskIntoConstraints_(False)

            content.addSubview_(h_stack)

            constraints = [
                h_stack.leadingAnchor().constraintEqualToAnchor_(content.leadingAnchor()),
                h_stack.trailingAnchor().constraintEqualToAnchor_(content.trailingAnchor()),
                h_stack.topAnchor().constraintEqualToAnchor_(content.topAnchor()),
                h_stack.bottomAnchor().constraintEqualToAnchor_(content.bottomAnchor()),
                self._gif_view.widthAnchor().constraintEqualToConstant_(100.0),
                self._gif_view.heightAnchor().constraintEqualToConstant_(100.0),
            ]
            for c in constraints:
                c.setActive_(True)

            self._built = True

        def _attach_to_main_window(self, main):
            if not self._win:
                return
            if main is self._parent_win:
                return
            if self._parent_win is not None:
                try:
                    self._parent_win.removeChildWindow_(self._win)
                except Exception:
                    # 既にデタッチ済み等のCocoa側ベストエフォート
                    pass
            self._parent_win = main
            if main is not None:
                try:
                    main.addChildWindow_ordered_(
                        self._win,
                        getattr(AppKit, "NSWindowAbove", 1),
                    )
                except Exception:
                    self._parent_win = None

        def _sync_frame_to_main_window(self):
            if not self._win:
                return
            pw, ph = self._panel_w, self._panel_h
            main = get_window(self._app_name)
            self._attach_to_main_window(main)
            if main is not None:
                mf = main.frame()
                x = mf.origin.x + (mf.size.width - pw) * 0.5
                y = mf.origin.y + (mf.size.height - ph) * 0.5
            else:
                vf = NSScreen.mainScreen().visibleFrame()
                x = vf.origin.x + (vf.size.width - pw) * 0.5
                y = vf.origin.y + (vf.size.height - ph) * 0.5
            self._win.setFrame_display_(NSMakeRect(x, y, pw, ph), True)

        def _pump_runloop(self):
            """
            NSRunLoop のタイマー処理に加え、キューに溜まったイベントを短く処理する。
            メインスレッドが AppKit を十分に回さないとシステムが待ちカーソル（ビーチボール）を出すため。

            mask=NSEventMaskAny で吸い出したイベントを無条件に sendEvent_ すると、
            裏の Kivy/SDL メインウィンドウ宛てのクリック等まで配送されてしまい、
            処理中ダイアログ表示中でも下のボタンが操作できてしまう（dequeue=True で
            キューから抜いた時点でイベントは消費されるため、転送しないイベントは
            そのまま捨てる＝実質的にブロックする）。

            ウィンドウの一致判定（event.window() is self._win）だけに頼ると、PyObjC の
            ブリッジ越しに得られるラッパーオブジェクトの同一性が信頼できない場合があり、
            実際に「短い1回目は効くが、時間のかかる2回目は効かない」という再現が報告された。
            そのため判定の主軸をイベント種別に置く：マウス/キーボード/スクロール/ジェスチャ
            等のユーザー入力イベントは、宛先ウィンドウに関わらず一律に配送しない。
            """
            app = NSApplication.sharedApplication()
            try:
                NSCursor.arrowCursor().set()
            except Exception:
                # カーソル設定のCocoa側ベストエフォート
                pass
            try:
                import Foundation

                mode = getattr(
                    Foundation,
                    "NSDefaultRunLoopMode",
                    "kCFRunLoopDefaultMode",
                )
                mask = getattr(AppKit, "NSEventMaskAny", 0xFFFFFFFFFFFFFFFF)
                past = NSDate.distantPast()
                for _ in range(48):
                    event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                        mask,
                        past,
                        mode,
                        True,
                    )
                    if event is None:
                        break
                    try:
                        event_type = event.type()
                    except Exception:
                        # イベント種別が取得できないCocoa側ベストエフォート（未転送として扱う）
                        event_type = None
                    if event_type in self._HOUSEKEEPING_EVENT_TYPES:
                        app.sendEvent_(event)
                        continue
                    # Any other event type is user input (mouse/keyboard/scroll/
                    # gesture/etc.). Our overlay has no interactive controls, so
                    # forward it only if it is somehow targeted at our own panel;
                    # otherwise drop it (do not forward to the main app window).
                    try:
                        event_window = event.window()
                    except Exception:
                        # イベントの宛先ウィンドウが取得できないCocoa側ベストエフォート（転送しない）
                        event_window = None
                    if event_window is self._win:
                        app.sendEvent_(event)
            except Exception:
                # イベントポンプ全体のベストエフォート（失敗してもオーバーレイ表示は継続）
                pass
            try:
                until = NSDate.dateWithTimeIntervalSinceNow_(0.02)
                NSRunLoop.currentRunLoop().runUntilDate_(until)
                app.updateWindows()
            except Exception:
                # ランループ処理のCocoa側ベストエフォート
                pass

        def _apply_pending(self):
            if self._pending_sub is not None and self._sub_label is not None:
                self._sub_label.setStringValue_(self._pending_sub or "")
                self._pending_sub = None

        def show(self):
            self._build()
            self._sync_frame_to_main_window()
            self._apply_pending()
            self._win.orderFront_(None)
            try:
                NSCursor.arrowCursor().set()
            except Exception:
                # カーソル設定のCocoa側ベストエフォート
                pass

        def update(self):
            if not self._built:
                return
            self._sync_frame_to_main_window()
            self._apply_pending()
            self._pump_runloop()

        def hide(self):
            if self._win:
                if self._parent_win is not None:
                    try:
                        self._parent_win.removeChildWindow_(self._win)
                    except Exception:
                        # 既にデタッチ済み等のCocoa側ベストエフォート
                        pass
                    self._parent_win = None
                self._win.orderOut_(None)

        def set_text(self, text):
            """右下サブテキスト（メインスレッドの update で反映）。"""
            self._pending_sub = text


# ─────────────────────────────────────────────────────────────────────────────
# ボタンクリックを受け取る Delegate
# ─────────────────────────────────────────────────────────────────────────────

if HAS_PYOBJC:
    class _ButtonTarget(NSObject):
        """ボタンのアクションターゲット。"""

        def init(self):
            self = objc.super(_ButtonTarget, self).init()
            if self is None:
                return None
            self._result = None
            self._dialog = None
            return self

        def setResult_dialog_(self, result, dialog):
            self._result = result
            self._dialog = dialog

        def handleClick_(self, sender):
            label = sender.title()
            self._result = label
            if self._dialog:
                self._dialog.close_with_result(label)


# ─────────────────────────────────────────────────────────────────────────────
# ダイアログクラス
# ─────────────────────────────────────────────────────────────────────────────

if HAS_PYOBJC:
    class _GifDialogWindow:
        """
        GIF + テキスト + ボタン群を持つ NSPanel ベースのダイアログ。
        """

        def __init__(
            self,
            gif_path: str,
            message: str,
            title: str = "",
            detail: str = "",
            buttons: list = None,
            cancel_label: str = "キャンセル",
            gif_size: tuple = (150, 150),
            window_width: int = 360,
        ):
            self._result = None
            self._closed = False
            self._buttons_config = buttons or ["OK", cancel_label]
            self._cancel_label = cancel_label

            # ── NSApplication セットアップ ──────────────────────────────────
            self._app = NSApplication.sharedApplication()
            self._app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            self._app.activateIgnoringOtherApps_(True)

            # ── ウィンドウ ────────────────────────────────────────────────
            style = (
                NSWindowStyleMaskTitled |
                NSWindowStyleMaskClosable
            )
            self._win = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, window_width, 100),
                style,
                NSBackingStoreBuffered,
                False,
            )
            self._win.setTitle_(title or "")
            self._win.setLevel_(NSFloatingWindowLevel)
            self._win.center()
            self._win.setReleasedWhenClosed_(False)

            # ── 背景ビュー ────────────────────────────────────────────────
            content = self._win.contentView()

            # ── GIF ImageView ─────────────────────────────────────────────
            gif_w, gif_h = gif_size
            img = NSImage.alloc().initWithContentsOfFile_(
                os.path.abspath(gif_path)
            )
            self._gif_view = NSImageView.alloc().initWithFrame_(
                NSMakeRect(0, 0, gif_w, gif_h)
            )
            self._gif_view.setImage_(img)
            self._gif_view.setImageFrameStyle_(NSImageFrameNone)
            self._gif_view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
            self._gif_view.setAnimates_(True)  # GIF アニメーション ON
            self._gif_view.setTranslatesAutoresizingMaskIntoConstraints_(False)

            # ── メッセージラベル ──────────────────────────────────────────
            self._msg_label = NSTextField.labelWithString_(message)
            self._msg_label.setAlignment_(NSTextAlignmentCenter)
            self._msg_label.setFont_(NSFont.systemFontOfSize_weight_(14, 0.0))
            self._msg_label.setTextColor_(NSColor.labelColor())
            self._msg_label.setLineBreakMode_(NSLineBreakByWordWrapping)
            self._msg_label.setMaximumNumberOfLines_(0)
            self._msg_label.setTranslatesAutoresizingMaskIntoConstraints_(False)

            # ── 詳細ラベル（任意）────────────────────────────────────────
            self._detail_label = None
            if detail:
                self._detail_label = NSTextField.labelWithString_(detail)
                self._detail_label.setAlignment_(NSTextAlignmentCenter)
                self._detail_label.setFont_(NSFont.systemFontOfSize_(11))
                self._detail_label.setTextColor_(NSColor.secondaryLabelColor())
                self._detail_label.setLineBreakMode_(NSLineBreakByWordWrapping)
                self._detail_label.setMaximumNumberOfLines_(0)
                self._detail_label.setTranslatesAutoresizingMaskIntoConstraints_(False)

            # ── ボタン群 ──────────────────────────────────────────────────
            self._targets = []
            self._btn_views = []
            for label in self._buttons_config:
                btn = NSButton.buttonWithTitle_target_action_("", None, None)
                btn.setTitle_(label)
                btn.setBezelStyle_(NSBezelStyleRounded)
                btn.setTranslatesAutoresizingMaskIntoConstraints_(False)
                if label == cancel_label:
                    btn.setKeyEquivalent_("\x1b")  # Escape キー
                else:
                    btn.setKeyEquivalent_("\r")    # Return キー（最初の非キャンセルボタン）

                target = _ButtonTarget.alloc().init()
                target.setResult_dialog_(label, self)
                btn.setTarget_(target)
                btn.setAction_(objc.selector(
                    target.handleClick_,
                    signature=b"v@:@"
                ))
                self._targets.append(target)
                self._btn_views.append(btn)

            # ── 水平ボタンスタック ────────────────────────────────────────
            self._btn_stack = NSStackView.stackViewWithViews_(self._btn_views)
            self._btn_stack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
            self._btn_stack.setSpacing_(10)
            self._btn_stack.setTranslatesAutoresizingMaskIntoConstraints_(False)

            # ── 垂直メインスタック ────────────────────────────────────────
            stack_items = [self._gif_view, self._msg_label]
            if self._detail_label:
                stack_items.append(self._detail_label)
            stack_items.append(self._btn_stack)

            self._vstack = NSStackView.stackViewWithViews_(stack_items)
            self._vstack.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
            self._vstack.setSpacing_(14)
            self._vstack.setEdgeInsets_((24, 24, 24, 24))  # top, left, bottom, right
            self._vstack.setTranslatesAutoresizingMaskIntoConstraints_(False)

            content.addSubview_(self._vstack)

            # ── レイアウト制約 ────────────────────────────────────────────
            constraints = [
                # スタックを content に貼り付け
                self._vstack.leadingAnchor().constraintEqualToAnchor_(
                    content.leadingAnchor()),
                self._vstack.trailingAnchor().constraintEqualToAnchor_(
                    content.trailingAnchor()),
                self._vstack.topAnchor().constraintEqualToAnchor_(
                    content.topAnchor()),
                self._vstack.bottomAnchor().constraintEqualToAnchor_(
                    content.bottomAnchor()),
                # GIF サイズ固定
                self._gif_view.widthAnchor().constraintEqualToConstant_(gif_w),
                self._gif_view.heightAnchor().constraintEqualToConstant_(gif_h),
                # ウィンドウ幅固定
                self._win.contentView().widthAnchor().constraintEqualToConstant_(
                    float(window_width)),
            ]
            for c in constraints:
                c.setActive_(True)

        def close_with_result(self, result: str):
            self._result = result
            self._closed = True
            self._win.close()
            self._app.stop_(None)

        def run(self) -> Optional[str]:
            self._win.makeKeyAndOrderFront_(None)
            self._app.run()
            return self._result

        def close(self):
            """外部から閉じる（progress_dialog 用）。"""
            if not self._closed:
                self._closed = True
                self._win.close()
                self._app.stop_(None)

        def update_message(self, text: str):
            """メッセージを動的に更新する。"""
            self._msg_label.setStringValue_(text)


# ─────────────────────────────────────────────────────────────────────────────
# 公開関数
# ─────────────────────────────────────────────────────────────────────────────

def gif_dialog(
    gif_path: str,
    message: str,
    title: str = "",
    detail: str = "",
    buttons: list = None,
    cancel_label: str = "キャンセル",
    gif_size: tuple = (150, 150),
    window_width: int = 360,
) -> Optional[str]:
    """
    GIF アニメーション付きダイアログを表示する。

    Parameters
    ----------
    gif_path     : GIF ファイルのパス
    message      : 大きく表示するメッセージ
    title        : ウィンドウタイトルバーの文字列
    detail       : メッセージ下の小さい補足テキスト（省略可）
    buttons      : ボタンラベルのリスト（例: ["OK", "キャンセル"]）
    cancel_label : Escape キーに割り当てるボタン
    gif_size     : GIF の表示サイズ (width, height) ピクセル
    window_width : ウィンドウ幅

    Returns
    -------
    押されたボタンのラベル文字列。ウィンドウを閉じた場合は None。
    """
    _require_pyobjc()
    dlg = _GifDialogWindow(
        gif_path=gif_path,
        message=message,
        title=title,
        detail=detail,
        buttons=buttons or ["OK", cancel_label],
        cancel_label=cancel_label,
        gif_size=gif_size,
        window_width=window_width,
    )
    return dlg.run()


def gif_progress_dialog(
    gif_path: str,
    message: str,
    task: callable,
    title: str = "処理中",
    detail: str = "",
    cancel_label: str = "キャンセル",
    gif_size: tuple = (120, 120),
    window_width: int = 340,
    on_cancel: callable = None,
) -> bool:
    """
    バックグラウンドタスクが完了するまで GIF 付きダイアログを表示する。

    Parameters
    ----------
    task         : バックグラウンドで実行する関数（引数なし）
    on_cancel    : キャンセルボタンが押されたときに呼ぶ関数（省略可）

    Returns
    -------
    True: タスク完了, False: キャンセル
    """
    _require_pyobjc()

    dlg = _GifDialogWindow(
        gif_path=gif_path,
        message=message,
        title=title,
        detail=detail,
        buttons=[cancel_label],
        cancel_label=cancel_label,
        gif_size=gif_size,
        window_width=window_width,
    )

    cancelled = threading.Event()

    def run_task():
        try:
            task()
        finally:
            if not cancelled.is_set():
                # メインスレッドで close
                dlg.close()

    t = threading.Thread(target=run_task, daemon=True)
    t.start()

    result = dlg.run()

    if result == cancel_label:
        cancelled.set()
        if on_cancel:
            on_cancel()
        return False

    t.join(timeout=0)
    return True


def _require_pyobjc():
    if not HAS_PYOBJC:
        raise ImportError(
            "PyObjC が必要です。インストール方法:\n"
            "    pip install pyobjc-framework-Cocoa\n\n"
            f"詳細: {_IMPORT_ERROR if not HAS_PYOBJC else ''}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# デモ
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import urllib.request, tempfile, time

    print("サンプル GIF をダウンロード中...")
    gif_url = "https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif"
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    try:
        urllib.request.urlretrieve(gif_url, tmp.name)
    except Exception:
        # オフラインの場合は最小限の 1x1 GIF を生成
        tmp.write(bytes([
            0x47,0x49,0x46,0x38,0x39,0x61,0x01,0x00,
            0x01,0x00,0x80,0x00,0x00,0xff,0xff,0xff,
            0x00,0x00,0x00,0x21,0xf9,0x04,0x00,0x0a,
            0x00,0x00,0x00,0x2c,0x00,0x00,0x00,0x00,
            0x01,0x00,0x01,0x00,0x00,0x02,0x02,0x4c,
            0x01,0x00,0x3b
        ]))
    tmp.close()

    # ── デモ 1: 確認ダイアログ ────────────────────────────────────────────
    print("\n1. gif_dialog() を表示します...")
    result = gif_dialog(
        gif_path=tmp.name,
        message="サーバーに接続できません",
        title="接続エラー",
        detail="ネットワーク設定を確認してから再試行してください。",
        buttons=["再試行", "キャンセル"],
        cancel_label="キャンセル",
        gif_size=(140, 140),
    )
    print(f"   → クリックされたボタン: {result}")

    # ── デモ 2: プログレスダイアログ ─────────────────────────────────────
    print("\n2. gif_progress_dialog() を表示します（3秒で自動終了）...")

    def fake_task():
        time.sleep(3)

    ok = gif_progress_dialog(
        gif_path=tmp.name,
        message="データをアップロード中...",
        title="送信中",
        detail="しばらくお待ちください",
        task=fake_task,
        on_cancel=lambda: print("   → キャンセルされました"),
    )
    print(f"   → 完了: {ok}")

    os.unlink(tmp.name)
    print("\n✅ デモ完了")
