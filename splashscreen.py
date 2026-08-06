"""起動スプラッシュスクリーン（macOS / PyObjC・Cocoa 実装）。

旧実装は tkinter を使っていたが、PyInstaller + Kivy 既定フックは tkinter を
除外するため .app では import 段階で失敗して表示されなかった。処理中 HUD
（macos.MacOSProcessingOverlay）と同じく Cocoa(AppKit) で実装し、ソース実行・
.app の双方で表示できるようにする。PyObjC が無い環境では ImportError を送出し、
呼び出し側（main._display_startup_splash）の try/except で無表示にフォールバックする。
"""

import os

_win = None


def _install_default_menu_bar(app):
    """自前でメニューバー（NSApp.mainMenu）を構築する。

    SDL2(Kivy) の macOS バックエンドは初期化時に ``NSApp == nil`` のときだけ
    メニューバー（Cocoa_CreateApplicationMenus）・activation policy・finishLaunching
    を設定する。スプラッシュが SDL より先に NSApplication を生成するとこの一度きりの
    セットアップが丸ごとスキップされ、.app でメニューバーが出なくなる。これを補うため
    最低限のメニューバー（アプリ／Edit／Window）をここで作る。
    """
    import AppKit
    from AppKit import NSMenu, NSMenuItem
    from Foundation import NSBundle

    if app.mainMenu() is not None:
        return

    # アプリ名（.app は CFBundleName、ソース実行はフォールバック）。
    app_name = None
    try:
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            app_name = info.get("CFBundleName")
    except Exception:
        app_name = None
    if not app_name:
        app_name = "Shade Wave"

    opt = getattr(AppKit, "NSEventModifierFlagOption", 1 << 19)
    cmd = getattr(AppKit, "NSEventModifierFlagCommand", 1 << 20)
    shift = getattr(AppKit, "NSEventModifierFlagShift", 1 << 17)

    main_menu = NSMenu.alloc().init()

    # --- アプリケーションメニュー（先頭項目の submenu が自動的にアプリメニューになる）。
    app_item = NSMenuItem.alloc().init()
    main_menu.addItem_(app_item)
    app_menu = NSMenu.alloc().init()
    app_item.setSubmenu_(app_menu)
    app_menu.addItemWithTitle_action_keyEquivalent_(f"About {app_name}", "orderFrontStandardAboutPanel:", "")
    app_menu.addItem_(NSMenuItem.separatorItem())
    app_menu.addItemWithTitle_action_keyEquivalent_(f"Hide {app_name}", "hide:", "h")
    hide_others = app_menu.addItemWithTitle_action_keyEquivalent_("Hide Others", "hideOtherApplications:", "h")
    hide_others.setKeyEquivalentModifierMask_(opt | cmd)
    app_menu.addItemWithTitle_action_keyEquivalent_("Show All", "unhideAllApplications:", "")
    app_menu.addItem_(NSMenuItem.separatorItem())
    app_menu.addItemWithTitle_action_keyEquivalent_(f"Quit {app_name}", "terminate:", "q")

    # --- Edit メニュー（コピー&ペースト等）。
    edit_item = NSMenuItem.alloc().init()
    main_menu.addItem_(edit_item)
    edit_menu = NSMenu.alloc().initWithTitle_("Edit")
    edit_item.setSubmenu_(edit_menu)
    edit_menu.addItemWithTitle_action_keyEquivalent_("Undo", "undo:", "z")
    redo = edit_menu.addItemWithTitle_action_keyEquivalent_("Redo", "redo:", "z")
    redo.setKeyEquivalentModifierMask_(shift | cmd)
    edit_menu.addItem_(NSMenuItem.separatorItem())
    edit_menu.addItemWithTitle_action_keyEquivalent_("Cut", "cut:", "x")
    edit_menu.addItemWithTitle_action_keyEquivalent_("Copy", "copy:", "c")
    edit_menu.addItemWithTitle_action_keyEquivalent_("Paste", "paste:", "v")
    edit_menu.addItemWithTitle_action_keyEquivalent_("Select All", "selectAll:", "a")

    # --- Window メニュー。
    window_item = NSMenuItem.alloc().init()
    main_menu.addItem_(window_item)
    window_menu = NSMenu.alloc().initWithTitle_("Window")
    window_item.setSubmenu_(window_menu)
    window_menu.addItemWithTitle_action_keyEquivalent_("Minimize", "performMiniaturize:", "m")
    window_menu.addItemWithTitle_action_keyEquivalent_("Zoom", "performZoom:", "")

    app.setMainMenu_(main_menu)
    try:
        app.setWindowsMenu_(window_menu)
    except Exception:
        # Windowメニュー登録のCocoa側ベストエフォート（失敗してもメニューバー自体は表示される）
        pass


def _install_dock_icon(app):
    """Dock アイコンを .app のアイコン（ShadeWave）に明示設定する。

    早期に NSApplication を生成すると applicationIconImage が PyObjC/python の
    デフォルト（ロケット）になり、finishLaunching 時に Dock がそれに化ける。
    バンドルの icns（無ければバンドル自体のアイコン）を読み込んで上書きする。
    """
    from AppKit import NSImage, NSWorkspace
    from Foundation import NSBundle

    bundle = NSBundle.mainBundle()
    icon = None

    # 1) Resources 内の icns を直接読む（.app 実行）。
    try:
        info = bundle.infoDictionary() or {}
        icon_name = info.get("CFBundleIconFile") or "Shade Wave"
        base = os.path.splitext(str(icon_name))[0]
        icns_path = bundle.pathForResource_ofType_(base, "icns")
        if icns_path:
            icon = NSImage.alloc().initWithContentsOfFile_(icns_path)
    except Exception:
        icon = None

    # 2) フォールバック: バンドル（または実行ファイル）自体のアイコン。
    if icon is None:
        try:
            path = bundle.bundlePath()
            if path:
                icon = NSWorkspace.sharedWorkspace().iconForFile_(path)
        except Exception:
            icon = None

    if icon is not None:
        try:
            app.setApplicationIconImage_(icon)
        except Exception:
            # Dockアイコン設定のCocoa側ベストエフォート（失敗してもデフォルトアイコンで動作継続）
            pass


def display_splash_screen(image_path):
    """borderless な最前面ウィンドウに画像を中央表示する。"""
    global _win

    import AppKit
    from AppKit import NSApplication, NSColor, NSImage, NSImageView, NSWindow, NSMakeRect
    from Foundation import NSDate, NSRunLoop

    abs_path = os.path.abspath(image_path)
    image = NSImage.alloc().initWithContentsOfFile_(abs_path)
    if image is None:
        raise FileNotFoundError(f"splash image not found: {abs_path}")

    # 旧 tkinter 実装の subsample(2) 相当（半分のサイズで表示）。
    size = image.size()
    w = max(1.0, float(size.width) / 2.0)
    h = max(1.0, float(size.height) / 2.0)

    # NSApplication を用意。
    # policy が Prohibited(2) だとウィンドウが出ない。一方 Accessory(1) にすると
    # メニューバーと Dock アイコンが消える（.app でメニューバーが出ない不具合の原因）。
    # 通常の前面アプリ＝Regular(0) に設定し、メニューバー／Dock を維持する。
    app = NSApplication.sharedApplication()
    try:
        if app.activationPolicy() != 0:  # NSApplicationActivationPolicyRegular 以外
            app.setActivationPolicy_(getattr(AppKit, "NSApplicationActivationPolicyRegular", 0))
    except Exception:
        # activationPolicy設定のCocoa側ベストエフォート(失敗してもスプラッシュ表示は続行)
        pass

    # finishLaunching より前に Dock アイコンを ShadeWave に固定（python ロケット化け対策）。
    try:
        _install_dock_icon(app)
    except Exception:
        # Dockアイコン設定のCocoa側ベストエフォート(失敗してもデフォルトアイコンで動作継続)
        pass

    style = getattr(AppKit, "NSWindowStyleMaskBorderless", 0)
    backing = getattr(AppKit, "NSBackingStoreBuffered", 2)
    _win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0.0, 0.0, w, h), style, backing, False
    )
    _win.setOpaque_(False)
    _win.setBackgroundColor_(NSColor.clearColor())
    _win.setLevel_(getattr(AppKit, "NSStatusWindowLevel", 25))
    _win.setIgnoresMouseEvents_(True)
    _win.setReleasedWhenClosed_(False)
    cb = getattr(AppKit, "NSWindowCollectionBehaviorTransient", 0)
    if cb:
        try:
            _win.setCollectionBehavior_(cb)
        except Exception:
            # collectionBehavior設定のCocoa側ベストエフォート
            pass

    view = NSImageView.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, w, h))
    view.setImage_(image)
    view.setImageScaling_(getattr(AppKit, "NSImageScaleProportionallyUpOrDown", 3))
    view.setImageFrameStyle_(getattr(AppKit, "NSImageFrameNone", 0))
    _win.setContentView_(view)

    _win.center()
    _win.orderFrontRegardless()

    # Kivy のイベントループ開始前なので、一度ランループを回して確実に描画させる。
    # ここまででスプラッシュを先に画面へ出してから、SDL2 が NSApp 既存を理由に
    # スキップするメニューバー生成・finishLaunching を肩代わりする。順序を後ろに
    # することで、finishLaunching のアプリ活性化がスプラッシュ表示を妨げない。
    try:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
    except Exception:
        # ランループ処理のCocoa側ベストエフォート(失敗しても以降の初期化は続行)
        pass

    try:
        _install_default_menu_bar(app)
    except Exception:
        # メニューバー構築のCocoa側ベストエフォート(失敗してもスプラッシュ表示は続行)
        pass
    try:
        app.finishLaunching()
    except Exception:
        # finishLaunchingのCocoa側ベストエフォート
        pass

    # スプラッシュを最前面に保ち、メニューバーを即時表示させるためアプリを活性化。
    try:
        _win.orderFrontRegardless()
        app.activateIgnoringOtherApps_(True)
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.02))
    except Exception:
        # アプリ活性化のCocoa側ベストエフォート
        pass


def close_splash_screen():
    """スプラッシュウィンドウを閉じる。"""
    global _win
    if _win is not None:
        try:
            _win.orderOut_(None)
            _win.close()
        except Exception:
            # ウィンドウクローズのCocoa側ベストエフォート(失敗しても_winは破棄する)
            pass
        _win = None


# 使用例
if __name__ == "__main__":
    import time

    display_splash_screen("assets/Shade Wave.png")
    for i in range(4):
        print("loop")
        time.sleep(1)
    close_splash_screen()
