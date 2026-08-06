# Shade Wave 総合コードレビュー

作成: 2026-07-17（前版: 2026-06-22）／ 対象: vendored（`.pixi/` `external/` `build/`）を除く実プロジェクトコード（約60K LOC）
方法: 3 並列の横断調査（アプリ層 ／ cores＋effect_backends ／ widgets＋utils＋helpers＋tests）＋ High/Medium 主要項目はソース直読による行レベル裏取り。
「✅検証済み」= ソースを直接確認した項目。無印は横断調査による報告（位置情報付き、±数行の誤差許容）。

## 優先度サマリ

| # | 重要度 | 種別 | 概要 | 位置 |
|---|--------|------|------|------|
| 1 | 高 ✅ | リーク | Window.mouse_pos バインディングリーク（mask_editor2） | mask_editor2.py:2513,2529,2533 / 2916,2933,2941 / 5167 |
| 2 | 高 ✅ | リーク | 同種リーク（旧 mask_editor、unbind 自体なし） | mask_editor.py:69 |
| 3 | 高 ✅ | リーク | 例外時の共有メモリ未クローズ/未 unlink | async_worker.py:139,196-236 |
| 4 | 高 ✅ | 性能 | キャンセル毎の無条件ワーカープロセス再起動 | async_worker.py:481-508 |
| 5 | 高 ✅ | 競合 | キャッシュ辞書のロックなし read-modify-write | file_cache_system.py:91-104 |
| 6 | 高 ✅ | デッド | 参照ゼロのデッドコード ~800 行＋kv | cmos_to_ccd_converter.py 他 |
| 7 | 高 ✅ | 副作用 | import 時にプロセス全体の警告を抑止 | fringe_removal.py:17 |
| 8 | 高 | 重複 | アダプタ定型 ~450 行が未共通化（旧 D 項の残り） | effect_backends/*_adapter.py |
| 9 | 中 ✅ | 堅牢性 | config.json 破損で起動クラッシュ | config.py:123-133 |
| 10 | 中 ✅ | 正しさ | カーブ系キャッシュキー `hash(np.sum())` の衝突 | effects.py ×8 |
| 11 | 中 | 競合 | 描画フラグ 4 つの非アトミックなハンドオフ | main.py:1880-1957 |
| 12 | 中 ✅ | 堅牢性 | `assert False` によるエラー処理 ×5 | mask_editor2.py |
| 13 | 中 | 正しさ | MergeDebevec に float32 入力（uint8 変換ヘルパ未使用） | exposure_fusion_debevec.py:10,39 |
| 14 | 中 | 正しさ | dtype で 0-255/0-1 を誤判定するレンジトラップ | lens_ghost.py:32-33 |
| 15 | 中 | 堅牢性 | import 時 `sys.exit(1)` | color_resolver.py:31-34 |
| 16 | 中 | 保守性 | ホットパスに恒久 WARNING 診断ログ | content_aware_fill.py:545 |
| 17 | 中 | 性能 | オーバーレイ再描画のフルアロケーション＋テクスチャ churn | mask_editor2.py:4908-4944 |
| 18 | 中 | 堅牢性 | 例外握りつぶし（旧 H 項、未解決） | アプリ層 ~60＋widgets ~95 |
| 19 | 中 | 重複 | genuine 重複群（helpers/effects/history/apply_lut 他） | 複数 |
| 20 | 中 | テスト | テストヘルパ複製 30 ファイル・署名 3 系統（旧 C 項、悪化） | tests/ |
| 21 | 中 | 構造 | god 構造（旧 B 項、悪化: MainWidget 274 メソッド） | main.py / mask_editor2.py 他 |
| 22-32 | 低 | 各種 | 死コード小物・import 副作用・マジックナンバー等 | 下記 |

---

## 1. High — バグ・リーク（実害あり）

> **2026-07-17 追記: 本セクションの #1〜#8 は全て対応済み**（mouse_pos リークは release() フック化、SHM は try/finally + 所有権モデル、restart は in-flight 条件付き、cache は cache_lock 保護、デッドコード 5 ファイル削除、filterwarnings 撤去、アダプタは BackendSelector へ集約）。以下の記述は対応前の記録。

### 1. ✅ Window.mouse_pos バインディングリーク（mask_editor2）
`FreeDrawMask` が `__init__`（mask_editor2.py:2513）と `start()`（2529）で二重 bind し、`end()`（2533）で 1 回しか unbind しない。`PolylineMask` も同構造（2916/2933/2941）。さらに `_remove_mask`（5167）は `end()` も unbind も呼ばないため、削除済みマスクが Window のオブザーバとして残留する。
- 影響: (1) 削除済みマスクが Window 経由で GC されないメモリリーク、(2) マウス移動のたびに過去の全マスクの `on_mouse_pos`（2588-, root 走査＋active 判定）が呼ばれる線形劣化。マスクの作成・削除を繰り返す編集セッションで顕在化。
- 対応: `_remove_mask` で `end()`（または unbind）を必ず呼ぶ。`__init__` と `start()` の二重 bind は片方に整理（Kivy の unbind は 1 回分しか外さない）。

### 2. ✅ 同種リーク（旧 mask_editor）
mask_editor.py:69 で `KVWindow.bind(mouse_pos=...)` するが、unbind がファイル内に存在しない。effects.py:1134/1282 がインペイント UI を開くたびに `MaskEditor` を新規生成し、閉じる時は `remove_widget` のみ。開閉の回数だけインスタンスが Window に残留する。
※ mask_editor.py はインペイント用の現役コード（mask_editor2 とは別責務）。削除不可、リーク修正のみ必要。

### 3. ✅ 共有メモリリーク（async_worker）
`worker_process` 内で `existing_shm`（async_worker.py:139）と `result_shm`（196、`create=True`）を開くが、close は try ブロック末尾（218/225）にある。`make_diff`（182）や queue put で例外が出ると 227 の except へ飛び、両ハンドルが未クローズ。特に `result_shm` は unlink もされず result_queue にも積まれないため SHM 名ごとリークする。エラーが繰り返されると共有メモリ領域を食い潰す。
- 対応: try/finally で確実に close。エラー経路では `result_shm.unlink()` も行う。

### 4. ✅ キャンセル毎の無条件プロセス再起動（async_worker）
`cancel_effect` / `cancel_all`（async_worker.py:481-508）が毎回 `self.restart()`（terminate/kill → join → 新 Process spawn）を実行。スライダー操作等で重いエフェクトのキャンセルが頻発すると macOS の spawn コストを都度払い、体感遅延・CPU スパイクの原因になる。
- 対応: latest_tasks による論理キャンセルは既にあるので、実行中タスクがある時だけ restart する条件分岐を入れる。

### 5. ✅ キャッシュ辞書のロックなし競合（file_cache_system）
`shared_resources['cache']` を ThreadPoolExecutor の完了コールバックスレッドがロックなしで read-modify-write（file_cache_system.py:91-99: 既存 History 取得 → 辞書上書きが非アトミック）。`delete_cache` / `clear_cache`（別スレッド）と競合すると History 取りこぼしやキャッシュ破損が起こり得る。さらに 104 行で `callback(...)` をワーカースレッドから直接呼んでおり、UI 更新経路がメインスレッド外で走る危険がある。
- 対応: cache 辞書を `threading.Lock` で保護。コールバックは Clock 経由でメインスレッドへ。

### 6. ✅ デッドコード（計 ~800 行＋kv）
参照ゼロを grep で確認済み:
- `cores/cmos_to_ccd_converter.py`（391 行）— クラス・モジュール名ともアプリコードから参照なし。
- `cores/painterly_color_mixer.py`（67 行）— 同上。
- `widgets/ghosteditor.py` ＋ `ghosteditor.kv` — 実 UI は ghost_canvas.py の `LensGhostCanvas`（effects.py で lazy import）に移行済み。main.kv:4293 の "GhostEditor" はボタンラベル文字列にすぎない。
- `cores/highlight_recovery.py` — 呼び出し元ゼロ。かつ `hlsrgb` を未 import のまま参照（44-46 行）しており、呼べば NameError（デフォルト `is_enhance_red=True` で必ず通る）。effects.py:14 / imageset.py:27 の import も未使用。
- 対応: 丸ごと削除（highlight_recovery は復活の予定があるなら hlsrgb import 修正＋テスト追加、なければ削除）。

### 7. ✅ グローバル警告抑止（fringe_removal）
fringe_removal.py:17 の `warnings.filterwarnings('ignore')` が effects.py 経由の import 時点でプロセス全体の警告を無効化する。他モジュールの DeprecationWarning・NumPy の RuntimeWarning（overflow/invalid 等）まで消え、障害切り分けを著しく困難にする。
- 対応: 関数内 `warnings.catch_warnings()` に局所化。

### 8. アダプタ定型 ~450 行が未共通化（旧 D 項の残り）
backend_utils.py（46 行）は leaf 6 関数（BackendStatus・optional import・preference・strict・enabled 判定）のみ。各アダプタは依然として個別実装:
`backend_status`×18 / `native_available`×18 / `_backend_preference`×17 / `_metal_device_available`×13（完全同一 7 行）/ `_metal_backend_enabled`×11 / `native_enabled`×9 / `_native_strict`×7。
相違点は (a) 無効値集合の数語、(b) status 説明文字列、(c) CPU/Metal の有無のみ。
- 対応: effect 名・env 名・無効値集合・利用可能バックエンド一覧を注入するパラメータ化基底（`BackendAdapter` クラス or 生成関数）へ集約。概算 ~450 行削減可。最大の削減余地。

---

## 2. Medium — 正しさ・堅牢性

> **2026-07-18 追記: #9〜#18 は対応済み**（config は JSONDecodeError 耐性化、カーブキャッシュキーは点データの tobytes ベースへ（計12箇所）、描画フラグは `_DrawRequest` 不変スナップショット化、`assert False` は全廃、MergeDebevec は uint8 変換経由、lens_ghost は整数 dtype のみ /255、color_resolver は例外送出化、PM_DIAG は debug 降格、`draw_mask_image` はバッファ/テクスチャ/Rectangle 再利用化、型なし except 13箇所を型付き化＋アプリ層の握りつぶしへログ/根拠コメント追加）。#19〜#21 は未着手。以下の記述は対応前の記録。

### 9. ✅ config.json 破損で起動クラッシュ
`load_config`（config.py:123-133）は `except FileNotFoundError` のみ。不正 JSON だと `json.JSONDecodeError` が素通りし、フォールバックなしで起動時にクラッシュする。
- 対応: `except (FileNotFoundError, json.JSONDecodeError)` にして破損時はデフォルト設定で継続＋ログ。

### 10. ✅ カーブ系キャッシュキーの衝突
`param_hash = hash(np.sum(points))` が effects.py の 8 箇所（4203/4228/4253/4279/4553/4576/4599/4622）。点の総和が不変な編集（左右対称に動かす等）で hash が変わらず、LUT が再計算されずに古い diff が残る。
- 対応: `hash(points.tobytes())` 等、点座標そのものに基づくキーへ横断的に変更。

### 11. 描画フラグの非アトミックなハンドオフ
main.py:1880-1913 / 1936-1957。UI スレッドの `start_draw_image` が `apply_draw_image_center` / `apply_draw_fast_display` / `apply_draw_skip_histogram` / `apply_draw_drag_quality` を個別代入してから `draw_event.set()`、描画ワーカーが個別読み取り。version 読み取りと各フラグ読み取りの間に次フレームが割り込むと、version N のフレームに N+1 用のフラグが混ざる。
- 対応: フラグ束を version 付きスナップショット 1 オブジェクトにまとめ、1 回の代入/読み取りに。
- 関連: main.py:1876-1878 の `draw_event.wait()/clear()` はロストウェイクアップの典型形（version 比較で自己修復するため実害は軽微、コメント補強推奨）。

### 12. ✅ `assert False` によるエラー処理 ×5（mask_editor2）
mask_editor2.py:1440/4707/4744/5078/5194。`python -O`（PyInstaller 配布物等）では assert が除去されてエラー経路が沈黙し、後続で不整合クラッシュに化ける。通常実行では graceful degradation ではなくハードクラッシュ。
- 対応: `raise RuntimeError(...)` かログ＋早期 return へ。

### 13. MergeDebevec に float32 入力
exposure_fusion_debevec.py:39 で `simulate_ev` 由来の float32（1.0 超あり）を `merge_debevec.process` にそのまま渡している。uint8 変換ヘルパ `_to_merge_input`（10-13 行）が定義済みなのに一度も呼ばれていない（✅未使用を確認）。`cv2.createMergeDebevec()` は 8bit LDR ブラケット前提のため float 入力は未定義動作。effects.py:948 から実利用される。
- 対応: `_to_merge_input` を通す（uint8 化の抜け漏れの修正）。

### 14. dtype レンジトラップ（lens_ghost）
lens_ghost.py:32-33 `if image.dtype != np.float32: image = image.astype(np.float32) / 255.0`。float64 の [0,1] 画像（numpy 演算の暗黙昇格など）が来ると 255 で割られてほぼ真っ黒になる。公開エントリなので実害あり。
- 対応: 「浮動小数点ならそのまま、整数型なら /255」の判定へ。

### 15. import 時 `sys.exit(1)`（color_resolver）
color_resolver.py:31-34。color_data.json が無い/壊れていると import 段階でプロセスごと即死。
- 対応: 例外送出にして呼び出し側でハンドル。

### 16. 恒久 WARNING 診断ログ（content_aware_fill）
content_aware_fill.py:545 の `logging.warning("[PM_DIAG] ...")` が inpaint 毎に発火。コメントは「一時ログ」。
- 対応: DEBUG 降格か撤去。

### 17. オーバーレイ再描画のフルアロケーション（mask_editor2）
`draw_mask_image`（mask_editor2.py:4908-4944）が毎回 `np.empty((h,w,2))` 新規確保 → 新テクスチャ生成 → フルバッファ `blit_buffer` → 新 `KVRectangle`。部分更新・テクスチャ再利用が一切なく、ドラッグ中の連続更新でテクスチャ churn。
- 対応: バッファ/テクスチャの再利用（サイズ不変ならブリットのみ）。

### 18. 例外握りつぶし（旧 H 項、未解決）
- アプリ層 `except: pass` 約 60 箇所: main.py 12 / macos.py 11 / splashscreen.py 10 / params.py 7 / effects.py 6 / async_worker.py 5 他。期待される競合（FileNotFoundError、Cocoa ベストエフォート）は妥当だが、`except Exception: pass` で UI 状態遷移を握る箇所（main.py:4792-4821 / 5027 / 5269 等）はエラー時に不整合のまま黙って進む。
- widgets 層 broad except 約 95 箇所: mask_editor2 49 / viewer 19 に集中（前版の「mask_editor2 10 箇所」は過少計上だった）。
- 型なし `except:`（KeyboardInterrupt/SystemExit まで飲む）: async_worker.py ×4（133/343/353/369）、effects.py ×2（702/2457）、waitinfo.py ×2（17/30）、export.py:664（ICC 読込失敗を全例外「not found」扱いにし権限/デコードエラーを誤ラベル）。
- 対応: 最低限 `logging.exception` を残し、握りつぶす根拠をコメント化。型なし except は型指定へ。

### 19. genuine 重複
- **helpers（旧 E 項の残り）**: `_soft_edit_mask` / `_ensure_result_size` が nano_banana_helper.py（56,64）/ qwen_image_helper.py（76,96）/ runware_object_eraser_helper.py（122,130）で実質同一。共有 mask_utils モジュールへ集約が即効。`setup`/`predict`/`predict_helper` の共通 I/F は 7 ヘルパに横断するが命名不統一（`setup` vs `setup_scunet` 等）で Protocol 化には署名統一が先。
- ✅ **effects.py:4465-4630**: `HuevsHue/HuevsLum/HuevsSat/LumvsLum/LumvsSat/SatvsLum/SatvsSat` の 7 クラス ~165 行がほぼ同一骨格（差分は param キー・LUT スケール式・合成演算子のみ）。テーブル駆動基底で ~40 行に縮約可。
- **history.py:167/184/213/233**: undo/redo の "All"/"BatchPaste" 分岐で「runtime_special 退避 → clear → deserialize → restore → set2widget_all」の 5 行ブロックが 4 回コピペ。ヘルパへ。
- **`apply_lut` 同名 3 実装**: core.py:663（1D LUT）/ lut_functions.py:315（LUT3D オブジェクト）/ cubelut.py:34（ラッパ）。同名・異シグネチャで誤用を招く。命名整理（`apply_curve_lut` / `apply_lut3d` 等）。
- **`release_ai_model_runtimes` ×2**: memory_manager.py:165 / cores/mask2/inference_runtime.py:288。

### 20. テストヘルパ複製（旧 C 項、悪化）
`_load_class_function` を持つテストファイルは 27 → **30** に増加。`_load_function` 14 / `_function_source` 13 / `_class_source` 4。共有ヘルパ（conftest.py / tests/support）は未導入。さらに `_load_class_function` の署名が 3 系統（`(path, class, func)` 21 本 / `(class, func)` 8 本 / `(func)` 1 本）に分岐しており、単純な複製ではなく契約不一致。根本原因は main.py の ImportBlocker（下記 21）。
- 対応: まず tests/ 共有ヘルパへ一本化＋署名統一。根治は 21 の解決。

### 21. god 構造（旧 B 項、悪化）
- **main.py**: 全体が `if __name__ == '__main__':` 配下＋ImportBlocker（214）で import 不可。`MainWidget` は **274 メソッド**（前版 234 から +40）。UI ウィジェットが numpy / cv2 / `open()` / `os.remove/rename/replace` / `Thread` を直接保持。150 行超関数: `on_fcs_get_file` 197 / `draw_image_core` 190（色変換・zero_wrap・clip・ヒストグラムを直接実行）/ `update_mask2_options_enabled` 144。
- **mask_editor2.py**: 5940 行 / 14 クラス / 367 メソッド。`BaseMask`（424-1141）が edge-refine・brush・serialize・hue/lum/sat 描画・キャッシュを抱える ~60 メソッド基底。`MaskEditor2`（4310-5899）は 108 メソッドの god コントローラ。分離可能な責務: (a) 座標変換群（5830-5897）→ CoordinateMapper、(b) AI 画像キャッシュ → AICacheManager、(c) レイヤ CRUD（4697-5228）→ LayerManager、(d) オーバーレイ描画（5454-5716）→ OverlayRenderer、(e) メモリスイープ（5229-5276）。
- **god 関数**: pipeline.py:939 `process_pipeline`（233 行・**引数 23 個** → 描画コンテキスト dataclass 化）、effects.py `GeometryEffect.make_diff` 256 行 / `AINoiseReductonEffect.make_diff` 218 行、core.py:1803 `adjust_hls_colors` 177 行、light_rays.py:130 `_compute_additive` 192 行 / 667 `_projected_radial_ray` 132 行。
- 対応（段階的、前版と同方針）: ファイル I/O → 履歴 → 画像処理呼び出しの順に `MainWidget` から薄いサービス層へ小さく抽出。

---

## 3. Low — 保守性・小物

> **2026-07-19 追記: #24〜#28 は対応済み**（#24 二重コピー解消+0埋め辞書のヘルパ化 ／ #25 コメント修正・内側prange→range・デッドブロック削除。`rgb2hls` への `@lock_numba` は「実行区間全体のグローバル直列化」という実装のため意図的に見送り、根拠をコメント化 ／ #26 prange→range 化（数値出力はバイト一致を確認）／ #27 crop_editor:203 は調査の結果**コードが正しくコメントが誤り**と判明（FloatLayout は pos_hint なしの子の pos を動かさないため parent.pos が正しい原点）。正確な説明に置換 ／ #28 fringe_removal docstring と color.py 冒頭コメントを実装に一致させた）。既知テスト失敗2件（allow_over_one スタブ・ai_image_cache の sys.path）も解消。#22, #23, #29〜#32 は未着手。

22. ✅ 死コード小物: async_worker.py:154-155 `target_effect = None` 二重代入 / 502-503 未使用 `task` / 108-473 に設計独白コメント 60 行超。main.py:532（空 on_start）/ 1859（旧描画コードのコメントアウト）。core.py:713-725（njit シグネチャ `f4[:,:]` 上到達不能な 3D マスク分岐）/ 701-706, 943-950 コメントアウト残骸。macos.py 末尾の docstring デモ＋`__main__` デモ。
23. import 時副作用: macos.py:224（Cocoa スクリーン列挙）、imageset.py:45（libraw バージョン照会ログ）。テスト・非 GUI 環境で問題になり得る。遅延初期化推奨。
24. memory_manager.py:95-97 `copy_image_for_cache` の二重コピー経路（ascontiguousarray が別配列を返した場合の不要 `.copy()`）。170-206 の 0 埋めフォールバック辞書リテラル ×4 → モジュール定数へ。
25. hlsrgb.py: 133 行「なぜかクラッシュするから prange が使えない」と言いつつ prange 使用の矛盾コメント。`rgb2hls`（10）のみ `@lock_numba` なし＋二重 prange。64-71 の何もしないデッドブロック。
26. prange 誤用: find_bounding_box.py:380（parallel なしの `@jit` で prange = 実質 range）、filters.py:246（逐次依存の内側処理と混在し可読性低下）。
27. crop_editor.py:203「親座標じゃないと X 方向にズレる（バグ？）」— 原因未特定の回避策コメント。crop の scale/translate 追従（200-213）は要調査。
28. docstring/コメント乖離: fringe_removal.py:395-479 の docstring が実装に無い np.gradient/scipy を「使用」と記載（実装は np.diff＋cv2.dilate）。cores/color.py:1「もはや使ってない」は誤り（effects/imageset/main/color_picker/hls_mask の 6 箇所から import される現役）。
29. マジックナンバー: hlsrgb.py C_max=1.5（169,215）/ GAMUT_SOFTCLIP_KNEE=0.95（109）、filters.py:112 の `555`、fringe_removal.py の percentile 群（52/76/86/92 等）— 由来コメントなし。
30. helpers/ の非コード資産: `facer number.png`（286KB）、`tests memo.txt`。docs へ退避か削除。
31. helpers/ri_helper.py（新規）: `sys.path.insert(0, ...)` の恒久汚染、`_CKPT` が cwd 依存の相対パス、import 時即ロード（遅延でない）。重大バグなし。sdxl_helper と同契約で旧 E 項の I/F 重複に該当。
32. viewer.py:1598 キーコード直値（97 等）→ `Keyboard.keycodes` 定数化。ドラッグ状態機械（ヒットテスト・ControlPoint ドラッグ）が crop_editor / mask_editor / distortion_painter / 各 BaseMask で独立実装 → 共通 mixin 化の余地（座標変換自体は params.window_to_tcg に集約済みで良好）。

---

## 4. 前版（2026-06-22）項目の推移

| 項目 | 前版評価 | 現況（2026-07-17） |
|---|---|---|
| A eval() 実行 | 解決 | 解決のまま |
| B main.py god 構造 | 高（大工事） | **悪化**（MainWidget 234→274 メソッド）→ 本版 #21 |
| C テスト文字列依存 | 中 | **悪化**（27→30 ファイル、署名 3 系統）→ 本版 #20 |
| D アダプタ重複 | 一部解決 | 残 ~450 行（最大の削減余地）→ 本版 #8 |
| E helpers 重複 | 中 | 部分残存（`_soft_edit_mask` 等 ×3）→ 本版 #19 |
| F read_pmck_dict 同名 | 解決 | 解決のまま |
| G util 小重複 | 低 | adjust_tone / gaussian_blur_cv / smoothstep は解決。apply_lut ×3・release_ai_model_runtimes ×2 残 → 本版 #19 |
| H except 握りつぶし | 一部対応 | アプリ層 ~60＋widgets ~95 で未解決（前版の mask_editor2「10」は過少計上）→ 本版 #18 |
| I mutable default | 解決 | 解決のまま（distortion_painter.py:500,505 確認済み） |
| AIJobManager wedge | 解決 | 解決のまま（cores/ai_job_manager/ に新規指摘なし） |

健全と確認できたモジュール（参考）: cores/pmck_store.py、cores/ai_image_cache.py、cores/expand_mask.py、utils/viewer_query.py（Kivy 非依存・テスト設計の模範例）、cores/ai_job_manager/。

---

## 推奨着手順

1. **#1-5 リーク・競合系バグ** — mouse_pos unbind（#1,2）は局所修正で即効。SHM try/finally（#3）、restart 条件分岐（#4）、cache ロック化（#5）も小規模。
2. **#6 デッドコード削除** — ~800 行＋kv を安全に削減（highlight_recovery は要判断: 削除 or NameError 修正）。
3. **#7,9,15 グローバル副作用・起動堅牢化** — filterwarnings 局所化、config JSONDecodeError、color_resolver sys.exit。各数行。
4. **#8 アダプタ基底化** — 最大の行数削減（~450 行）。status 生成・native/metal 選択の一元化を慎重に設計。
5. **#10,12,13,14 正しさ系** — キャッシュキー、assert False、MergeDebevec、dtype トラップ。
6. **#19 重複統合 → #18 例外ログ化 → #20 テストヘルパ一本化**。
7. **#21 B 項の段階分割** — 最も効くが大工事。ファイル I/O → 履歴 → 画像処理の順に小さく抽出。mask_editor2 は 5 責務（座標変換 / AI キャッシュ / レイヤ CRUD / オーバーレイ / スイープ）の抽出から。
