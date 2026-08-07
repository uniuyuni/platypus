import os
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ImageTransformMetalBackendTest(unittest.TestCase):
    def test_metal_fit_crop_to_canvas_runs_when_available(self):
        from effect_backends import image_transform_adapter, image_transform_reference

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        previous_area_mode = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_AREA_MODE")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        os.environ["PLATYPUS_IMAGE_TRANSFORM_AREA_MODE"] = "exact"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            rng = np.random.default_rng(123)
            image = rng.random((96, 128, 3), dtype=np.float32)
            kwargs = dict(
                source_rect=(5, 7, 93, 71),
                canvas_width=80,
                canvas_height=60,
                draw_width=78,
                draw_height=60,
                offset_x=1,
                offset_y=0,
                interpolation="area",
            )

            actual = image_transform_adapter.fit_crop_to_canvas(image, **kwargs)
            expected = image_transform_reference.fit_crop_to_canvas(image, **kwargs)

            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-3)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous
            if previous_area_mode is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_AREA_MODE", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_AREA_MODE"] = previous_area_mode

    def test_metal_transform_to_canvas_runs_when_available(self):
        import cv2

        from effect_backends import image_transform_adapter, image_transform_reference

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            rng = np.random.default_rng(456)
            image = rng.random((96, 128, 3), dtype=np.float32)
            matrix = cv2.getRotationMatrix2D((64, 48), 17.0, 1.0)
            matrix[0, 2] += 7.0
            matrix[1, 2] -= 5.0
            kwargs = dict(
                matrix=matrix,
                canvas_width=144,
                canvas_height=120,
                transform_type="affine",
                interpolation="linear",
                border_mode="reflect",
            )

            actual = image_transform_adapter.transform_to_canvas(image, **kwargs)
            expected = image_transform_reference.transform_to_canvas(image, **kwargs)

            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=2.5e-2)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

    def test_metal_transform_crop_to_canvas_runs_when_available(self):
        import cv2

        from effect_backends import image_transform_adapter, image_transform_reference

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            rng = np.random.default_rng(789)
            image = rng.random((128, 192, 3), dtype=np.float32)
            size = 192
            center = (96, 64)
            matrix = cv2.getRotationMatrix2D(center, -11.0, 1.0)
            matrix[0, 2] += (size / 2) - center[0]
            matrix[1, 2] += (size / 2) - center[1]
            kwargs = dict(
                matrix=matrix,
                source_rect=(20, 18, 140, 120),
                transform_width=size,
                transform_height=size,
                canvas_width=96,
                canvas_height=80,
                draw_width=93,
                draw_height=80,
                offset_x=1,
                offset_y=0,
                transform_type="affine",
                interpolation="linear",
                border_mode="reflect",
            )

            actual = image_transform_adapter.transform_crop_to_canvas(image, **kwargs)
            expected = image_transform_reference.transform_crop_to_canvas(image, **kwargs)

            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            diff = np.abs(actual - expected)
            self.assertLess(float(np.mean(diff)), 0.09)
            self.assertLess(float(np.max(diff)), 0.6)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

    def test_metal_transform_crop_to_canvas_area_matches_two_pass_downscale(self):
        import cv2

        from effect_backends import image_transform_adapter, image_transform_reference

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            yy, xx = np.mgrid[0:180, 0:260].astype(np.float32)
            image = np.stack(
                [
                    0.5 + 0.5 * np.sin(xx * 0.21),
                    0.5 + 0.5 * np.cos(yy * 0.19),
                    (((xx.astype(np.int32) // 5) + (yy.astype(np.int32) // 5)) & 1).astype(np.float32),
                ],
                axis=2,
            ).astype(np.float32)
            size = 260
            center = (130, 90)
            matrix = cv2.getRotationMatrix2D(center, -7.0, 1.0)
            matrix[0, 2] += (size / 2) - center[0]
            matrix[1, 2] += (size / 2) - center[1]
            kwargs = dict(
                matrix=matrix,
                source_rect=(18, 22, 210, 170),
                transform_width=size,
                transform_height=size,
                canvas_width=70,
                canvas_height=56,
                draw_width=69,
                draw_height=56,
                offset_x=1,
                offset_y=0,
                transform_type="affine",
                interpolation="area",
                border_mode="reflect",
            )

            actual = image_transform_adapter.transform_crop_to_canvas(image, **kwargs)
            expected = image_transform_reference.transform_crop_to_canvas(image, **kwargs)
            diff = np.abs(actual - expected)

            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertLess(float(np.mean(diff)), 0.035)
            self.assertLess(float(np.percentile(diff, 99)), 0.20)
            self.assertLess(float(np.max(diff)), 0.65)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

    def test_metal_transform_crop_to_canvas_nearest_zoom_runs_when_available(self):
        import cv2

        from effect_backends import image_transform_adapter, image_transform_reference

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            rng = np.random.default_rng(890)
            image = rng.random((96, 144, 3), dtype=np.float32)
            size = 144
            center = (72, 48)
            matrix = cv2.getRotationMatrix2D(center, 9.0, 1.0)
            matrix[0, 2] += (size / 2) - center[0]
            matrix[1, 2] += (size / 2) - center[1]
            kwargs = dict(
                matrix=matrix,
                source_rect=(42, 44, 48, 32),
                transform_width=size,
                transform_height=size,
                canvas_width=96,
                canvas_height=64,
                draw_width=96,
                draw_height=64,
                offset_x=0,
                offset_y=0,
                transform_type="affine",
                interpolation="nearest",
                border_mode="reflect",
            )

            actual = image_transform_adapter.transform_crop_to_canvas(image, **kwargs)
            transformed = image_transform_adapter.transform_to_canvas(
                image,
                matrix,
                size,
                size,
                transform_type="affine",
                interpolation="linear",
                border_mode="reflect",
            )
            expected = image_transform_adapter.fit_crop_to_canvas(
                transformed,
                kwargs["source_rect"],
                kwargs["canvas_width"],
                kwargs["canvas_height"],
                kwargs["draw_width"],
                kwargs["draw_height"],
                kwargs["offset_x"],
                kwargs["offset_y"],
                "nearest",
            )
            diff = np.abs(actual - expected)

            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertLess(float(np.mean(diff)), 1e-5)
            self.assertLess(float(np.max(diff)), 1e-4)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

    def test_metal_transform_crop_to_canvas_lens_strength_runs_when_available(self):
        import cv2

        from effect_backends import image_transform_adapter

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            yy, xx = np.mgrid[0:96, 0:128].astype(np.float32)
            image = np.stack(
                [
                    xx / 127.0,
                    yy / 95.0,
                    (xx + yy) / (127.0 + 95.0),
                ],
                axis=2,
            ).astype(np.float32)
            matrix = np.eye(3, dtype=np.float64)
            kwargs = dict(
                matrix=matrix,
                source_rect=(0, 0, 128, 96),
                transform_width=128,
                transform_height=96,
                canvas_width=128,
                canvas_height=96,
                draw_width=128,
                draw_height=96,
                offset_x=0,
                offset_y=0,
                transform_type="perspective",
                interpolation="linear",
                border_mode="constant",
                lens_strength=30.0,
                lens_scale=1.0,
            )

            actual = image_transform_adapter.transform_crop_to_canvas(image, **kwargs)
            center_x, center_y = 64.0, 48.0
            max_radius = np.sqrt(center_x**2 + center_y**2)
            dx = (xx - center_x) / max_radius
            dy = (yy - center_y) / max_radius
            r2 = dx * dx + dy * dy
            distortion = 1.0 + (30.0 / 200.0) * r2
            map_x = (center_x + dx * distortion * max_radius).astype(np.float32)
            map_y = (center_y + dy * distortion * max_radius).astype(np.float32)
            expected = cv2.remap(
                image,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            diff = np.abs(actual - expected)

            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertLess(float(np.mean(diff)), 0.01)
            self.assertLess(float(np.max(diff)), 0.08)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

    def test_metal_transform_crop_mesh_integer_taps_match_per_tap_bicubic_replica(self):
        """area/nearest の LUT 経路が旧 per-tap bicubic 式と一致することの厳密検証。

        タップが整数キャンバス座標である area/nearest では、LUT 化した bicubic
        mesh 参照は旧実装 (sample_mesh_map_cubic をタップ毎に評価) と同一式のため
        float 丸め誤差 (atol~1e-4) 内で一致しなければならない。
        """
        from effect_backends import image_transform_adapter

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            rng = np.random.default_rng(42)
            th, tw = 80, 120
            image = rng.random((th, tw, 3)).astype(np.float32)
            grid_w, grid_h = 5, 4
            coarse_x = np.linspace(0, tw - 1, grid_w, dtype=np.float32)
            coarse_y = np.linspace(0, th - 1, grid_h, dtype=np.float32)
            base_x, base_y = np.meshgrid(coarse_x, coarse_y)
            # mesh_map_* はキャンバス座標からの「変位」
            mesh_x = (3.0 * np.sin(base_y / (th - 1) * np.pi)).astype(np.float32)
            mesh_y = (2.0 * np.sin(base_x / (tw - 1) * np.pi)).astype(np.float32)

            def cubic_w(x):
                a = -0.75
                x = abs(x)
                if x <= 1.0:
                    return (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0
                if x < 2.0:
                    return a * x**3 - 5.0 * a * x**2 + 8.0 * a * x - 4.0 * a
                return 0.0

            def mesh_cubic(mesh, tx, ty):
                gx = (tx + 0.5) * grid_w / tw - 0.5
                gy = (ty + 0.5) * grid_h / th - 0.5
                ix = int(np.floor(gx))
                iy = int(np.floor(gy))
                acc = 0.0
                ws = 0.0
                for yy in range(-1, 3):
                    sy = min(max(iy + yy, 0), grid_h - 1)
                    wy = cubic_w(gy - (iy + yy))
                    for xx in range(-1, 3):
                        sx = min(max(ix + xx, 0), grid_w - 1)
                        wx = cubic_w(gx - (ix + xx))
                        w = wx * wy
                        acc += float(mesh[sy, sx]) * w
                        ws += w
                if ws != 0.0:
                    return acc / ws
                return float(mesh[min(max(iy, 0), grid_h - 1), min(max(ix, 0), grid_w - 1)])

            def bilinear_const(img, sx, sy):
                x0 = int(np.floor(sx))
                y0 = int(np.floor(sy))
                ax = sx - x0
                ay = sy - y0

                def rd(xx, yy):
                    if xx < 0 or yy < 0 or xx >= tw or yy >= th:
                        return np.zeros(3, dtype=np.float64)
                    return img[yy, xx].astype(np.float64)

                top = rd(x0, y0) * (1 - ax) + rd(x0 + 1, y0) * ax
                bot = rd(x0, y0 + 1) * (1 - ax) + rd(x0 + 1, y0 + 1) * ax
                return top * (1 - ay) + bot * ay

            def tap(txi, tyi):
                mtx = float(txi) + mesh_cubic(mesh_x, float(txi), float(tyi))
                mty = float(tyi) + mesh_cubic(mesh_y, float(txi), float(tyi))
                return bilinear_const(image, mtx, mty)

            dw, dh = 60, 40  # 2x 縮小 → area の footprint は 2x2 タップ
            common = dict(
                matrix=np.eye(3, dtype=np.float64),
                source_rect=(0, 0, tw, th),
                transform_width=tw,
                transform_height=th,
                canvas_width=dw,
                canvas_height=dh,
                draw_width=dw,
                draw_height=dh,
                offset_x=0,
                offset_y=0,
                transform_type="perspective",
                border_mode="constant",
                mesh_map_x=mesh_x,
                mesh_map_y=mesh_y,
            )

            # --- area ---
            actual = image_transform_adapter.transform_crop_to_canvas(
                image, interpolation="area", **common
            )
            expected = np.zeros((dh, dw, 3), dtype=np.float64)
            for dy in range(dh):
                ty0 = dy * th / dh
                ty1 = (dy + 1) * th / dh
                for dx in range(dw):
                    tx0 = dx * tw / dw
                    tx1 = (dx + 1) * tw / dw
                    acc = np.zeros(3, dtype=np.float64)
                    ws = 0.0
                    for yy in range(int(np.floor(ty0)), int(np.ceil(ty1))):
                        wy = max(0.0, min(ty1, yy + 1) - max(ty0, yy))
                        if wy <= 0.0:
                            continue
                        for xx in range(int(np.floor(tx0)), int(np.ceil(tx1))):
                            wx = max(0.0, min(tx1, xx + 1) - max(tx0, xx))
                            w = wx * wy
                            if w <= 0.0:
                                continue
                            acc += tap(xx, yy) * w
                            ws += w
                    expected[dy, dx] = acc / ws if ws > 0.0 else 0.0
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-4)

            # --- nearest ---
            actual_n = image_transform_adapter.transform_crop_to_canvas(
                image, interpolation="nearest", **common
            )
            expected_n = np.zeros((dh, dw, 3), dtype=np.float64)
            for dy in range(dh):
                tyi = min(int(np.floor(dy * th / dh)), th - 1)
                for dx in range(dw):
                    txi = min(int(np.floor(dx * tw / dw)), tw - 1)
                    expected_n[dy, dx] = tap(txi, tyi)
            np.testing.assert_allclose(actual_n, expected_n, rtol=0.0, atol=1e-4)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

    def test_metal_transform_crop_mesh_dense_bake_is_bit_exact(self):
        """PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE=1 (密マップ bake) が off とビット一致すること。

        bake カーネルはメインカーネルの LUT 経路 (mesh_lut_lookup) と同一式を
        整数キャンバス座標で評価するため、構成上ビット一致しなければならない。
        """
        from effect_backends import image_transform_adapter

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        previous_dense = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            rng = np.random.default_rng(7)
            th, tw = 96, 128
            image = rng.random((th, tw, 3)).astype(np.float32)
            grid_w, grid_h = 6, 5
            coarse_x = np.linspace(0, tw - 1, grid_w, dtype=np.float32)
            coarse_y = np.linspace(0, th - 1, grid_h, dtype=np.float32)
            base_x, base_y = np.meshgrid(coarse_x, coarse_y)
            # mesh_map_* はキャンバス座標からの「変位」
            mesh_x = (2.5 * np.sin(base_y / (th - 1) * np.pi)).astype(np.float32)
            mesh_y = (1.5 * np.sin(base_x / (tw - 1) * np.pi)).astype(np.float32)
            common = dict(
                matrix=np.eye(3, dtype=np.float64),
                source_rect=(8, 4, 112, 88),
                transform_width=tw,
                transform_height=th,
                canvas_width=64,
                canvas_height=48,
                draw_width=56,
                draw_height=44,
                offset_x=4,
                offset_y=2,
                transform_type="perspective",
                border_mode="constant",
                mesh_map_x=mesh_x,
                mesh_map_y=mesh_y,
            )

            for interpolation in ("area", "nearest"):
                os.environ["PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE"] = "0"
                off = image_transform_adapter.transform_crop_to_canvas(
                    image, interpolation=interpolation, **common
                )
                os.environ["PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE"] = "1"
                on = image_transform_adapter.transform_crop_to_canvas(
                    image, interpolation=interpolation, **common
                )
                np.testing.assert_allclose(on, off, rtol=0.0, atol=1e-6)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous
            if previous_dense is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE"] = previous_dense

    def test_metal_transform_crop_to_canvas_mesh_map_runs_when_available(self):
        from effect_backends import image_transform_adapter, image_transform_reference

        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        try:
            status = image_transform_adapter.backend_status()
            if status.backend != "effect_backends._image_transform_metal":
                self.skipTest(f"Metal backend is unavailable: {status.detail}")

            yy, xx = np.mgrid[0:80, 0:120].astype(np.float32)
            image = np.stack(
                [
                    xx / 119.0,
                    yy / 79.0,
                    (xx + yy) / (119.0 + 79.0),
                ],
                axis=2,
            ).astype(np.float32)
            grid_w = 5
            grid_h = 4
            coarse_x = np.linspace(0, 119, grid_w, dtype=np.float32)
            coarse_y = np.linspace(0, 79, grid_h, dtype=np.float32)
            base_x, base_y = np.meshgrid(coarse_x, coarse_y)
            # mesh_map_* はキャンバス座標からの「変位」
            mesh_x = (3.0 * np.sin(base_y / 79.0 * np.pi)).astype(np.float32)
            mesh_y = (2.0 * np.sin(base_x / 119.0 * np.pi)).astype(np.float32)
            kwargs = dict(
                matrix=np.eye(3, dtype=np.float64),
                source_rect=(0, 0, 120, 80),
                transform_width=120,
                transform_height=80,
                canvas_width=120,
                canvas_height=80,
                draw_width=120,
                draw_height=80,
                offset_x=0,
                offset_y=0,
                transform_type="perspective",
                interpolation="linear",
                border_mode="constant",
                mesh_map_x=mesh_x,
                mesh_map_y=mesh_y,
            )

            actual = image_transform_adapter.transform_crop_to_canvas(image, **kwargs)
            expected = image_transform_reference.transform_crop_to_canvas(image, **kwargs)
            diff = np.abs(actual - expected)

            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertLess(float(np.mean(diff)), 0.02)
            self.assertLess(float(np.max(diff)), 0.15)
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous


if __name__ == "__main__":
    unittest.main()
