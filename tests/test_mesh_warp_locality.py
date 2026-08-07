import pathlib
import sys
import unittest

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MeshWarpLocalityTest(unittest.TestCase):
    """mesh warp が「動かした CP の周囲だけ」を歪めることの回帰テスト。

    coarse map を full 解像度へ拡大する経路には二つの落とし穴があり、どちらも
    「変形していない領域の直線まで画像全体が波打つ」形で表面化した:

      1. 拡大側 (cv2.resize / Metal shader) は配列 index を half-pixel 中心
         ((j+0.5)*len/n - 0.5) として読むのに、生成側が linspace(0, len-1, n) で
         サンプルしていた → map 全体が線形にずれる (grid_step=32 で ±16px)。
      2. 絶対座標の map を bicubic (Keys a=-0.75) で拡大していた。この kernel は
         1 次多項式を再現できないため、ほぼ identity = 線形ランプである map に
         coarse セル周期の波 (±0.047*grid_step px) が注入される。

    どちらも「変位を拡大して座標を足し戻す + サンプル位置を規約に合わせる」で消える。
    """

    def setUp(self):
        import params
        from cores.distortion_correction import warp_correction

        self.params = params
        self.wc = warp_correction

        self.orig_w, self.orig_h = 1200, 800
        self.size = max(self.orig_w, self.orig_h)
        self.mesh_size = (4, 4)
        self.control_points = {(2, 2): (0.06, 0.0)}
        self.tcg_info = params.param_to_tcg_info({
            'original_img_size': (self.orig_w, self.orig_h),
            'rotation': 0.0,
            'rotation2': 0.0,
            'flip_mode': 0,
            'crop_rect': None,
            'disp_info': (0, 0, self.orig_w, self.orig_h, 1.0),
        })

    def _exact_mls_map(self):
        """coarse 近似なしで full 解像度に MLS を直接評価した「正解」の map。"""
        size = self.size

        class _Dummy:
            shape = (size, size, 3)

        base = self.wc.get_mesh_coordinates((size, size), self.mesh_size)
        rows, cols = self.mesh_size
        src_tcg, dst_tcg = [], []
        for r in range(rows + 1):
            for c in range(cols + 1):
                bx, by = base[r, c]
                off_x, off_y = self.control_points.get((r, c), (0.0, 0.0))
                src_tcg.append((bx, by))
                dst_tcg.append((bx + off_x, by + off_y))
        for px, py in self.wc.outer_ring_pins_tcg():
            src_tcg.append((px, py))
            dst_tcg.append((px, py))

        to_px = lambda p: self.params.tcg_to_ref_image(p[0], p[1], _Dummy(), self.tcg_info)
        src_px = np.array([to_px(p) for p in src_tcg])
        dst_px = np.array([to_px(p) for p in dst_tcg])
        gx, gy = np.meshgrid(
            np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64)
        )
        return self.wc._mls_affine_map(src_px, dst_px, gx, gy)

    def test_coarse_axis_coords_matches_resize_convention(self):
        """coarse_axis_coords が「拡大側が想定するサンプル位置」を返す。

        coarse 配列の 1 点だけを立てて拡大すると、ピークは拡大側がその index に
        割り当てた full 座標に立つ。生成側 (coarse_axis_coords) がそれと同じ位置を
        返していなければ、変形が本来と違う場所に出る。
        """
        width = height = 1200
        grid_w = grid_h = 38
        axis = self.wc.coarse_axis_coords(width, grid_w)
        for j in (5, 19, 31):
            coarse = np.zeros((grid_h, grid_w), dtype=np.float32)
            coarse[:, j] = 1.0
            up = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
            peak = int(np.argmax(up[height // 2]))
            self.assertAlmostEqual(peak, axis[j], delta=1.0)

    def test_upsample_of_zero_displacement_is_exact_identity(self):
        """変位 0 の領域は拡大後も厳密に identity（= 直線が波打たない）。"""
        width, height = 1200, 800
        map_x, map_y = self.wc.upsample_coarse_mesh_map(
            np.zeros((13, 19), dtype=np.float32),
            np.zeros((13, 19), dtype=np.float32),
            width,
            height,
        )
        np.testing.assert_array_equal(map_x, np.arange(width, dtype=np.float32)[None, :].repeat(height, 0))
        np.testing.assert_array_equal(map_y, np.arange(height, dtype=np.float32)[:, None].repeat(width, 1))

    def test_upsampled_map_matches_exact_mls_far_from_control_point(self):
        """coarse → full 拡大後の map が厳密 MLS と一致し、遠方が identity のままである。"""
        size = self.size
        disp = self.wc.calculate_mesh_mls_coarse_map(
            size, size, self.mesh_size, self.control_points,
            tcg_info=self.tcg_info, grid_step=32,
        )
        self.assertIsNotNone(disp)
        map_x, map_y = self.wc.upsample_coarse_mesh_map(disp[0], disp[1], size, size)

        exact_x, exact_y = self._exact_mls_map()
        err = np.hypot(map_x - exact_x, map_y - exact_y)
        self.assertLess(float(err.mean()), 0.05)
        # coarse 近似の誤差は変形の急な中心部だけに残り、遠方はサブピクセル。
        self.assertLess(float(err[:, :200].max()), 0.2)

        # 動かした CP (画像中央) から 2 セル以上離れた左端の帯は、厳密 MLS 自身が
        # ほぼ identity。修正前はここに ±18px の線形ずれと波が乗っていた。
        ident_x = np.arange(size, dtype=np.float32)[None, :]
        ident_y = np.arange(size, dtype=np.float32)[:, None]
        band = np.hypot(map_x[:, :200] - ident_x[:, :200], map_y[:, :200] - ident_y)
        self.assertLess(float(band.max()), 2.0)

    def test_warp_mesh_keeps_untouched_region_stable(self):
        """warp_mesh の実効変位も遠方でサブピクセルに収まる。"""
        size = self.size
        img = np.zeros((size, size, 3), dtype=np.float32)
        warped = self.wc.warp_mesh(
            img, self.mesh_size, self.control_points, tcg_info=self.tcg_info
        )
        self.assertEqual(warped.shape, img.shape)

        disp = self.wc.calculate_mesh_mls_coarse_map(
            size, size, self.mesh_size, self.control_points,
            tcg_info=self.tcg_info, grid_step=32,
        )
        # 変位マップ自体が、CP から離れた領域ではほぼ 0。
        coarse_x, coarse_y = disp
        grid_w = coarse_x.shape[1]
        left = slice(0, max(1, grid_w // 6))
        self.assertLess(float(np.abs(coarse_x[:, left]).max()), 1.0)
        self.assertLess(float(np.abs(coarse_y[:, left]).max()), 1.0)


if __name__ == '__main__':
    unittest.main()
