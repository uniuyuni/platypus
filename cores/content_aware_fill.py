"""PatchMatch Inpainting with Laplace-based Gradient Fill.

The hole is filled in two stages:

1. ``_laplace_fill`` diffuses smooth color into the hole. This gives the correct
   global color structure (and is all that is needed for flat regions like sky).
   It is used here as the *coarse initializer* for the texture stage.

2. A real multi-scale **PatchMatch** engine (nearest-neighbour field with
   propagation + random search, dense patch-voting reconstruction, coarse-to-fine
   with EM iterations) synthesizes coherent texture/structure into the hole. This
   is what makes the fill work on textured content (grass, foliage, brick, water,
   walls), not just smooth gradients.

Everything runs in PyTorch on the selected device (MPS/CUDA/CPU); no external
dependencies. The public ``content_aware_fill`` entrypoint keeps its previous
signature so callers do not need to change.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from dataclasses import dataclass
import logging
import time

_INF = 1.0e12


@dataclass
class _TensorPackage:
    image: torch.Tensor
    mask: torch.Tensor


class PatchMatchInpainting:
    def __init__(
        self,
        max_working_size: int = 384,
        roi_margin: int = 48,
        speed_mode: str = "ultra",
        verbose: bool = False,
        device: str | None = None,
        texture_patch_size: int = 7,
        texture_strength: float = 1.0,  # retained for API compatibility
        search_iterations: int = 2,     # retained for API compatibility
        match_scale: float = 0.25,      # retained for API compatibility
        candidate_count: int = 64,      # retained for API compatibility
        # --- New PatchMatch engine knobs ---
        nnf_patch_size: int = 7,
        vote_patch_size: int | None = None,
        pm_max_size: int = 512,
        pyramid_min_size: int = 32,
        pm_iters: int = 4,
        em_iters: int = 3, # 作成時は2, 増やすとクオリティが上がるらしい
        random_search_alpha: float = 0.5,
        seed: int | None = 1234,
    ):
        self.max_working_size = max(128, int(max_working_size))
        self.roi_margin = max(8, int(roi_margin))
        self.speed_mode = str(speed_mode)
        self.verbose = bool(verbose)

        # Low-texture (sky) fast path. Compared against range-normalized mean edge
        # magnitude (e/scale) so only genuinely flat ROIs skip PatchMatch; kept well
        # below any real texture to avoid re-introducing blur on textured content.
        self.low_texture_edge_threshold = 0.12
        if self.speed_mode == "quality":
            self.max_working_size = max(self.max_working_size, 512)
            self.low_texture_edge_threshold = 0.10

        # --- PatchMatch engine parameters ---
        self.nnf_patch_size = max(3, int(nnf_patch_size) | 1)  # ensure odd
        if vote_patch_size is None:
            # Derive a sensible voting patch from the requested texture patch size,
            # but keep it small enough to stay sharp and cheap.
            vote_patch_size = min(int(texture_patch_size) | 1, 9)
        self.vote_patch_size = max(3, int(vote_patch_size) | 1)
        self.pm_max_size = max(64, int(pm_max_size))
        self.pyramid_min_size = max(16, int(pyramid_min_size))
        self.pm_iters = max(1, int(pm_iters))
        self.em_iters = max(1, int(em_iters))
        if self.speed_mode == "quality":
            self.em_iters = max(self.em_iters, 3)
        self.random_search_alpha = min(0.9, max(0.1, float(random_search_alpha)))
        self.seed = seed

        # Deprecated / passthrough (kept so old call sites keep working).
        self.texture_strength = max(0.0, min(1.0, float(texture_strength)))

        self.device = self._resolve_device(device)

    def _log(self, msg: str):
        if self.verbose:
            now = time.strftime("%H:%M:%S")
            logging.info("[PatchMatch %s] %s", now, msg)

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is not None:
            d = torch.device(device)
            if d.type == "mps" and not torch.backends.mps.is_available():
                return torch.device("cpu")
            if d.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return d

        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _to_tensor(self, image: np.ndarray, mask: np.ndarray) -> _TensorPackage:
        im_t = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device, non_blocking=True)
        # mask must be boolean for logical operations later (~mask)
        mk_t = (torch.from_numpy(mask) > 0).to(self.device, non_blocking=True)

        return _TensorPackage(
            image=im_t,
            mask=mk_t,
        )

    @staticmethod
    def _to_numpy(output: torch.Tensor, pkg: _TensorPackage) -> np.ndarray:
        out = output.detach().cpu().permute(1, 2, 0).contiguous().numpy()
        return out

    @staticmethod
    def _edge_mag(image: torch.Tensor) -> torch.Tensor:
        y = (0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3]).unsqueeze(0)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device).view(1, 1, 3, 3)
        gx = F.conv2d(y, kx, padding=1)
        gy = F.conv2d(y, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)[0]

    def _guided_filter(self, guide: torch.Tensor, src: torch.Tensor, radius: int = 7, eps: float = 1e-4) -> torch.Tensor:
        """He et al., Guided Image Filtering."""
        def mean_filter(x, r):
             x_pad = F.pad(x, (r, r, r, r), mode='reflect')
             k_size = 2 * r + 1
             kernel = torch.ones((1, 1, k_size, k_size), device=x.device) / (k_size * k_size)
             c = x.shape[1]
             kernel = kernel.repeat(c, 1, 1, 1)
             return F.conv2d(x_pad, kernel, groups=c)

        mean_I = mean_filter(guide, radius)
        mean_p = mean_filter(src, radius)
        mean_Ip = mean_filter(guide * src, radius)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = mean_filter(guide * guide, radius)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = mean_filter(a, radius)
        mean_b = mean_filter(b, radius)

        q = mean_a * guide + mean_b
        return q

    def _laplace_fill(self, image: torch.Tensor, mask: torch.Tensor, max_iter: int = 1000) -> torch.Tensor:
        """
        Solve Laplace equation (Delta u = 0) to fill the hole smoothly.
        Implemented using Multi-Scale approach for better global color propagation.
        """
        if not mask.any():
            return image

        c, h, w = image.shape

        # --- Multi-Scale Initialization ---
        min_size = 64
        if h > min_size and w > min_size:
            scale_factor = 0.125  # 1/8 size

            target_h = max(min_size, int(h * scale_factor))
            target_w = max(min_size, int(w * scale_factor))

            if target_h < h and target_w < w:
                img_small = F.interpolate(image.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)[0]
                mask_small = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='nearest')[0, 0] > 0.5

                prior_small = self._laplace_fill(img_small, mask_small, max_iter=200)

                prior = F.interpolate(prior_small.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)[0]
            else:
                prior = None
        else:
            prior = None

        # --- Base Level Solver ---
        if prior is not None:
             filled = torch.where(mask.unsqueeze(0), prior, image)
        else:
             k_dial = torch.ones((1, 1, 3, 3), device=self.device)
             mask_float = mask.float().unsqueeze(0)
             mask_padded = F.pad(mask_float, (1, 1, 1, 1), mode='replicate')
             boundary_mask = (F.conv2d(mask_padded, k_dial, padding=0) > 0)[0] & (~mask)

             if boundary_mask.any():
                  mean_color = image[:, boundary_mask].mean(dim=1).view(3, 1, 1)
             else:
                  mean_color = image.mean(dim=(1, 2)).view(3, 1, 1)

             image_input = image.unsqueeze(0)
             smooth_boundary_img = self._guided_filter(image_input, image_input, radius=3, eps=1e-3)[0]

             filled = torch.where(mask.unsqueeze(0), mean_color, smooth_boundary_img)

        kernel = torch.tensor([
            [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0]
        ], device=self.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)

        mask_expanded = mask.unsqueeze(0)

        current_iter = 50 if prior is not None else max_iter
        if self.speed_mode == "quality":
             current_iter = min(current_iter, 200)

        for i in range(current_iter):
             filled_padded = F.pad(filled, (1, 1, 1, 1), mode='replicate')
             new_val = F.conv2d(filled_padded, kernel, groups=c)
             filled = torch.where(mask_expanded, new_val, image)

        return filled

    # ------------------------------------------------------------------
    #  PatchMatch engine
    # ------------------------------------------------------------------

    def _valid_source_mask(self, hole: torch.Tensor, patch: int) -> torch.Tensor:
        """Centers whose ``patch x patch`` window contains no hole pixel."""
        half = patch // 2
        pooled = F.max_pool2d(
            hole.float().unsqueeze(0).unsqueeze(0), kernel_size=patch, stride=1, padding=half
        )[0, 0]
        return pooled <= 0.5  # True where window is fully known

    @staticmethod
    def _shift(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Return ``out[..., y, x] = t[..., y - dy, x - dx]`` with replicate borders.

        ``t`` is ``(C, H, W)``.
        """
        if dy == 0 and dx == 0:
            return t
        pb_y, pa_y = max(dy, 0), max(-dy, 0)
        pb_x, pa_x = max(dx, 0), max(-dx, 0)
        x = F.pad(t.unsqueeze(0), (pb_x, pa_x, pb_y, pa_y), mode="replicate")[0]
        h, w = t.shape[-2], t.shape[-1]
        return x[..., pa_y:pa_y + h, pa_x:pa_x + w]

    def _unfold_patches(self, img: torch.Tensor, patch: int) -> torch.Tensor:
        """Return ``(C*patch*patch, H*W)`` reflect-padded patch matrix."""
        half = patch // 2
        padded = F.pad(img.unsqueeze(0), (half, half, half, half), mode="reflect")
        unf = F.unfold(padded, kernel_size=patch)[0]  # (C*patch*patch, H*W)
        return unf.contiguous()

    def _init_nnf(
        self, hole_idx: torch.Tensor, ys_h: torch.Tensor, xs_h: torch.Tensor,
        valid_source: torch.Tensor,
    ) -> torch.Tensor | None:
        """Random valid-source offsets for each hole pixel. Returns ``(2, Lh)`` or None."""
        vy, vx = torch.nonzero(valid_source, as_tuple=True)
        n_valid = vy.numel()
        if n_valid == 0:
            return None
        lh = hole_idx.numel()
        sel = torch.randint(0, n_valid, (lh,), device=self.device)
        off = torch.stack([vy[sel].float() - ys_h.float(), vx[sel].float() - xs_h.float()], dim=0)
        return off

    def _distance(
        self, patches: torch.Tensor, target_cols: torch.Tensor,
        ys_h: torch.Tensor, xs_h: torch.Tensor, cand_off: torch.Tensor,
        valid_flat: torch.Tensor, h: int, w: int,
    ) -> torch.Tensor:
        """Gather-based SSD for a candidate offset field at hole pixels.

        ``patches`` : ``(C*K*K, H*W)`` unfolded current estimate.
        ``target_cols`` : ``(C*K*K, Lh)`` precomputed target patches.
        ``cand_off`` : ``(2, Lh)`` candidate (dy, dx).
        Returns ``(Lh,)`` SSD, ``_INF`` where the source is invalid/out of bounds.
        """
        sy = ys_h + cand_off[0]
        sx = xs_h + cand_off[1]
        oob = (sy < 0) | (sy > h - 1) | (sx < 0) | (sx > w - 1)
        syc = sy.clamp(0, h - 1).long()
        sxc = sx.clamp(0, w - 1).long()
        src_flat = syc * w + sxc
        valid = valid_flat[src_flat] & (~oob)
        src_cols = patches[:, src_flat]
        d = ((target_cols - src_cols) ** 2).sum(dim=0)
        return torch.where(valid, d, torch.full_like(d, _INF))

    def _solve_nnf_level(
        self, cur: torch.Tensor, orig: torch.Tensor, hole: torch.Tensor,
        valid_source: torch.Tensor, off_full: torch.Tensor,
    ) -> torch.Tensor:
        """Run EM (NNF search + voting) at one pyramid level. Returns updated cur."""
        c, h, w = cur.shape
        K = self.nnf_patch_size
        valid_flat = valid_source.reshape(-1)

        hole_flat = hole.reshape(-1)
        hole_idx = torch.nonzero(hole_flat, as_tuple=False).squeeze(1)
        lh = hole_idx.numel()
        if lh == 0:
            return cur

        ys_h = (hole_idx // w).float()
        xs_h = (hole_idx % w).float()

        # working copy of hole offsets
        off_h = off_full.reshape(2, -1)[:, hole_idx].clone()

        prop_steps = [s for s in (1, 2, 4) if s < max(h, w)]
        if not prop_steps:
            prop_steps = [1]
        r_max = float(max(h, w))

        for em in range(self.em_iters):
            patches = self._unfold_patches(cur, K)
            target_cols = patches[:, hole_idx]
            best_d = self._distance(patches, target_cols, ys_h, xs_h, off_h, valid_flat, h, w)

            for _ in range(self.pm_iters):
                # --- Propagation (jump flood) ---
                for s in prop_steps:
                    for dy, dx in ((s, 0), (-s, 0), (0, s), (0, -s)):
                        off_full.reshape(2, -1)[:, hole_idx] = off_h
                        cand = self._shift(off_full, dy, dx).reshape(2, -1)[:, hole_idx]
                        d = self._distance(patches, target_cols, ys_h, xs_h, cand, valid_flat, h, w)
                        better = d < best_d
                        if better.any():
                            best_d = torch.where(better, d, best_d)
                            off_h = torch.where(better.unsqueeze(0), cand, off_h)

                # --- Random search ---
                r = r_max
                while r >= 1.0:
                    ri = int(round(r))
                    rand = torch.randint(-ri, ri + 1, (2, lh), device=self.device).float()
                    cand = off_h + rand
                    d = self._distance(patches, target_cols, ys_h, xs_h, cand, valid_flat, h, w)
                    better = d < best_d
                    if better.any():
                        best_d = torch.where(better, d, best_d)
                        off_h = torch.where(better.unsqueeze(0), cand, off_h)
                    r *= self.random_search_alpha

            off_full.reshape(2, -1)[:, hole_idx] = off_h
            # --- Voting reconstruction from real (known) pixels ---
            cur = self._vote_reconstruct(orig, hole, off_full, self.vote_patch_size)

        return cur

    def _vote_reconstruct(
        self, source_img: torch.Tensor, hole: torch.Tensor, off_full: torch.Tensor, K: int
    ) -> torch.Tensor:
        """Dense patch voting: each hole pixel pastes its matched source patch; average overlaps."""
        c, h, w = source_img.shape
        half = K // 2

        hole_flat = hole.reshape(-1)
        hole_idx = torch.nonzero(hole_flat, as_tuple=False).squeeze(1)
        lh = hole_idx.numel()
        if lh == 0:
            return source_img

        ys_h = (hole_idx // w).long()
        xs_h = (hole_idx % w).long()
        off_h = off_full.reshape(2, -1)[:, hole_idx]
        src_y = (ys_h.float() + off_h[0]).round().clamp(0, h - 1).long()
        src_x = (xs_h.float() + off_h[1]).round().clamp(0, w - 1).long()

        oy, ox = torch.meshgrid(
            torch.arange(-half, half + 1, device=self.device),
            torch.arange(-half, half + 1, device=self.device),
            indexing="ij",
        )
        oy = oy.reshape(-1)
        ox = ox.reshape(-1)

        acc = torch.zeros((c, h * w), device=self.device)
        counts = torch.zeros((h * w,), device=self.device)

        # Accumulate in chunks of anchors to bound peak memory on large holes.
        chunk = max(1, 4_000_000 // max(1, oy.numel()))
        for start in range(0, lh, chunk):
            end = min(start + chunk, lh)
            cy, cx = ys_h[start:end], xs_h[start:end]
            sy, sx = src_y[start:end], src_x[start:end]

            t_y = (cy.unsqueeze(1) + oy.unsqueeze(0)).clamp(0, h - 1)
            t_x = (cx.unsqueeze(1) + ox.unsqueeze(0)).clamp(0, w - 1)
            s_y = (sy.unsqueeze(1) + oy.unsqueeze(0)).clamp(0, h - 1)
            s_x = (sx.unsqueeze(1) + ox.unsqueeze(0)).clamp(0, w - 1)

            values = source_img[:, s_y.reshape(-1), s_x.reshape(-1)]  # (3, n*K*K)
            t_flat = (t_y * w + t_x).reshape(-1)
            acc.index_add_(1, t_flat, values)
            counts.index_add_(0, t_flat, torch.ones_like(t_flat, dtype=torch.float32))

        counts = counts.clamp_min(1.0)
        filled = (acc / counts).reshape(c, h, w)
        return torch.where(hole.unsqueeze(0), filled, source_img)

    def _multiscale_patchmatch(
        self, roi_img: torch.Tensor, roi_hole: torch.Tensor, coarse_init: torch.Tensor
    ) -> torch.Tensor | None:
        """Coarse-to-fine PatchMatch. Returns the final offset field ``(2, H, W)`` or None."""
        c, h, w = roi_img.shape
        K = self.nnf_patch_size

        # Global validity check: is there enough known context to sample from?
        valid_full = self._valid_source_mask(roi_hole, K)
        if int(valid_full.sum().item()) < max(16, K * K):
            self._log("insufficient valid source region -> laplace-only fallback")
            return None

        # Build pyramid sizes (coarse -> fine). Finest processed level <= pm_max_size.
        max_side = max(h, w)
        fine_scale = min(1.0, self.pm_max_size / float(max_side))
        fh = max(self.pyramid_min_size, int(round(h * fine_scale)))
        fw = max(self.pyramid_min_size, int(round(w * fine_scale)))

        sizes = []
        ch, cw = fh, fw
        while True:
            sizes.append((ch, cw))
            if min(ch, cw) <= self.pyramid_min_size:
                break
            ch = max(self.pyramid_min_size, ch // 2)
            cw = max(self.pyramid_min_size, cw // 2)
        sizes = list(reversed(sizes))  # coarsest first

        off = None
        prev_cur = None
        prev_h = prev_w = None
        for level, (lh, lw) in enumerate(sizes):
            img_l = F.interpolate(roi_img.unsqueeze(0), size=(lh, lw), mode="bilinear", align_corners=False)[0]
            hole_l = F.interpolate(roi_hole.float().unsqueeze(0).unsqueeze(0), size=(lh, lw), mode="nearest")[0, 0] > 0.5
            if not hole_l.any():
                continue
            valid_l = self._valid_source_mask(hole_l, K)

            if off is None:
                # coarse initialization: Laplace fill for color + random valid NNF
                init_l = F.interpolate(coarse_init.unsqueeze(0), size=(lh, lw), mode="bilinear", align_corners=False)[0]
                cur = torch.where(hole_l.unsqueeze(0), init_l, img_l)
                hole_flat = hole_l.reshape(-1)
                hole_idx = torch.nonzero(hole_flat, as_tuple=False).squeeze(1)
                ys_h = (hole_idx // lw).float()
                xs_h = (hole_idx % lw).float()
                off_h = self._init_nnf(hole_idx, ys_h, xs_h, valid_l)
                if off_h is None:
                    continue
                off = torch.zeros((2, lh, lw), device=self.device)
                off.reshape(2, -1)[:, hole_idx] = off_h
            else:
                # upsample offsets (scale with resolution)
                off = F.interpolate(off.unsqueeze(0), size=(lh, lw), mode="nearest")[0]
                off[0] *= lh / float(prev_h)
                off[1] *= lw / float(prev_w)
                # seed the hole with the previous (coarser) estimate; the hole in
                # img_l holds the *unwanted* content and must never seed the fill.
                cur_up = F.interpolate(prev_cur.unsqueeze(0), size=(lh, lw), mode="bilinear", align_corners=False)[0]
                cur = torch.where(hole_l.unsqueeze(0), cur_up, img_l)

            cur = self._solve_nnf_level(cur, img_l, hole_l, valid_l, off)
            prev_cur = cur
            prev_h, prev_w = lh, lw

        if off is None:
            return None

        # Upsample the final offset field to full ROI resolution.
        if (prev_h, prev_w) != (h, w):
            off = F.interpolate(off.unsqueeze(0), size=(h, w), mode="nearest")[0]
            off[0] *= h / float(prev_h)
            off[1] *= w / float(prev_w)
        return off

    # ------------------------------------------------------------------

    def inpaint(self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor):
        t0 = time.perf_counter()
        if self.seed is not None:
            torch.manual_seed(self.seed)
        pkg = self._to_tensor(image, mask)
        self._log(f"start patchmatch: device={self.device}, image={tuple(pkg.image.shape)}, hole={int(pkg.mask.sum().item())}")

        if not pkg.mask.any():
            return self._to_numpy(pkg.image, pkg)

        ys = torch.nonzero(pkg.mask, as_tuple=False)[:, 0]
        xs = torch.nonzero(pkg.mask, as_tuple=False)[:, 1]
        y0 = max(0, int(ys.min().item()) - self.roi_margin)
        y1 = min(pkg.mask.shape[0] - 1, int(ys.max().item()) + self.roi_margin)
        x0 = max(0, int(xs.min().item()) - self.roi_margin)
        x1 = min(pkg.mask.shape[1] - 1, int(xs.max().item()) + self.roi_margin)

        roi_img = pkg.image[:, y0 : y1 + 1, x0 : x1 + 1]
        roi_mask = pkg.mask[y0 : y1 + 1, x0 : x1 + 1]
        rh, rw = roi_mask.shape

        # Downscale working resolution if the ROI is very large.
        scale = 1.0
        max_side = max(rh, rw)
        if max_side > self.max_working_size:
            scale = self.max_working_size / float(max_side)
            nh = max(32, int(round(rh * scale)))
            nw = max(32, int(round(rw * scale)))
            roi_img_w = F.interpolate(roi_img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)[0]
            roi_mask_w = F.interpolate(roi_mask.float().unsqueeze(0).unsqueeze(0), size=(nh, nw), mode="nearest")[0, 0] > 0.5
            self._log(f"downscale roi: ({rh},{rw}) -> ({nh},{nw})")
        else:
            roi_img_w = roi_img
            roi_mask_w = roi_mask

        # 1. Coarse color initialization (smooth gradient fill).
        coarse_init = self._laplace_fill(roi_img_w, roi_mask_w)

        # 2. Multi-scale PatchMatch texture synthesis -> offset field at working res.
        #    Skip it for flat / low-texture ROIs (e.g. sky): the Laplace fill is already
        #    optimal there, and this keeps the common case fast.
        _low_tex = self._is_low_texture(roi_img_w, roi_mask_w)
        if _low_tex:
            self._log("low-texture ROI -> laplace-only fast path")
            off = None
        else:
            off = self._multiscale_patchmatch(roi_img_w, roi_mask_w, coarse_init)

        # [PM_DIAG] どの経路を通ったか（low_texture=laplaceのみ / PatchMatch）とROIサイズの診断ログ。
        logging.debug(
            "[PM_DIAG] fill image=%s mask_hole=%d roi=(%d,%d) work=(%d,%d) scale=%.3f "
            "low_texture=%s patchmatch=%s",
            tuple(pkg.image.shape), int(pkg.mask.sum().item()), rh, rw,
            roi_img_w.shape[-2], roi_img_w.shape[-1], scale, _low_tex, off is not None,
        )

        if off is None:
            # Fallback: flat / low-context region -> Laplace fill only (e.g. sky).
            filled_w = coarse_init
        else:
            # 3. Final dense voting at working resolution from the real pixels.
            filled_w = self._vote_reconstruct(roi_img_w, roi_mask_w, off, self.vote_patch_size)
            # 4. Seam handling: feather a thin boundary band between fill and known.
            filled_w = self._seam_blend(roi_img_w, filled_w, roi_mask_w)

        # Upscale back to full ROI resolution if we worked downscaled.
        if scale < 1.0:
            final_roi = F.interpolate(filled_w.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False)[0]
            final_roi = torch.where(roi_mask.unsqueeze(0), final_roi, roi_img)
        else:
            final_roi = torch.where(roi_mask.unsqueeze(0), filled_w, roi_img)

        out = pkg.image.clone()
        out[:, y0 : y1 + 1, x0 : x1 + 1] = final_roi
        self._log(f"finish inpaint: total_elapsed={time.perf_counter()-t0:.3f}s")
        return self._to_numpy(out, pkg)

    def _is_low_texture(self, roi_img: torch.Tensor, hole: torch.Tensor) -> bool:
        """True when the known region is smooth enough that Laplace fill suffices (sky)."""
        known = ~hole
        if not known.any():
            return False
        em = self._edge_mag(roi_img)[0][known]
        vals = roi_img[:, known]
        # Range-aware threshold so it works for HDR/linear working spaces, not just [0,1].
        scale = float((vals.amax() - vals.amin()).clamp_min(1e-3))
        return float(em.mean()) < self.low_texture_edge_threshold * scale

    def _seam_blend(self, original: torch.Tensor, filled: torch.Tensor, hole: torch.Tensor, band: int = 3) -> torch.Tensor:
        """Feather a thin band around the hole boundary to hide seams.

        Inside the hole we keep the synthesized pixels; only a ``band``-wide ring
        straddling the boundary is cross-faded so the transition is smooth.
        """
        hole_f = hole.float().unsqueeze(0).unsqueeze(0)
        k = 2 * band + 1
        # Distance-like ramp: blur the hole mask, use it as alpha near the border.
        blur = T.GaussianBlur(kernel_size=k, sigma=float(band))
        alpha = blur(hole_f)[0]  # 1 deep inside hole, ~0 far outside, ramp at border
        # Only affect a neighborhood of the boundary; deep interior stays fully filled.
        result = alpha * filled + (1.0 - alpha) * original
        # Guarantee the deep hole interior is exactly the synthesized fill.
        return torch.where(hole.unsqueeze(0), torch.where(alpha > 0.98, filled, result), original)


def content_aware_fill(
    image,
    mask,
    roi_margin: int = 256,
    max_working_size: int = 1536,
    speed_mode: str = "quality",
    verbose: bool = True,
    device: str | None = None,
    texture_patch_size: int = 7,
    texture_strength: float = 1.0,
    match_scale: float = 0.5,
    candidate_count: int = 256,
    nnf_patch_size: int = 7,
    vote_patch_size: int | None = None,
    pm_max_size: int = 512,
    pm_iters: int = 4,
    em_iters: int = 2,
):
    return PatchMatchInpainting(
        max_working_size=max_working_size,
        roi_margin=roi_margin,
        speed_mode=speed_mode,
        verbose=verbose,
        device=device,
        texture_patch_size=texture_patch_size,
        texture_strength=texture_strength,
        match_scale=match_scale,
        candidate_count=candidate_count,
        nnf_patch_size=nnf_patch_size,
        vote_patch_size=vote_patch_size,
        pm_max_size=pm_max_size,
        pm_iters=pm_iters,
        em_iters=em_iters,
    ).inpaint(image, mask)
