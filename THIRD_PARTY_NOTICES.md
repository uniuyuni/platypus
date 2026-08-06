# Third Party Notices

Shade Wave is licensed under the GNU General Public License
version 3 or later. This file records notable third-party code, ports,
algorithms, and bundled components that require attribution or separate terms.

This notice file is not exhaustive for every transitive Python or conda package
in a local development environment. Binary distributions should additionally
include the license notices generated from the exact bundled environment.

## RawTherapee

Files:

- `external/libraw_enhanced/core/cpu_accelerator.cpp`
- `external/libraw_enhanced/core/metal/demosaic_bayer_amaze.metal`
- `external/libraw_enhanced/core/metal/demosaic_xtrans_1pass.metal`
- `external/libraw_enhanced/core/metal/demosaic_xtrans_3pass.metal`

These files contain demosaicing implementations ported from or derived from
RawTherapee algorithms, including AMaZE and X-Trans demosaicing paths.

RawTherapee is licensed under the GNU General Public License version 3.
Because these ports are integrated into `libraw_enhanced`, `libraw_enhanced`
is licensed as GPL-3.0-or-later in this repository.

Upstream: https://github.com/RawTherapee/RawTherapee

## LibRaw

`external/libraw_enhanced` links against and vendors/builds LibRaw.
LibRaw is distributed under a dual license: LGPL-2.1 and CDDL-1.0.
This project uses LibRaw under the LGPL-2.1 terms unless a distribution
explicitly documents otherwise.

Upstream: https://www.libraw.org/

## Lensfun and lensfunpy

Lens correction uses `lensfunpy`, a Python wrapper around the Lensfun lens
database and correction library.

`lensfunpy` is licensed under the MIT License.
Lensfun is licensed under LGPL-3.0-or-later. Binary distributions that bundle
`lensfunpy`, `liblensfun`, or Lensfun database files must include the license
texts and notices from the exact bundled packages.

Upstreams:

- https://github.com/letmaik/lensfunpy
- https://github.com/lensfun/lensfun

## Adobe DNG SDK

Files:

- `cores/dng_temperature.py`

`cores/dng_temperature.py` is based on the DNG SDK temperature/tint conversion
logic and black-body lookup table. The DNG SDK License Agreement grants a
non-exclusive, worldwide, royalty-free license to use, reproduce, prepare
derivative works from, publicly display, publicly perform, distribute, and
sublicense the SDK software for any purpose, subject to its restrictions.
The license text provided with DNG SDK 1.7.1 is reproduced in
`licenses/DNG_SDK_LICENSE.txt`.

Required notice summary:

- Do not remove copyright or other notices included in the DNG SDK software or
  documentation.
- Include those notices in copies of DNG SDK software distributed in
  human-readable format.
- Adobe and the DNG logo are Adobe trademarks and may not be used to endorse or
  promote a product without separate permission.
- If distributing the SDK software in a commercial product, the distributor
  agrees to defend, indemnify, and hold harmless Adobe against claims arising
  out of that distribution.

The DNG SDK license is separate from the DNG File Format Specification patent
license.

Upstream: https://helpx.adobe.com/camera-raw/digital-negative.html

## Colour Science for Python

Files:

- `effect_backends/colour_functions_reference.py`
- `effect_backends/colour_functions_adapter.py`
- `effect_backends/colour_functions_cpu.c`
- `effect_backends/colour_functions_capi.h`
- `effect_backends/colour_functions_pybind.cpp`

The Python reference implementation and native fused display-transform backend
implement a subset of Colour Science compatible colour-space conversion
behaviour and data. Colour Science for Python is licensed under BSD-3-Clause.

Copyright 2013 Colour Developers.

Upstream: https://github.com/colour-science/colour

## External AI and Image Processing Projects

The repository setup can clone or install optional external projects under
`external/`. Their license terms remain separate from Shade Wave's project
license and must be included when those projects, model weights, or binaries
are redistributed.

- `external/SAM3`: Meta SAM License.
- `external/depth_pro`: Apple license in `external/depth_pro/LICENSE`.
- `external/SCUNet`: Apache-2.0.
- `external/radiance_codec`: see the upstream project notice.
- `external/radiance_denoise`: see the upstream project notice.
- `external/demosaicnet_torch`: research implementation; verify upstream
  license and model-weight terms before redistribution.

## LaMa (big-lama)

The `helpers/ri_helper.py` inpainting / object-eraser path uses the **big-lama**
model as its structure model, loaded from `./checkpoints/big-lama.pt`
(TorchScript weights). LaMa ("Resolution-robust Large Mask Inpainting with
Fourier Convolutions", Suvorov et al.) is published by advimman/lama under the
**Apache License 2.0**. The model weights and the LaMa source carry their own
Apache-2.0 terms; include them when redistributing the weights or a packaged
build that bundles them. Upstream: https://github.com/advimman/lama

The surrounding inference framework (40MP tiling, HDR tone mapping, harmonic
tone correction, PatchMatch texture, ShadeWave integration) is the sibling
`nagi_inpaint` project and is not part of big-lama.

## Packaging Notes

The macOS `.app` bundle can include dynamic libraries from the pixi/conda
environment. Licenses such as LGPL, MPL, GPL-with-exception, and other notices
may apply to bundled runtime libraries. In particular, HEIF/JXL/TIFF/EXR/VIPS
support may pull in codec libraries with their own redistribution requirements.
