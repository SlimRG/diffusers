# -*- coding: utf-8 -*-
"""
SlimLanPaint

A lightweight, optional tiled-inference helper for WanVACEPipeline.

Goal:
- Reduce peak VRAM during denoising by running the transformer on overlapping spatial tiles
  and blending the predictions back together (feathered window).
- Keep the integration *optional*: if you don't pass a SlimLanPaint instance to the pipeline,
  the pipeline behaves exactly as upstream.

Notes:
- This is intentionally model-agnostic: it only assumes the model is "fully convolutional"
  over (H, W) in latent space (i.e. it can accept smaller spatial crops).
- Blending is performed in FP32 by default for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class SlimLanPaintConfig:
    """
    Configuration for tiled inference.

    tile_size:
        Tile size in latent-space pixels. You may pass:
        - int: use the same size for height and width
        - (int, int): (tile_h, tile_w)

    tile_overlap:
        Overlap (in latent-space pixels) between neighboring tiles. Larger overlap reduces seams
        but increases compute.

    fp32_accum:
        If True, accumulate blended predictions in float32 and cast back to the model dtype.
    """

    tile_size: Union[int, Tuple[int, int]] = 128
    tile_overlap: int = 32
    fp32_accum: bool = True


class SlimLanPaint:
    """
    Optional helper that computes model predictions in overlapping tiles and blends them.

    Expected tensor shapes (WanVACE):
        hidden_states:         [B, C, T, H, W]
        control_hidden_states: [B, Cc, T, H, W]

    The method returns the blended prediction with the same shape as hidden_states.
    """

    def __init__(self, *, config: SlimLanPaintConfig | None = None, **kwargs):
        if config is None:
            config = SlimLanPaintConfig(**kwargs)
        self.config = config

        # Cache precomputed 2D feather windows keyed by:
        # (h, w, top, bottom, left, right, device, dtype)
        self._window_cache: Dict[Tuple[int, int, bool, bool, bool, bool, torch.device, torch.dtype], torch.Tensor] = {}

    @staticmethod
    def _as_pair(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(v, int):
            return (v, v)
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, int) for x in v):
            return v
        raise TypeError(f"tile_size must be int or (int, int), got: {type(v)} {v!r}")

    @staticmethod
    def _compute_starts(full: int, tile: int, overlap: int) -> Tuple[int, ...]:
        if tile >= full:
            return (0,)
        step = max(1, tile - overlap)
        starts = list(range(0, full, step))
        last = full - tile
        if starts[-1] != last:
            starts.append(last)
        # de-dup while keeping order
        out = []
        seen = set()
        for s in starts:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return tuple(out)

    def _feather_1d(self, length: int, overlap: int, fade_left: bool, fade_right: bool, *, device, dtype) -> torch.Tensor:
        if overlap <= 0:
            return torch.ones(length, device=device, dtype=dtype)

        w = torch.ones(length, device=device, dtype=dtype)
        ramp = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)

        if fade_left and overlap > 0:
            w[:overlap] = torch.minimum(w[:overlap], ramp)
        if fade_right and overlap > 0:
            w[-overlap:] = torch.minimum(w[-overlap:], ramp.flip(0))
        return w

    def _get_window(
        self,
        h: int,
        w: int,
        *,
        overlap: int,
        top: bool,
        bottom: bool,
        left: bool,
        right: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (h, w, top, bottom, left, right, device, dtype)
        win = self._window_cache.get(key)
        if win is not None:
            return win

        wy = self._feather_1d(h, overlap, fade_left=top, fade_right=bottom, device=device, dtype=dtype)
        wx = self._feather_1d(w, overlap, fade_left=left, fade_right=right, device=device, dtype=dtype)
        win = (wy[:, None] * wx[None, :]).contiguous()
        self._window_cache[key] = win
        return win

    @torch.no_grad()
    def predict_noise(
        self,
        *,
        model: Any,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        control_hidden_states_scale: float,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Run `model(...)` in overlapping tiles and blend the prediction.

        This is a drop-in replacement for:
            model(hidden_states=..., timestep=..., encoder_hidden_states=..., control_hidden_states=...)[0]
        """
        if hidden_states.ndim != 5:
            raise ValueError(f"hidden_states must be [B,C,T,H,W], got shape={tuple(hidden_states.shape)}")
        if control_hidden_states.ndim != 5:
            raise ValueError(
                f"control_hidden_states must be [B,Cc,T,H,W], got shape={tuple(control_hidden_states.shape)}"
            )

        tile_h, tile_w = self._as_pair(self.config.tile_size)
        H, W = int(hidden_states.shape[-2]), int(hidden_states.shape[-1])

        tile_h = max(1, min(tile_h, H))
        tile_w = max(1, min(tile_w, W))
        overlap = int(self.config.tile_overlap)
        overlap = max(0, min(overlap, tile_h - 1, tile_w - 1))

        # No tiling needed
        if tile_h == H and tile_w == W:
            return model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=control_hidden_states_scale,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        starts_y = self._compute_starts(H, tile_h, overlap)
        starts_x = self._compute_starts(W, tile_w, overlap)

        # Accumulate in float32 for stability (especially with bf16/fp16 models)
        out_dtype = torch.float32 if self.config.fp32_accum else hidden_states.dtype
        device = hidden_states.device

        pred_accum = torch.zeros_like(hidden_states, dtype=out_dtype, device=device)
        weight_accum = torch.zeros((1, 1, 1, H, W), dtype=out_dtype, device=device)

        for y0 in starts_y:
            y1 = y0 + tile_h
            top = y0 > 0
            bottom = y1 < H

            for x0 in starts_x:
                x1 = x0 + tile_w
                left = x0 > 0
                right = x1 < W

                hs = hidden_states[..., y0:y1, x0:x1]
                cs = control_hidden_states[..., y0:y1, x0:x1]

                tile_pred = model(
                    hidden_states=hs,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    control_hidden_states=cs,
                    control_hidden_states_scale=control_hidden_states_scale,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                win = self._get_window(
                    tile_h,
                    tile_w,
                    overlap=overlap,
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right,
                    device=device,
                    dtype=out_dtype,
                )

                win = win.view(1, 1, 1, tile_h, tile_w)
                tile_pred = tile_pred.to(out_dtype)
                pred_accum[..., y0:y1, x0:x1] += tile_pred * win
                weight_accum[..., y0:y1, x0:x1] += win

        pred = pred_accum / torch.clamp(weight_accum, min=torch.finfo(out_dtype).eps)
        return pred.to(dtype=hidden_states.dtype)
