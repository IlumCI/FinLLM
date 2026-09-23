"""Hardware and precision utilities -- the GPU+CPU+RAM hybrid.

LAMb was written CPU-first but device-agnostic. This module is the single place
that picks the compute device, sets up mixed precision, and reports what it found,
so every entry point scales to a GPU with no code change:

* **GPU (or CPU/MPS) does the neural compute.** ``resolve_device`` auto-selects
  ``cuda`` > ``mps`` > ``cpu`` (override with ``--device`` or ``LAMB_DEVICE``), and
  ``Amp`` turns on mixed precision on CUDA automatically (bf16 on Ampere+, else
  fp16 with a gradient scaler); on CPU it stays fp32 unless asked, so there is no
  regression on the CPU-only path.
* **CPU runs the exact Rust kernels** (``lamb_core``: verifier, curriculum
  sampler, top-k store) alongside the device compute -- correctness-critical work
  that is cheap and native, overlapping the accelerator.
* **RAM holds the buffers** (replay buffers, held-out eval sets, the exact
  retrieval tier); ``configure_threads`` gives the CPU legs every core.

Nothing here requires a GPU to import or test; on a CPU-only box it resolves to
``cpu`` and the mixed-precision paths become no-ops.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Optional

import torch


def resolve_device(prefer: str = "auto") -> str:
    """Return a concrete device string. ``auto`` picks cuda > mps > cpu.

    An explicit value (``cpu``, ``cuda``, ``cuda:1``, ``mps``) is honoured as-is.
    The ``LAMB_DEVICE`` environment variable overrides the argument, so a run can
    be pinned without touching code.
    """
    prefer = (os.environ.get("LAMB_DEVICE") or prefer or "auto").lower()
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _device_type(device: str) -> str:
    return device.split(":", 1)[0]


def configure_threads(n: Optional[int] = None) -> int:
    """Give torch every CPU core (the CPU leg of the hybrid). Returns the count."""
    n = n or os.cpu_count() or 1
    with contextlib.suppress(Exception):
        torch.set_num_threads(int(n))
    return torch.get_num_threads()


def to_device(x: Any, device: str, non_blocking: bool = True) -> Any:
    """Move a tensor / dict / list of tensors to ``device`` (async when pinned)."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: to_device(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device, non_blocking) for v in x)
    return x


class Amp:
    """Mixed-precision helper: an autocast context plus a scaled backward+step.

    Defaults: enabled on CUDA (bf16 when supported, else fp16 with a
    :class:`GradScaler`), disabled on CPU/MPS. Pass ``enabled=True`` to force CPU
    bf16 autocast. When disabled, ``autocast`` is a null context and
    ``backward_step`` is a plain optimizer step, so the fp32 CPU path is unchanged.
    """

    def __init__(self, device: str, enabled: Optional[bool] = None):
        self.device = device
        self.dev_type = _device_type(device)
        is_cuda = self.dev_type == "cuda"
        self.enabled = bool(is_cuda) if enabled is None else bool(enabled)
        if is_cuda:
            bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            self.dtype = torch.bfloat16 if bf16 else torch.float16
        else:
            self.dtype = torch.bfloat16  # CPU/MPS autocast dtype if forced on
        # A gradient scaler is only needed for fp16 (bf16 has fp32 range).
        self.use_scaler = self.enabled and is_cuda and self.dtype is torch.float16
        try:
            self.scaler = torch.amp.GradScaler(device_type="cuda", enabled=self.use_scaler)
        except Exception:  # older torch, or no cuda build
            self.scaler = None
            self.use_scaler = False

    def autocast(self):
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.dev_type, dtype=self.dtype)

    def backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                      params, grad_clip: float = 0.0) -> None:
        optimizer.zero_grad(set_to_none=True)
        if self.use_scaler and self.scaler is not None:
            self.scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

    def summary(self) -> str:
        if not self.enabled:
            return "amp=off(fp32)"
        return f"amp={str(self.dtype).replace('torch.', '')}{'+scaler' if self.use_scaler else ''}"


def device_report(device: Optional[str] = None, amp: Optional["Amp"] = None) -> str:
    """One-line banner: device, accelerator name/VRAM, torch build, threads, AMP."""
    device = device or resolve_device()
    parts = [f"device={device}"]
    if _device_type(device) == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device() if ":" not in device else int(device.split(":")[1])
        p = torch.cuda.get_device_properties(idx)
        parts.append(f"gpu={p.name}")
        parts.append(f"vram={p.total_memory / 1e9:.0f}GB")
    parts.append(f"torch={torch.__version__}")
    parts.append(f"threads={torch.get_num_threads()}")
    if amp is not None:
        parts.append(amp.summary())
    return " ".join(parts)


# Scaling presets: bump width/depth/batch/thinking together. Keyed names so a CLI
# can offer --scale tiny|small|base|large without spelling out every knob. The
# larger presets assume a GPU; on CPU they still run, just slowly.
SCALE_PRESETS = {
    "tiny":  dict(d_model=96,  n_heads=4,  recurrent_steps=4,  batch_size=64),
    "small": dict(d_model=256, n_heads=8,  recurrent_steps=6,  batch_size=128),
    "base":  dict(d_model=512, n_heads=8,  recurrent_steps=8,  batch_size=256),
    "large": dict(d_model=1024, n_heads=16, recurrent_steps=12, batch_size=512),
}


def scale_preset(name: str) -> dict:
    if name not in SCALE_PRESETS:
        raise ValueError(f"unknown scale preset {name!r}; choose from {list(SCALE_PRESETS)}")
    return dict(SCALE_PRESETS[name])


def add_hardware_args(p):
    """Attach the shared ``--device / --amp / --no-amp / --threads`` flags."""
    p.add_argument("--device", default="auto",
                   help="cuda | cpu | mps | cuda:N | auto (auto picks the best available)")
    p.add_argument("--amp", dest="amp", action="store_true", default=None,
                   help="force mixed precision (bf16/fp16); default: on for CUDA, off for CPU")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="force fp32")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: all cores)")
    return p


def resolve_hardware(args):
    """Resolve ``--device``, set CPU threads; return ``(device, amp_flag)``."""
    device = resolve_device(getattr(args, "device", "auto"))
    configure_threads(getattr(args, "threads", None))
    return device, getattr(args, "amp", None)
