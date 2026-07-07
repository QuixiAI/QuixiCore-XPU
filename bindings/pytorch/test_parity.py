"""Parity harness: QuixiCore XPU kernels vs PyTorch references on torch.xpu.

A second correctness oracle alongside the C++ fp64 gate — here the reference is
PyTorch's own XPU eager kernels. Run after building the extension:

    source /opt/intel/oneapi/setvars.sh
    cd bindings/pytorch && python setup.py build_ext --inplace
    python test_parity.py
"""

import sys

import torch
import torch.nn.functional as F

import tk_xpu

DEV = "xpu"
TOL = {torch.float32: (1e-4, 1e-4), torch.bfloat16: (8e-3, 8e-3), torch.float16: (2e-3, 2e-3)}


def check(name, got, ref, dt):
    rtol, atol = TOL[dt]
    ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
    md = (got.float() - ref.float()).abs().max().item()
    print(f"  {name:16s} {str(dt):16s} max_abs={md:.3e}  {'ok' if ok else 'FAIL'}")
    return ok


def main():
    assert torch.xpu.is_available(), "torch.xpu not available"
    print(f"torch {torch.__version__}  xpu devices={torch.xpu.device_count()}  "
          f"tk_xpu devices={tk_xpu.device_count()}")
    g = torch.Generator(device=DEV).manual_seed(0)
    fails = 0
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.randn(64, 2048, device=DEV, dtype=dt, generator=g)
        fails += not check("gelu", tk_xpu.gelu(x), F.gelu(x), dt)
        fails += not check("silu", tk_xpu.silu(x), F.silu(x), dt)
        fails += not check("softmax", tk_xpu.softmax(x), F.softmax(x, dim=-1), dt)
        w = torch.randn(2048, device=DEV, dtype=dt, generator=g)
        b = torch.randn(2048, device=DEV, dtype=dt, generator=g)
        # reduce in f32 and cast once at the end (matches the kernel; casting the
        # rsqrt scale to bf16 mid-computation would over-quantize the reference).
        ref_rms = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(dt)
        fails += not check("rms_norm", tk_xpu.rms_norm(x, w, 1e-6), ref_rms, dt)
        fails += not check("layernorm", tk_xpu.layernorm(x, w, b, 1e-5),
                           F.layer_norm(x, (2048,), w, b, 1e-5), dt)
        # attention: [H, S, D] vs F.sdpa on [1, H, S, D]
        H, S, D = 8, 128, 64
        Q = torch.randn(H, S, D, device=DEV, dtype=dt, generator=g) * 0.5
        K = torch.randn(H, S, D, device=DEV, dtype=dt, generator=g) * 0.5
        V = torch.randn(H, S, D, device=DEV, dtype=dt, generator=g)
        ref_attn = F.scaled_dot_product_attention(Q[None], K[None], V[None], is_causal=True)[0]
        fails += not check("attention", tk_xpu.attention(Q, K, V, True), ref_attn, dt)
        # argmax (int compare)
        logits = torch.randn(64, 4000, device=DEV, dtype=dt, generator=g)
        am = tk_xpu.argmax(logits).cpu()
        ref_am = logits.argmax(-1).to(torch.int32).cpu()
        ok = bool((am == ref_am).all())
        print(f"  {'argmax':16s} {str(dt):16s} mism={int((am != ref_am).sum())}  {'ok' if ok else 'FAIL'}")
        fails += not ok
    # dense_gemm (f32/bf16)
    for dt in (torch.float32, torch.bfloat16):
        a = torch.randn(128, 160, device=DEV, dtype=dt, generator=g) * 0.2
        b2 = torch.randn(160, 96, device=DEV, dtype=dt, generator=g) * 0.2
        fails += not check("dense_gemm", tk_xpu.dense_gemm(a, b2), a @ b2, dt)
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
