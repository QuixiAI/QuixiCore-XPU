"""Parity harness: QuixiCore XPU kernels vs PyTorch references on torch.xpu.

A second correctness oracle alongside the C++ fp64 gate — here the reference is
PyTorch's own XPU eager kernels. Run after building the extension:

    source /opt/intel/oneapi/setvars.sh
    cd bindings/pytorch
    ../../.venv/bin/python setup.py build_ext --inplace
    ../../.venv/bin/python test_parity.py
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
        grad_out = torch.randn(x.shape, device=DEV, dtype=dt, generator=g)
        x_ref = x.detach().clone().requires_grad_(True)
        grad_ref = torch.autograd.grad(
            F.gelu(x_ref), x_ref, grad_outputs=grad_out
        )[0]
        fails += not check(
            "gelu_backward",
            tk_xpu.gelu_backward(grad_out, x),
            grad_ref,
            dt,
        )
        fails += not check("silu", tk_xpu.silu(x), F.silu(x), dt)
        fails += not check("softmax", tk_xpu.softmax(x), F.softmax(x, dim=-1), dt)
        w = torch.randn(2048, device=DEV, dtype=dt, generator=g)
        b = torch.randn(2048, device=DEV, dtype=dt, generator=g)
        # reduce in f32 and cast once at the end (matches the kernel; casting the
        # rsqrt scale to bf16 mid-computation would over-quantize the reference).
        ref_rms = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(dt)
        fails += not check("rms_norm", tk_xpu.rms_norm(x, w, 1e-6), ref_rms, dt)
        residual = torch.randn(64, 2048, device=DEV, dtype=dt, generator=g)
        residual_before = residual.clone()
        summed = x.float() + residual_before.float()
        inv_rms = torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + 1e-6)
        fused_ref = ((summed * inv_rms).to(dt) * w).to(dt)
        fused = tk_xpu.fused_add_rms_norm(x, residual, w, 1e-6)
        fails += not check("fused_add_rms", fused, fused_ref, dt)
        fails += not check("fused_residual", residual, summed.to(dt), dt)
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

    # W8A16 checkpoint layout: BF16 activations and [N,K] raw e4m3 weights.
    M, N, K = 4, 128, 256
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=g) * 0.2
    weight_fp8 = (
        torch.randn(N, K, device=DEV, dtype=torch.float32, generator=g) * 0.2
    ).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.5, 0.75, N, device=DEV, dtype=torch.float32)
    got = tk_xpu.fp8_gemm_w8a16(
        x, weight_fp8.view(torch.uint8), scales, 0, True
    )
    ref = (x.float() @ weight_fp8.float().T * scales).to(torch.bfloat16)
    fails += not check("fp8_w8a16", got, ref, torch.bfloat16)

    # Split NVFP4 MoE must match the fused route and remain graph-safe when its
    # caller owns the scratch allocation for the graph lifetime.
    M, E, TOP_K, K, I = 1, 2, 1, 32, 32
    hidden = (
        torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=g)
        * 0.01
    )
    topk_ids = torch.tensor([[0]], device=DEV, dtype=torch.int32)
    topk_weights = torch.ones(M, TOP_K, device=DEV, dtype=torch.float32)
    w13 = torch.full((E, 2 * I, K // 2), 0x11, device=DEV, dtype=torch.uint8)
    w13_scales = torch.full(
        (E, 2 * I, K // 16), 0x38, device=DEV, dtype=torch.uint8
    )
    w13_global = torch.full((E,), 2.0**-22, device=DEV, dtype=torch.float32)
    w2 = torch.full((E, K, I // 2), 0x11, device=DEV, dtype=torch.uint8)
    w2_scales = torch.full(
        (E, K, I // 16), 0x38, device=DEV, dtype=torch.uint8
    )
    w2_global = torch.full((E,), 2.0**-22, device=DEV, dtype=torch.float32)
    moe_args = (
        hidden,
        topk_ids,
        topk_weights,
        w13,
        w13_scales,
        w13_global,
        w2,
        w2_scales,
        w2_global,
    )
    moe_fused = tk_xpu.nvfp4_moe(*moe_args, None, False, True)
    moe_scratch = torch.empty(
        M * TOP_K, 2 * I, device=DEV, dtype=torch.float32
    )
    moe_split = tk_xpu.nvfp4_moe(*moe_args, moe_scratch, True, True)
    fails += not check("nvfp4_moe_split", moe_split, moe_fused, torch.float32)

    tk_xpu.synchronize_current_stream()
    moe_graph = tk_xpu.XPUGraph()
    moe_graph.capture_begin()
    moe_graph_output = tk_xpu.nvfp4_moe(
        *moe_args, moe_scratch, True, True
    )
    moe_graph.capture_end()
    moe_graph.replay()
    moe_graph.synchronize()
    fails += not check(
        "nvfp4_moe_graph", moe_graph_output, moe_fused, torch.float32
    )

    # The binding bypasses Python's device-wide graph synchronization and waits
    # only on the captured current stream.
    graph_input = torch.randn(4096, device=DEV, dtype=torch.float32, generator=g)
    tk_xpu.silu(graph_input)
    tk_xpu.synchronize_current_stream()
    graph = tk_xpu.XPUGraph(True)
    graph.capture_begin()
    graph_output = tk_xpu.silu(graph_input)
    graph.capture_end()
    graph.instantiate()
    graph.replay()
    graph.synchronize()
    fails += not check("xpu_graph", graph_output, F.silu(graph_input), torch.float32)
    graph_input.mul_(0.5)
    graph.replay()
    graph.synchronize()
    fails += not check(
        "xpu_graph_replay", graph_output, F.silu(graph_input), torch.float32
    )
    graph.reset()
    try:
        graph.synchronize()
    except RuntimeError as error:
        graph_reset_ok = "no capture stream" in str(error)
    else:
        graph_reset_ok = False
    print(f"  {'xpu_graph_reset':16s} {'':16s}  {'ok' if graph_reset_ok else 'FAIL'}")
    fails += not graph_reset_ok

    negative_pool_ok = False
    try:
        tk_xpu.XPUGraph().capture_begin((-1, 0))
    except RuntimeError as error:
        negative_pool_ok = "non-negative" in str(error)
    print(f"  {'xpu_graph_pool':16s} {'':16s}  {'ok' if negative_pool_ok else 'FAIL'}")
    fails += not negative_pool_ok

    # Slot zero is a valid GDN state slot. Invalid slots leave both state
    # tensors unchanged, zero the recurrent core, and still pass through z.
    batch, slots = 1, 2
    qkvz = (
        torch.randn(batch, 12288, device=DEV, dtype=torch.bfloat16, generator=g)
        * 0.01
    )
    ba = torch.zeros(batch, 64, device=DEV, dtype=torch.bfloat16)
    conv_state = torch.zeros(slots, 3, 8192, device=DEV, dtype=torch.bfloat16)
    ssm_state = torch.zeros(slots, 32, 128, 128, device=DEV)
    conv_weight = torch.zeros(8192, 4, device=DEV, dtype=torch.bfloat16)
    conv_bias = torch.zeros(8192, device=DEV, dtype=torch.bfloat16)
    a_log = torch.zeros(32, device=DEV)
    dt_bias = torch.zeros(32, device=DEV, dtype=torch.bfloat16)
    expected_z = qkvz[:, 8192:].reshape(batch, 32, 128)
    core, z = tk_xpu.qwen_gdn_decode(
        qkvz,
        ba,
        conv_state,
        ssm_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        torch.tensor([0], device=DEV, dtype=torch.int32),
    )
    expected_conv = torch.zeros_like(conv_state)
    expected_conv[0, 2] = qkvz[0, :8192]
    gdn_slot_zero_ok = (
        torch.equal(conv_state, expected_conv)
        and torch.count_nonzero(ssm_state).item() == 0
        and torch.count_nonzero(core).item() == 0
        and torch.equal(z, expected_z)
    )
    print(f"  {'gdn_slot_zero':16s} {'torch.bfloat16':16s}  {'ok' if gdn_slot_zero_ok else 'FAIL'}")
    fails += not gdn_slot_zero_ok

    for invalid_index in (-1, slots):
        conv_before = conv_state.clone()
        ssm_before = ssm_state.clone()
        core, z = tk_xpu.qwen_gdn_decode(
            qkvz,
            ba,
            conv_state,
            ssm_state,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            torch.tensor([invalid_index], device=DEV, dtype=torch.int32),
        )
        invalid_ok = (
            torch.equal(conv_state, conv_before)
            and torch.equal(ssm_state, ssm_before)
            and torch.count_nonzero(core).item() == 0
            and torch.equal(z, expected_z)
        )
        print(
            f"  {f'gdn_invalid_{invalid_index}':16s} "
            f"{'torch.bfloat16':16s}  {'ok' if invalid_ok else 'FAIL'}"
        )
        fails += not invalid_ok
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
