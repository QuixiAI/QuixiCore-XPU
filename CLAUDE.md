# Claude Instructions

Follow `AGENTS.md` in this repository. For kernel work, read `perf/perf.md`
before editing and update `perf/optimization_status.md` with measured results.

Do not commit any kernel implementation, routing change, benchmark change, or
performance claim until at least one focused performance optimization run has
been completed on an affected kernel. If the required XPU runtime is
unavailable, report the blocker instead of committing a claimed optimization.

## Performance philosophy: smash truisms with evidence

Never accept a hardware or library "truism" without empirically testing it on the
actual Intel GPU. In this backend the same assumption flips by hardware and by
dtype — measured on Arc Pro B60: oneDNN GPU layernorm accepts bf16/f16
scale/shift (the "must be f32" belief was just a defaulted API parameter); native
SYCL beats oneDNN 1.58x on f32 layernorm; oneDNN beats native SYCL 1.70x on bf16
layernorm. There is no universal winner.

Concretely:

- Treat "X must be f32", "the vendor lib is always faster", "staging/pipelining
  is the fast path", "it's memory-bound" as **hypotheses to test on B60**, not
  facts. Copying NVIDIA/Apple machinery blindly is a known trap (see the Metal
  notebook's anti-staging reversals and Marlin's cp.async pipeline that has no
  Intel analogue).
- Name the assumed bottleneck, then confirm it with profiling — it is often not
  what you assumed (e.g. "tanh-bound" was really "scalar 16-bit access-bound").
- Fix the ruler first: device profiling events, warmup, adaptive batching;
  re-verify a gap on a corrected harness before optimizing.
- When a vendor-vs-native (or any A/B) question arises, build both, measure both
  on the B60, ship both as co-equal variants, and route `Variant::best` from the
  data per (op, dtype). Record the reversal at the call site and in
  `perf/optimization_status.md` so the next pass starts from evidence.

Corollary for quantization: **all quant formats work on XPU** — mxfp4, nvfp4,
mxfp8, fp8, AWQ, GGUF k-quants, int8/w8a8, bitnet. They are data encodings +
dequant algorithms, not NVIDIA hardware features (NVIDIA marketing pushed the
"these are ours" myth). They decode to a compute dtype and/or feed Intel
XMX/DPAS int8/fp8 paths. Do not waste cycles reasoning about whether a format is
"supported" on Intel — assume it is in scope and implement the decode. A format
may be marked `unsupported` in `.quixicore/quant-formats.yaml` ONLY after an
actual on-B60 attempt shows a concrete blocker, never on inherited assumption.
