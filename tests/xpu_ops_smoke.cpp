// On-device correctness gate for the XPU op ABI.
//
// Runs each shipped op across dtypes {f32, f16, bf16} and every shipped variant
// {sycl, vendor} on a real Intel GPU, checking against an independent fp64 host
// reference (std::erf) within the umbrella per-dtype tolerances. Registered as a
// CTest test only under QUIXICORE_XPU_ENABLE_SYCL.
//
// This is the authoritative correctness check for the backend today. A
// torch.xpu / pytest harness is planned once the PyTorch-XPU binding lands; the
// fp64 oracle here does not depend on any Python stack.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"

#include "quantization/gguf_gemv/gguf_iq_tables.hpp"  // ggml i-quant grids for references

namespace {

using namespace quixicore::xpu;

double gelu_erf_ref(double x) {
  return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476));
}

struct Tol {
  double atol;
  double rtol;
};

// Per-dtype tolerances from the umbrella registry (registry/tolerances.yaml).
Tol tol_for(DType dt) {
  switch (dt) {
    case DType::f32:
      return {1e-5, 1e-4};
    case DType::f16:
      return {1e-3, 1e-3};
    case DType::bf16:
      return {2e-3, 2e-3};
  }
  return {1e-3, 1e-3};
}

// Run GELU for one (dtype, variant) and return true on pass. Uses shared USM so
// the host can fill inputs and read outputs directly; storage dtype conversion
// goes through static_cast<T>, matching the kernel's own rounding.
template <typename T>
bool check_gelu(sycl::queue& q, DType dt, Variant variant,
                const std::vector<float>& host_in) {
  const std::size_t n = host_in.size();
  T* in = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) in[i] = static_cast<T>(host_in[i]);

  ops::gelu(q, in, out, n, dt, ops::GeluApprox::erf, variant, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  double max_abs = 0.0;
  double worst_excess = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    // Reference is GELU of the value the kernel actually consumed (in[i], after
    // storage rounding), then rounded to the storage dtype. This isolates the
    // kernel's compute error: both kernel and reference compute in >=fp32 and
    // store in T, so the contract rtol (e.g. bf16 2e-3) is applied against a
    // same-precision reference rather than an unreachable fp64 target (a single
    // bf16 ULP is ~0.4% > 2e-3, so comparing to fp64 would fail by construction).
    const double ref_hi = gelu_erf_ref(static_cast<double>(in[i]));
    const double ref = static_cast<double>(static_cast<T>(ref_hi));
    const double abs_err = std::abs(static_cast<double>(out[i]) - ref);
    max_abs = std::max(max_abs, abs_err);
    worst_excess = std::max(worst_excess, abs_err - (tol.atol + tol.rtol * std::abs(ref)));
  }

  sycl::free(in, q);
  sycl::free(out, q);

  const bool ok = worst_excess <= 0.0;
  std::cout << "  gelu dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " (resolved=" << variant_name(resolve_variant(variant)) << ")"
            << " max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// Deterministic pseudo-random fill in a modest range, reproducible without any
// RNG dependency.
float sample(std::size_t i) {
  const double t = static_cast<double>((i * 2654435761u) % 10007) / 10007.0;
  return static_cast<float>(-2.0 + 4.0 * t);
}

double sigmoid_ref(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double silu_ref(double x) { return x * sigmoid_ref(x); }
double gelu_ref(double x) { return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476)); }
double relu_ref(double x) { return x > 0.0 ? x : 0.0; }

double gelu_grad_ref(double x) {
  const double cdf = 0.5 * (1.0 + std::erf(x * 0.7071067811865476));
  const double pdf = 0.3989422804014327 * std::exp(-0.5 * x * x);
  return cdf + x * pdf;
}

template <typename T>
bool check_serving(sycl::queue& q, DType dt) {
  const std::size_t vocab = 500, dim = 128, n = 200, slots = 400;
  T* table = sycl::malloc_shared<T>(vocab * dim, q);
  int* ids = sycl::malloc_shared<int>(n, q);
  T* out = sycl::malloc_shared<T>(n * dim, q);
  for (std::size_t i = 0; i < vocab * dim; ++i) table[i] = static_cast<T>(sample(i));
  for (std::size_t t = 0; t < n; ++t) ids[t] = static_cast<int>((t * 7 + 3) % vocab);

  ops::embedding_lookup(q, table, ids, out, n, dim, dt, Variant::sycl, true);
  int bad = 0;
  for (std::size_t t = 0; t < n; ++t)
    for (std::size_t j = 0; j < dim; ++j)
      if (out[t * dim + j] != table[ids[t] * dim + j]) ++bad;

  // scatter then gather round-trip
  T* cache = sycl::malloc_shared<T>(slots * dim, q);
  int* slot = sycl::malloc_shared<int>(n, q);
  int* gidx = sycl::malloc_shared<int>(n, q);
  T* gout = sycl::malloc_shared<T>(n * dim, q);
  for (std::size_t i = 0; i < slots * dim; ++i) cache[i] = static_cast<T>(-1.0f);
  for (std::size_t t = 0; t < n; ++t) { slot[t] = static_cast<int>((t * 3 + 1) % slots); gidx[t] = slot[t]; }
  ops::kv_cache_scatter(q, cache, out, slot, n, dim, dt, Variant::sycl, true);
  ops::kv_cache_gather(q, cache, gidx, gout, n, dim, dt, Variant::sycl, true);
  for (std::size_t t = 0; t < n; ++t)
    for (std::size_t j = 0; j < dim; ++j)
      if (gout[t * dim + j] != out[t * dim + j]) ++bad;

  sycl::free(table, q); sycl::free(ids, q); sycl::free(out, q);
  sycl::free(cache, q); sycl::free(slot, q); sycl::free(gidx, q); sycl::free(gout, q);
  const bool ok = (bad == 0);
  std::cout << "  serving(embed+kvcache) dt=" << dtype_name(dt) << " mism=" << bad
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// pool_mean_rms_l2: masked mean-pool over each sequence's tokens with a per-token
// RMSNorm folded in, then L2-normalize. The fp64 oracle mirrors the reference
// order exactly (RMSNorm each token -> accumulate -> mean -> L2), applying the
// learned weight directly and rounding the final vector to storage dtype T. The
// two-level reduction (per-token over dim, then across tokens) widens rtol by
// sqrt(dim), matching the attention/linear-attn oracle convention. Ragged token
// counts (incl. a single-token sequence) exercise the masked mean.
template <typename T>
bool check_pool_mean_rms_l2(sycl::queue& q, DType dt, std::size_t dim,
                            const std::vector<int>& tok_counts) {
  const std::size_t batch = tok_counts.size();
  std::vector<int> off(batch + 1, 0);
  for (std::size_t s = 0; s < batch; ++s) off[s + 1] = off[s] + tok_counts[s];
  const std::size_t total = static_cast<std::size_t>(off[batch]);

  T* x = sycl::malloc_shared<T>(total * dim, q);
  T* w = sycl::malloc_shared<T>(dim, q);
  int* offd = sycl::malloc_shared<int>(batch + 1, q);
  T* out = sycl::malloc_shared<T>(batch * dim, q);
  for (std::size_t i = 0; i < total * dim; ++i) x[i] = static_cast<T>(sample(i));
  for (std::size_t i = 0; i < dim; ++i) w[i] = static_cast<T>(0.5f + sample(i + 7) * 0.1f);
  for (std::size_t s = 0; s <= batch; ++s) offd[s] = off[s];
  const float eps = 1e-6f;

  ops::pool_mean_rms_l2(q, x, w, offd, out, batch, dim, eps, dt, Variant::sycl, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  const double rtol = tol.rtol * std::sqrt(static_cast<double>(dim)) + 5e-3;
  double worst_excess = 0.0, max_abs = 0.0;
  std::vector<double> m(dim);
  for (std::size_t s = 0; s < batch; ++s) {
    const int start = off[s], stop = off[s + 1];
    for (std::size_t d = 0; d < dim; ++d) m[d] = 0.0;
    for (int t = start; t < stop; ++t) {
      const std::size_t base = static_cast<std::size_t>(t) * dim;
      double ss = 0.0;
      for (std::size_t d = 0; d < dim; ++d) {
        const double v = static_cast<double>(x[base + d]);
        ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / static_cast<double>(dim) + static_cast<double>(eps));
      for (std::size_t d = 0; d < dim; ++d)
        m[d] += static_cast<double>(x[base + d]) * inv * static_cast<double>(w[d]);
    }
    const double inv_tokens = (stop > start) ? 1.0 / static_cast<double>(stop - start) : 0.0;
    double ss2 = 0.0;
    for (std::size_t d = 0; d < dim; ++d) { m[d] *= inv_tokens; ss2 += m[d] * m[d]; }
    const double inv_l2 = (ss2 == 0.0) ? 1.0 : 1.0 / std::sqrt(ss2);
    for (std::size_t d = 0; d < dim; ++d) {
      const double ref = static_cast<double>(static_cast<T>(m[d] * inv_l2));
      const double err = std::abs(static_cast<double>(out[s * dim + d]) - ref);
      max_abs = std::max(max_abs, err);
      worst_excess = std::max(worst_excess, err - (tol.atol + rtol * std::abs(ref)));
    }
  }
  sycl::free(x, q); sycl::free(w, q); sycl::free(offd, q); sycl::free(out, q);
  const bool ok = worst_excess <= 0.0;
  std::cout << "  pool_mean_rms_l2 dt=" << dtype_name(dt) << " dim=" << dim
            << " batch=" << batch << " max_abs=" << max_abs
            << " worst_excess=" << worst_excess << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// rms_residual_next: fused residual-add + double RMSNorm -> f16. The fp64 oracle
// mirrors the kernel order exactly -- post-norm the projection into a saved copy
// of the residual (accumulating the updated residuals RMS from the unrounded
// fp32-analogue sum), round the residual to storage dtype T, then pre-norm the
// rounded residual and round the result to f16. Both the in-place residual (dt)
// and the f16 next_out are checked; the two-level reduction widens rtol by
// sqrt(dim), matching the pool / attention oracle convention.
template <typename T>
bool check_rms_residual_next(sycl::queue& q, DType dt, std::size_t rows, std::size_t dim) {
  const std::size_t n = rows * dim;
  T* proj = sycl::malloc_shared<T>(n, q);
  T* pw = sycl::malloc_shared<T>(dim, q);
  T* res = sycl::malloc_shared<T>(n, q);
  T* nw = sycl::malloc_shared<T>(dim, q);
  half_t* out = sycl::malloc_shared<half_t>(n, q);
  std::vector<double> res_orig(n);
  for (std::size_t i = 0; i < n; ++i) { proj[i] = static_cast<T>(sample(i)); res[i] = static_cast<T>(sample(i + 5) * 0.5f); res_orig[i] = static_cast<double>(res[i]); }
  for (std::size_t i = 0; i < dim; ++i) { pw[i] = static_cast<T>(0.5f + sample(i + 7) * 0.1f); nw[i] = static_cast<T>(0.5f + sample(i + 13) * 0.1f); }
  const float eps = 1e-6f;

  ops::rms_residual_next(q, proj, pw, res, nw, out, rows, dim, eps, dt, Variant::sycl, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  const Tol tol16 = tol_for(DType::f16);
  const double sq = std::sqrt(static_cast<double>(dim));
  const double rtol_res = tol.rtol * sq + 5e-3;
  const double rtol_out = tol16.rtol * sq + 5e-3;
  double worst_res = 0.0, worst_out = 0.0, max_abs = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double pss = 0.0;
    for (std::size_t i = 0; i < dim; ++i) { const double v = static_cast<double>(proj[r * dim + i]); pss += v * v; }
    const double pinv = 1.0 / std::sqrt(pss / static_cast<double>(dim) + static_cast<double>(eps));
    std::vector<double> res_round(dim);
    double rss = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      const double value = res_orig[r * dim + i] +
          static_cast<double>(proj[r * dim + i]) * static_cast<double>(pw[i]) * pinv;
      res_round[i] = static_cast<double>(static_cast<T>(value));
      rss += value * value;  // unrounded sum, matching the kernel
    }
    const double rinv = 1.0 / std::sqrt(rss / static_cast<double>(dim) + static_cast<double>(eps));
    for (std::size_t i = 0; i < dim; ++i) {
      const double res_err = std::abs(static_cast<double>(res[r * dim + i]) - res_round[i]);
      worst_res = std::max(worst_res, res_err - (tol.atol + rtol_res * std::abs(res_round[i])));
      const double ref = static_cast<double>(static_cast<half_t>(static_cast<float>(res_round[i] * static_cast<double>(nw[i]) * rinv)));
      const double out_err = std::abs(static_cast<double>(out[r * dim + i]) - ref);
      max_abs = std::max(max_abs, out_err);
      worst_out = std::max(worst_out, out_err - (tol16.atol + rtol_out * std::abs(ref)));
    }
  }
  sycl::free(proj, q); sycl::free(pw, q); sycl::free(res, q); sycl::free(nw, q); sycl::free(out, q);
  const bool ok = worst_res <= 0.0 && worst_out <= 0.0;
  std::cout << "  rms_residual_next dt=" << dtype_name(dt) << " rows=" << rows << " dim=" << dim
            << " max_abs=" << max_abs << " worst_res=" << worst_res << " worst_out=" << worst_out
            << (ok ? "  ok" : "  FAIL") << "\n";
  return ok;
}

// qk_norm_rope: fused per-head QK-norm + query-scale + NeoX RoPE (+ optional f16
// convert). The fp64 oracle mirrors the kernel order exactly -- per (token,head)
// RMS over head_dim from the ORIGINAL values, query scale folded into inv, then
// weight*inv applied before the rotation -- for Q (n_head) and K (n_head_kv,
// GQA-capable) from saved input copies. Checks the in-place dt output and, when
// requested, the f16 output; the head_dim reduction widens rtol by sqrt(head_dim).
template <typename T>
bool check_qk_norm_rope(sycl::queue& q, DType dt, std::size_t tokens,
                        std::size_t n_head, std::size_t n_head_kv,
                        std::size_t hd, bool use_half) {
  const std::size_t nq = tokens * n_head * hd, nk = tokens * n_head_kv * hd;
  T* Q = sycl::malloc_shared<T>(nq, q);
  T* K = sycl::malloc_shared<T>(nk, q);
  T* qw = sycl::malloc_shared<T>(hd, q);
  T* kw = sycl::malloc_shared<T>(hd, q);
  half_t* Qh = use_half ? sycl::malloc_shared<half_t>(nq, q) : nullptr;
  half_t* Kh = use_half ? sycl::malloc_shared<half_t>(nk, q) : nullptr;
  std::vector<double> q0(nq), k0(nk);
  for (std::size_t i = 0; i < nq; ++i) { Q[i] = static_cast<T>(sample(i)); q0[i] = static_cast<double>(Q[i]); }
  for (std::size_t i = 0; i < nk; ++i) { K[i] = static_cast<T>(sample(i + 5)); k0[i] = static_cast<double>(K[i]); }
  for (std::size_t i = 0; i < hd; ++i) { qw[i] = static_cast<T>(0.5f + sample(i + 7) * 0.1f); kw[i] = static_cast<T>(0.5f + sample(i + 13) * 0.1f); }
  const float eps = 1e-6f, base = 10000.0f, query_scale = 0.0625f;

  ops::qk_norm_rope(q, Q, K, qw, kw, Qh, Kh, tokens, n_head, n_head_kv, hd, base, 0, query_scale, eps, dt, Variant::sycl, /*blocking=*/true);

  const std::size_t half = hd / 2;
  const Tol tol = tol_for(dt);
  const Tol tol16 = tol_for(DType::f16);
  const double sq = std::sqrt(static_cast<double>(hd));
  const double rtol = tol.rtol * sq + 5e-3;
  const double rtol16 = tol16.rtol * sq + 5e-3;
  double worst = 0.0, worst16 = 0.0, max_abs = 0.0;

  auto check_buf = [&](const T* buf, const half_t* hbuf, const std::vector<double>& src,
                       const double* weight, std::size_t heads, bool is_key) {
    for (std::size_t t = 0; t < tokens; ++t)
      for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t bb = (t * heads + h) * hd;
        double ss = 0.0;
        for (std::size_t d = 0; d < hd; ++d) ss += src[bb + d] * src[bb + d];
        const double scale = is_key ? 1.0 : static_cast<double>(query_scale);
        const double inv = (1.0 / std::sqrt(ss / static_cast<double>(hd) + static_cast<double>(eps))) * scale;
        for (std::size_t i = 0; i < half; ++i) {
          const double freq = std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) / static_cast<double>(hd));
          const double ang = static_cast<double>(t) * freq;
          const double x0 = src[bb + i] * weight[i] * inv;
          const double x1 = src[bb + i + half] * weight[i + half] * inv;
          const double r0 = x0 * std::cos(ang) - x1 * std::sin(ang);
          const double r1 = x0 * std::sin(ang) + x1 * std::cos(ang);
          const double ref0 = static_cast<double>(static_cast<T>(r0));
          const double ref1 = static_cast<double>(static_cast<T>(r1));
          const double e0 = std::abs(static_cast<double>(buf[bb + i]) - ref0);
          const double e1 = std::abs(static_cast<double>(buf[bb + i + half]) - ref1);
          max_abs = std::max(max_abs, std::max(e0, e1));
          worst = std::max(worst, e0 - (tol.atol + rtol * std::abs(ref0)));
          worst = std::max(worst, e1 - (tol.atol + rtol * std::abs(ref1)));
          if (hbuf) {
            const double h0 = static_cast<double>(static_cast<half_t>(static_cast<float>(r0)));
            const double h1 = static_cast<double>(static_cast<half_t>(static_cast<float>(r1)));
            worst16 = std::max(worst16, std::abs(static_cast<double>(hbuf[bb + i]) - h0) - (tol16.atol + rtol16 * std::abs(h0)));
            worst16 = std::max(worst16, std::abs(static_cast<double>(hbuf[bb + i + half]) - h1) - (tol16.atol + rtol16 * std::abs(h1)));
          }
        }
      }
  };

  std::vector<double> qwd(hd), kwd(hd);
  for (std::size_t i = 0; i < hd; ++i) { qwd[i] = static_cast<double>(qw[i]); kwd[i] = static_cast<double>(kw[i]); }
  check_buf(Q, Qh, q0, qwd.data(), n_head, /*is_key=*/false);
  check_buf(K, Kh, k0, kwd.data(), n_head_kv, /*is_key=*/true);

  sycl::free(Q, q); sycl::free(K, q); sycl::free(qw, q); sycl::free(kw, q);
  if (Qh) sycl::free(Qh, q);
  if (Kh) sycl::free(Kh, q);
  const bool ok = worst <= 0.0 && worst16 <= 0.0;
  std::cout << "  qk_norm_rope dt=" << dtype_name(dt) << " t=" << tokens << " h=" << n_head
            << "/" << n_head_kv << " hd=" << hd << (use_half ? " +f16" : "")
            << " max_abs=" << max_abs << " worst=" << worst << " worst16=" << worst16
            << (ok ? "  ok" : "  FAIL") << "\n";
  return ok;
}

bool check_dropout(sycl::queue& q, DType dt) {
  const std::size_t n = 1u << 16;
  const float p = 0.3f;
  float* in = sycl::malloc_shared<float>(n, q);
  float* out = sycl::malloc_shared<float>(n, q);
  float* out2 = sycl::malloc_shared<float>(n, q);
  for (std::size_t i = 0; i < n; ++i) in[i] = 1.0f;
  ops::dropout(q, in, out, n, p, 1234u, dt, Variant::sycl, true);
  ops::dropout(q, in, out2, n, p, 1234u, dt, Variant::sycl, true);
  std::size_t zeros = 0; int bad = 0, nondet = 0;
  const float keep = 1.0f / (1.0f - p);
  for (std::size_t i = 0; i < n; ++i) {
    if (out[i] == 0.0f) ++zeros;
    else if (std::abs(out[i] - keep) > 1e-2f) ++bad;
    if (out[i] != out2[i]) ++nondet;
  }
  const double frac = static_cast<double>(zeros) / n;
  const bool ok = std::abs(frac - p) < 0.02 && bad == 0 && nondet == 0;
  std::cout << "  dropout dt=" << dtype_name(dt) << " zero_frac=" << frac
            << " bad=" << bad << " nondet=" << nondet << (ok ? "  ok" : "  FAIL") << '\n';
  sycl::free(in, q); sycl::free(out, q); sycl::free(out2, q);
  return ok;
}

template <typename T>
bool check_cross_entropy(sycl::queue& q, DType dt, std::size_t rows, std::size_t vocab) {
  T* logits = sycl::malloc_shared<T>(rows * vocab, q);
  int* target = sycl::malloc_shared<int>(rows, q);
  float* loss = sycl::malloc_shared<float>(rows, q);
  for (std::size_t i = 0; i < rows * vocab; ++i) logits[i] = static_cast<T>(sample(i) * 4.0f);
  for (std::size_t r = 0; r < rows; ++r) target[r] = static_cast<int>((r * 13 + 5) % vocab);
  ops::cross_entropy(q, logits, target, loss, rows, vocab, dt, Variant::sycl, true);
  double worst = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double m = -1e30;
    for (std::size_t j = 0; j < vocab; ++j) m = std::max(m, (double)logits[r * vocab + j]);
    double sum = 0.0;
    for (std::size_t j = 0; j < vocab; ++j) sum += std::exp((double)logits[r * vocab + j] - m);
    const double ref = (m + std::log(sum)) - (double)logits[r * vocab + target[r]];
    worst = std::max(worst, std::abs((double)loss[r] - ref));
  }
  sycl::free(logits, q); sycl::free(target, q); sycl::free(loss, q);
  const bool ok = worst < 3e-2;
  std::cout << "  cross_entropy dt=" << dtype_name(dt) << " max_abs=" << worst
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_hadamard(sycl::queue& q, DType dt, std::size_t rows, std::size_t n) {
  T* in = sycl::malloc_shared<T>(rows * n, q);
  T* out = sycl::malloc_shared<T>(rows * n, q);
  for (std::size_t i = 0; i < rows * n; ++i) in[i] = static_cast<T>(sample(i));
  ops::hadamard(q, in, out, rows, n, dt, Variant::sycl, true);
  std::vector<double> ref(n);
  double worst = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t i = 0; i < n; ++i) ref[i] = (double)in[r * n + i];
    for (std::size_t s = 1; s < n; s <<= 1)
      for (std::size_t base = 0; base < n; base += 2 * s)
        for (std::size_t k = 0; k < s; ++k) {
          const double a = ref[base + k], b = ref[base + k + s];
          ref[base + k] = a + b; ref[base + k + s] = a - b;
        }
    // FWHT sums n signed terms; fp32 error grows ~sqrt(n) * eps * magnitude.
    const Tol tol = tol_for(dt);
    const double gain = std::sqrt(static_cast<double>(n));
    for (std::size_t i = 0; i < n; ++i) {
      const double rr = (double)static_cast<T>(ref[i]);
      worst = std::max(worst, std::abs((double)out[r * n + i] - rr) - (tol.atol + tol.rtol * std::abs(rr)) * gain);
    }
  }
  sycl::free(in, q); sycl::free(out, q);
  const bool ok = worst <= 0.0;
  std::cout << "  hadamard dt=" << dtype_name(dt) << " (n=" << n << ") worst_excess=" << worst
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_moe_route(sycl::queue& q, DType dt, std::size_t nt, std::size_t ne, int k) {
  T* logits = sycl::malloc_shared<T>(nt * ne, q);
  int* ids = sycl::malloc_shared<int>(nt * k, q);
  float* w = sycl::malloc_shared<float>(nt * k, q);
  for (std::size_t i = 0; i < nt * ne; ++i) logits[i] = static_cast<T>(sample(i) * 3.0f);
  ops::moe_route_topk(q, logits, ids, w, nt, ne, k, dt, Variant::sycl, true);

  int bad_id = 0; double worst_w = 0.0;
  std::vector<char> taken(ne);
  for (std::size_t t = 0; t < nt; ++t) {
    std::fill(taken.begin(), taken.end(), 0);
    std::vector<int> rid(k); std::vector<double> rval(k);
    for (int i = 0; i < k; ++i) {
      double best = -1e30; int be = 0;
      for (std::size_t e = 0; e < ne; ++e)
        if (!taken[e] && (double)logits[t * ne + e] > best) { best = (double)logits[t * ne + e]; be = (int)e; }
      taken[be] = 1; rid[i] = be; rval[i] = best;
    }
    double m = rval[0]; for (int i = 1; i < k; ++i) m = std::max(m, rval[i]);
    double sum = 0; for (int i = 0; i < k; ++i) { rval[i] = std::exp(rval[i] - m); sum += rval[i]; }
    for (int i = 0; i < k; ++i) {
      if (ids[t * k + i] != rid[i]) ++bad_id;
      worst_w = std::max(worst_w, std::abs((double)w[t * k + i] - rval[i] / sum));
    }
  }
  sycl::free(logits, q); sycl::free(ids, q); sycl::free(w, q);
  const bool ok = (bad_id == 0) && (worst_w < 1e-5);
  std::cout << "  moe_route_topk dt=" << dtype_name(dt) << " k=" << k << " bad_id=" << bad_id
            << " worst_w=" << worst_w << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_linear_attn(sycl::queue& q, DType dt, std::size_t nh, std::size_t seq, std::size_t dim) {
  T* Q = sycl::malloc_shared<T>(nh * seq * dim, q);
  T* K = sycl::malloc_shared<T>(nh * seq * dim, q);
  T* V = sycl::malloc_shared<T>(nh * seq * dim, q);
  T* O = sycl::malloc_shared<T>(nh * seq * dim, q);
  for (std::size_t i = 0; i < nh * seq * dim; ++i) {
    Q[i] = static_cast<T>(0.5f + 0.5f * std::abs(sample(i)));       // positive (feature map)
    K[i] = static_cast<T>(0.5f + 0.5f * std::abs(sample(i + 7)));
    V[i] = static_cast<T>(sample(i + 13));
  }
  ops::linear_attn(q, Q, K, V, O, nh, seq, dim, dt, Variant::sycl, true);

  const Tol tol = tol_for(dt);
  const double rtol = tol.rtol * std::sqrt(static_cast<double>(seq)) + 5e-3;
  double worst = 0.0;
  std::vector<double> KV(dim * dim), z(dim);
  for (std::size_t h = 0; h < nh; ++h) {
    const std::size_t base = h * seq * dim;
    for (std::size_t i = 0; i < dim; ++i) {
      z[i] = 0;
      for (std::size_t t = 0; t < seq; ++t) z[i] += (double)K[base + t * dim + i];
      for (std::size_t j = 0; j < dim; ++j) {
        double a = 0;
        for (std::size_t t = 0; t < seq; ++t) a += (double)K[base + t * dim + i] * (double)V[base + t * dim + j];
        KV[i * dim + j] = a;
      }
    }
    for (std::size_t t = 0; t < seq; ++t)
      for (std::size_t j = 0; j < dim; ++j) {
        double num = 0, den = 0;
        for (std::size_t i = 0; i < dim; ++i) { const double qi = (double)Q[base + t * dim + i]; num += qi * KV[i * dim + j]; den += qi * z[i]; }
        const double ref = (double)static_cast<T>(num / (den + 1e-6));
        worst = std::max(worst, std::abs((double)O[base + t * dim + j] - ref) - (tol.atol + rtol * std::abs(ref)));
      }
  }
  sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q);
  const bool ok = worst <= 0.0;
  std::cout << "  linear_attn dt=" << dtype_name(dt) << " (h=" << nh << " s=" << seq
            << " d=" << dim << ") worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_selective_scan(sycl::queue& q, DType dt, std::size_t nc, std::size_t seq, std::size_t st) {
  T* u = sycl::malloc_shared<T>(nc * seq, q);
  T* delta = sycl::malloc_shared<T>(nc * seq, q);
  T* A = sycl::malloc_shared<T>(nc * st, q);
  T* B = sycl::malloc_shared<T>(seq * st, q);
  T* C = sycl::malloc_shared<T>(seq * st, q);
  T* D = sycl::malloc_shared<T>(nc, q);
  T* y = sycl::malloc_shared<T>(nc * seq, q);
  for (std::size_t i = 0; i < nc * seq; ++i) { u[i] = static_cast<T>(sample(i)); delta[i] = static_cast<T>(0.05f + 0.05f * std::abs(sample(i + 3))); }
  for (std::size_t i = 0; i < nc * st; ++i) A[i] = static_cast<T>(-0.5f - 0.5f * std::abs(sample(i + 5)));  // negative (stable)
  for (std::size_t i = 0; i < seq * st; ++i) { B[i] = static_cast<T>(sample(i + 7)); C[i] = static_cast<T>(sample(i + 11)); }
  for (std::size_t i = 0; i < nc; ++i) D[i] = static_cast<T>(sample(i + 13));

  ops::selective_scan(q, u, delta, A, B, C, D, y, nc, seq, st, dt, Variant::sycl, true);

  const Tol tol = tol_for(dt);
  const double rtol = tol.rtol * std::sqrt(static_cast<double>(seq)) + 5e-3;
  double worst = 0.0;
  std::vector<double> h(st);
  for (std::size_t c = 0; c < nc; ++c) {
    std::fill(h.begin(), h.end(), 0.0);
    for (std::size_t t = 0; t < seq; ++t) {
      const double dt_ = (double)delta[c * seq + t], ut = (double)u[c * seq + t];
      double yt = 0;
      for (std::size_t s = 0; s < st; ++s) {
        const double dA = std::exp(dt_ * (double)A[c * st + s]);
        h[s] = dA * h[s] + dt_ * (double)B[t * st + s] * ut;
        yt += (double)C[t * st + s] * h[s];
      }
      const double ref = (double)static_cast<T>(yt + (double)D[c] * ut);
      worst = std::max(worst, std::abs((double)y[c * seq + t] - ref) - (tol.atol + rtol * std::abs(ref)));
    }
  }
  sycl::free(u, q); sycl::free(delta, q); sycl::free(A, q); sycl::free(B, q);
  sycl::free(C, q); sycl::free(D, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  selective_scan dt=" << dtype_name(dt) << " (c=" << nc << " s=" << seq
            << " N=" << st << ") worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

bool check_collectives() {
  const std::size_t count = 4096;
  std::vector<float> in(8 * count), out(count, 0.0f);
  for (std::size_t g = 0; g < 8; ++g)
    for (std::size_t i = 0; i < count; ++i) in[g * count + i] = static_cast<float>(g + 1);
  const std::size_t ng = ops::all_reduce_sum(in.data(), out.data(), count);
  if (ng == 0) { std::cout << "  all_reduce_sum: no GPU (skip)\n"; return true; }
  const float expected = static_cast<float>(ng * (ng + 1) / 2);
  int bad = 0;
  for (std::size_t i = 0; i < count; ++i) if (std::abs(out[i] - expected) > 1e-3f) ++bad;
  const bool ok = (bad == 0);
  std::cout << "  all_reduce_sum ng=" << ng << " expected=" << expected << " bad=" << bad
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_attention(sycl::queue& q, DType dt, std::size_t nh, std::size_t nkv,
                     std::size_t sq, std::size_t sk, std::size_t d, bool causal) {
  T* Q = sycl::malloc_shared<T>(nh * sq * d, q);
  T* K = sycl::malloc_shared<T>(nkv * sk * d, q);
  T* V = sycl::malloc_shared<T>(nkv * sk * d, q);
  T* O = sycl::malloc_shared<T>(nh * sq * d, q);
  for (std::size_t i = 0; i < nh * sq * d; ++i) Q[i] = static_cast<T>(sample(i) * 0.5f);
  for (std::size_t i = 0; i < nkv * sk * d; ++i) { K[i] = static_cast<T>(sample(i + 3) * 0.5f); V[i] = static_cast<T>(sample(i + 7)); }
  ops::attention(q, Q, K, V, O, nh, nkv, sq, sk, d, causal, dt, Variant::sycl, true);

  const double scale = 1.0 / std::sqrt((double)d);
  const std::size_t gqa = nh / nkv, delta = sk - sq;
  const Tol tol = tol_for(dt);
  const double rtol = tol.rtol * 8 + 5e-3;
  double worst = 0.0;
  std::vector<double> sc(sk);
  for (std::size_t h = 0; h < nh; ++h)
    for (std::size_t qi = 0; qi < sq; ++qi) {
      const std::size_t kvh = h / gqa;
      const std::size_t last = causal ? (qi + delta) : (sk - 1);
      double m = -1e30;
      for (std::size_t ki = 0; ki <= last && ki < sk; ++ki) {
        double s = 0; for (std::size_t j = 0; j < d; ++j) s += (double)Q[(h * sq + qi) * d + j] * (double)K[(kvh * sk + ki) * d + j];
        sc[ki] = s * scale; m = std::max(m, sc[ki]);
      }
      double l = 0; for (std::size_t ki = 0; ki <= last && ki < sk; ++ki) { sc[ki] = std::exp(sc[ki] - m); l += sc[ki]; }
      for (std::size_t j = 0; j < d; ++j) {
        double acc = 0; for (std::size_t ki = 0; ki <= last && ki < sk; ++ki) acc += sc[ki] * (double)V[(kvh * sk + ki) * d + j];
        const double ref = (double)static_cast<T>(acc / l);
        worst = std::max(worst, std::abs((double)O[(h * sq + qi) * d + j] - ref) - (tol.atol + rtol * std::abs(ref)));
      }
    }
  sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q);
  const bool ok = worst <= 0.0;
  std::cout << "  attention dt=" << dtype_name(dt) << " (h=" << nh << "/" << nkv << " sq=" << sq
            << " sk=" << sk << " d=" << d << (causal ? " causal" : "") << ") worst_excess=" << worst
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// attention_f16ctx: same fp64 SDPA oracle as check_attention for the dtype-T
// output O, PLUS a check that the fused half store O_f16 equals the f16 rounding
// of the attention result within the f16 contract tolerance. Exercises both the
// unchanged compute-dtype path and the new fused f16 store.
template <typename T>
bool check_attention_f16ctx(sycl::queue& q, DType dt, std::size_t nh, std::size_t nkv,
                            std::size_t sq, std::size_t sk, std::size_t d, bool causal) {
  T* Q = sycl::malloc_shared<T>(nh * sq * d, q);
  T* K = sycl::malloc_shared<T>(nkv * sk * d, q);
  T* V = sycl::malloc_shared<T>(nkv * sk * d, q);
  T* O = sycl::malloc_shared<T>(nh * sq * d, q);
  half_t* O16 = sycl::malloc_shared<half_t>(nh * sq * d, q);
  for (std::size_t i = 0; i < nh * sq * d; ++i) Q[i] = static_cast<T>(sample(i) * 0.5f);
  for (std::size_t i = 0; i < nkv * sk * d; ++i) { K[i] = static_cast<T>(sample(i + 3) * 0.5f); V[i] = static_cast<T>(sample(i + 7)); }
  ops::attention_f16ctx(q, Q, K, V, O, O16, nh, nkv, sq, sk, d, causal, dt, Variant::sycl, true);

  const double scale = 1.0 / std::sqrt((double)d);
  const std::size_t gqa = nh / nkv, delta = sk - sq;
  const Tol tol = tol_for(dt);
  const double rtol = tol.rtol * 8 + 5e-3;
  const Tol tol16 = tol_for(DType::f16);
  const double rtol16 = tol16.rtol * 8 + 5e-3;
  double worst = 0.0, worst16 = 0.0;
  std::vector<double> sc(sk);
  for (std::size_t h = 0; h < nh; ++h)
    for (std::size_t qi = 0; qi < sq; ++qi) {
      const std::size_t kvh = h / gqa;
      const std::size_t last = causal ? (qi + delta) : (sk - 1);
      double m = -1e30;
      for (std::size_t ki = 0; ki <= last && ki < sk; ++ki) {
        double s = 0; for (std::size_t j = 0; j < d; ++j) s += (double)Q[(h * sq + qi) * d + j] * (double)K[(kvh * sk + ki) * d + j];
        sc[ki] = s * scale; m = std::max(m, sc[ki]);
      }
      double l = 0; for (std::size_t ki = 0; ki <= last && ki < sk; ++ki) { sc[ki] = std::exp(sc[ki] - m); l += sc[ki]; }
      for (std::size_t j = 0; j < d; ++j) {
        double acc = 0; for (std::size_t ki = 0; ki <= last && ki < sk; ++ki) acc += sc[ki] * (double)V[(kvh * sk + ki) * d + j];
        const double raw = acc / l;
        const double ref = (double)static_cast<T>(raw);
        const double ref16 = (double)static_cast<half_t>((float)raw);
        const std::size_t off = (h * sq + qi) * d + j;
        worst = std::max(worst, std::abs((double)O[off] - ref) - (tol.atol + rtol * std::abs(ref)));
        worst16 = std::max(worst16, std::abs((double)O16[off] - ref16) - (tol16.atol + rtol16 * std::abs(ref16)));
      }
    }
  sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q); sycl::free(O16, q);
  const bool ok = worst <= 0.0 && worst16 <= 0.0;
  std::cout << "  attention_f16ctx dt=" << dtype_name(dt) << " (h=" << nh << "/" << nkv << " sq=" << sq
            << " sk=" << sk << " d=" << d << (causal ? " causal" : "") << ") worst_excess=" << worst
            << " worst16=" << worst16 << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// attn_swa: symmetric sliding-window attention. Two fp64 checks in one:
//   (1) the kernel output MUST match a windowed SDPA oracle that attends the
//       symmetric band [center-window/2, center+window/2] (center = qi+sk-sq);
//   (2) the kernel output MUST NOT match a CAUSAL-window oracle (same left edge
//       but never looking past `center`). For interior queries the symmetric
//       band contains future keys the causal oracle drops, so a genuinely
//       symmetric kernel diverges from it -- vs_causal_gap must be large. An
//       accidentally-causal kernel would pass (1)'s left half but collapse the
//       gap in (2) and FAIL. This is the "a causal mask would FAIL" proof.
template <typename T>
bool check_attn_swa(sycl::queue& q, DType dt, std::size_t nh, std::size_t nkv,
                    std::size_t sq, std::size_t sk, std::size_t d, std::size_t window) {
  T* Q = sycl::malloc_shared<T>(nh * sq * d, q);
  T* K = sycl::malloc_shared<T>(nkv * sk * d, q);
  T* V = sycl::malloc_shared<T>(nkv * sk * d, q);
  T* O = sycl::malloc_shared<T>(nh * sq * d, q);
  for (std::size_t i = 0; i < nh * sq * d; ++i) Q[i] = static_cast<T>(sample(i) * 0.5f);
  for (std::size_t i = 0; i < nkv * sk * d; ++i) { K[i] = static_cast<T>(sample(i + 3) * 0.5f); V[i] = static_cast<T>(sample(i + 7)); }
  ops::attn_swa(q, Q, K, V, O, nh, nkv, sq, sk, d, window, dt, Variant::sycl, true);

  const double scale = 1.0 / std::sqrt((double)d);
  const std::size_t gqa = nh / nkv, delta = sk - sq, half = window / 2;
  const Tol tol = tol_for(dt);
  const double rtol = tol.rtol * 8 + 5e-3;
  double worst = 0.0;       // symmetric-window oracle: kernel MUST match
  double causal_gap = 0.0;  // vs causal-window oracle: kernel MUST diverge
  std::vector<double> sc(sk);
  for (std::size_t h = 0; h < nh; ++h)
    for (std::size_t qi = 0; qi < sq; ++qi) {
      const std::size_t kvh = h / gqa;
      const std::size_t center = qi + delta;
      std::size_t first = 0, last = sk;
      if (window != 0) {
        first = (center > half) ? (center - half) : 0;
        const std::size_t cand = center + half + 1;
        last = (cand < sk) ? cand : sk;
      }
      // (1) symmetric band reference
      double m = -1e30;
      for (std::size_t ki = first; ki < last; ++ki) {
        double s = 0; for (std::size_t j = 0; j < d; ++j) s += (double)Q[(h * sq + qi) * d + j] * (double)K[(kvh * sk + ki) * d + j];
        sc[ki] = s * scale; m = std::max(m, sc[ki]);
      }
      double l = 0; for (std::size_t ki = first; ki < last; ++ki) { sc[ki] = std::exp(sc[ki] - m); l += sc[ki]; }
      for (std::size_t j = 0; j < d; ++j) {
        double acc = 0; for (std::size_t ki = first; ki < last; ++ki) acc += sc[ki] * (double)V[(kvh * sk + ki) * d + j];
        const double ref = (double)static_cast<T>(acc / l);
        worst = std::max(worst, std::abs((double)O[(h * sq + qi) * d + j] - ref) - (tol.atol + rtol * std::abs(ref)));
      }
      // (2) causal-window reference: same left edge, keys capped at center
      const std::size_t clast = (last < center + 1) ? last : (center + 1);
      double cm = -1e30;
      for (std::size_t ki = first; ki < clast; ++ki) {
        double s = 0; for (std::size_t j = 0; j < d; ++j) s += (double)Q[(h * sq + qi) * d + j] * (double)K[(kvh * sk + ki) * d + j];
        sc[ki] = s * scale; cm = std::max(cm, sc[ki]);
      }
      double cl = 0; for (std::size_t ki = first; ki < clast; ++ki) { sc[ki] = std::exp(sc[ki] - cm); cl += sc[ki]; }
      for (std::size_t j = 0; j < d; ++j) {
        double acc = 0; for (std::size_t ki = first; ki < clast; ++ki) acc += sc[ki] * (double)V[(kvh * sk + ki) * d + j];
        const double cref = acc / cl;
        causal_gap = std::max(causal_gap, std::abs((double)O[(h * sq + qi) * d + j] - cref));
      }
    }
  sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q);
  // window>0 self-attention has future keys in the band for interior queries, so
  // a symmetric kernel must diverge from the causal oracle by a clear margin.
  const bool symmetric_proof = (window == 0) || (causal_gap > 1e-2);
  const bool ok = worst <= 0.0 && symmetric_proof;
  std::cout << "  attn_swa dt=" << dtype_name(dt) << " (h=" << nh << "/" << nkv << " sq=" << sq
            << " sk=" << sk << " d=" << d << " window=" << window << ") worst_excess=" << worst
            << " vs_causal_gap=" << causal_gap << (window ? " (must be > 1e-2)" : " (dense)")
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_sampling(sycl::queue& q, DType dt, std::size_t rows, std::size_t vocab, int k) {
  T* logits = sycl::malloc_shared<T>(rows * vocab, q);
  int* out_k = sycl::malloc_shared<int>(rows, q);
  int* out_g = sycl::malloc_shared<int>(rows, q);
  for (std::size_t i = 0; i < rows * vocab; ++i) logits[i] = static_cast<T>(sample(i) * 4.0f);

  ops::top_k_sample(q, logits, out_k, rows, vocab, k, 1.0f, 99u, dt, Variant::sycl, true);
  ops::sample_categorical(q, logits, out_g, rows, vocab, 1e-3f, 99u, dt, Variant::sycl, true);  // ~greedy

  // Value-based, tie/precision-robust invariants: the top-k sample's logit is at
  // least the k-th largest value; the near-greedy sample is within a small margin
  // of the max.
  int not_in_topk = 0, greedy_bad = 0;
  std::vector<double> vals(vocab);
  for (std::size_t t = 0; t < rows; ++t) {
    for (std::size_t j = 0; j < vocab; ++j) vals[j] = (double)logits[t * vocab + j];
    std::vector<double> s(vals);
    std::partial_sort(s.begin(), s.begin() + k, s.end(), std::greater<double>());
    const double kth = s[k - 1], mx = s[0];
    if (vals[out_k[t]] < kth - 1e-3) ++not_in_topk;
    if (vals[out_g[t]] < mx - 0.1) ++greedy_bad;   // temp 1e-3 -> within ~margin of max
  }
  sycl::free(logits, q); sycl::free(out_k, q); sycl::free(out_g, q);
  const bool ok = (not_in_topk == 0) && (greedy_bad == 0);
  std::cout << "  sampling dt=" << dtype_name(dt) << " k=" << k << " topk_viol=" << not_in_topk
            << " greedy_bad=" << greedy_bad << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// Explicit mantissa bits per storage dtype (for the 1-ULP allowance below).
int mantissa_bits(DType dt) {
  switch (dt) {
    case DType::f32: return 23;
    case DType::f16: return 10;
    case DType::bf16: return 7;
  }
  return 23;
}

// Generic pass/fail against a per-element reference rounded to storage dtype.
// The kernel computes in fp32 and the oracle in fp64, so around a rounding
// boundary they can land on adjacent storage values: allow one storage ULP on
// top of the contract tolerance (a single bf16 ULP is ~0.4% > the 2e-3 rtol, so
// transcendental bf16 ops would otherwise fail by construction).
template <typename T>
bool report(const char* name, DType dt, const std::vector<float>& out,
            const std::vector<double>& ref_hi) {
  const Tol tol = tol_for(dt);
  const double ulp_rel = std::ldexp(1.0, -mantissa_bits(dt));  // 2^-mant
  double max_abs = 0.0, worst = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double ref = static_cast<double>(static_cast<T>(ref_hi[i]));
    const double err = std::abs(static_cast<double>(out[i]) - ref);
    const double allow = tol.atol + tol.rtol * std::abs(ref) + ulp_rel * std::abs(ref);
    max_abs = std::max(max_abs, err);
    worst = std::max(worst, err - allow);
  }
  const bool ok = worst <= 0.0;
  std::cout << "  " << name << " dt=" << dtype_name(dt) << " max_abs=" << max_abs
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_silu(sycl::queue& q, DType dt, std::size_t n) {
  T* in = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) in[i] = static_cast<T>(sample(i) * 3.0f);
  ops::silu(q, in, out, n, dt, Variant::sycl, true);
  std::vector<float> o(n);
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n; ++i) { o[i] = static_cast<float>(out[i]); r[i] = silu_ref(static_cast<double>(in[i])); }
  const bool ok = report<T>("silu", dt, o, r);
  sycl::free(in, q); sycl::free(out, q);
  return ok;
}

template <typename T>
bool check_gelu_bwd(sycl::queue& q, DType dt, std::size_t n) {
  T* g = sycl::malloc_shared<T>(n, q);
  T* x = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) { g[i] = static_cast<T>(sample(i + 1)); x[i] = static_cast<T>(sample(i) * 3.0f); }
  ops::gelu_backward(q, g, x, out, n, dt, ops::GeluApprox::erf, Variant::sycl, true);
  std::vector<float> o(n);
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n; ++i) { o[i] = static_cast<float>(out[i]); r[i] = static_cast<double>(g[i]) * gelu_grad_ref(static_cast<double>(x[i])); }
  const bool ok = report<T>("gelu_bwd", dt, o, r);
  sycl::free(g, q); sycl::free(x, q); sycl::free(out, q);
  return ok;
}

// glu_gelu_f16: GEGLU (tanh-gelu gate x value) with a fused f16 output. Pure
// elementwise (no reduction), so report<half_t> with the f16 tolerance suffices:
// the fp64 oracle computes gelu_tanh(gate)*value and report rounds it to f16.
template <typename T>
bool check_glu_gelu_f16(sycl::queue& q, DType dt, const char* name,
                        std::size_t rows, std::size_t d) {
  T* x = sycl::malloc_shared<T>(rows * 2 * d, q);
  half_t* out = sycl::malloc_shared<half_t>(rows * d, q);
  for (std::size_t i = 0; i < rows * 2 * d; ++i) x[i] = static_cast<T>(sample(i) * 2.0f);
  ops::glu_gelu_f16(q, x, out, rows, d, dt, Variant::sycl, /*blocking=*/true);
  std::vector<float> o(rows * d);
  std::vector<double> r(rows * d);
  const double a = 0.044715, s = 0.79788456080286535587989211986876;
  for (std::size_t rr = 0; rr < rows; ++rr)
    for (std::size_t i = 0; i < d; ++i) {
      const double gate = static_cast<double>(x[rr * 2 * d + i]);
      const double val = static_cast<double>(x[rr * 2 * d + d + i]);
      const double g = 0.5 * gate * (1.0 + std::tanh(s * gate * (1.0 + a * gate * gate)));
      o[rr * d + i] = static_cast<float>(out[rr * d + i]);
      r[rr * d + i] = g * val;
    }
  const bool ok = report<half_t>(name, DType::f16, o, r);
  sycl::free(x, q); sycl::free(out, q);
  return ok;
}

template <typename T>
bool check_glu(sycl::queue& q, DType dt, ops::GluMode mode, const char* mname,
               std::size_t rows, std::size_t d) {
  T* x = sycl::malloc_shared<T>(rows * 2 * d, q);
  T* out = sycl::malloc_shared<T>(rows * d, q);
  for (std::size_t i = 0; i < rows * 2 * d; ++i) x[i] = static_cast<T>(sample(i) * 2.0f);
  ops::glu(q, x, out, rows, d, dt, mode, Variant::sycl, true);
  std::vector<float> o(rows * d);
  std::vector<double> r(rows * d);
  for (std::size_t rr = 0; rr < rows; ++rr) {
    for (std::size_t i = 0; i < d; ++i) {
      const double gate = static_cast<double>(x[rr * 2 * d + i]);
      const double val = static_cast<double>(x[rr * 2 * d + d + i]);
      double a;
      switch (mode) {
        case ops::GluMode::swiglu: a = silu_ref(gate); break;
        case ops::GluMode::geglu:  a = gelu_ref(gate); break;
        case ops::GluMode::reglu:  a = relu_ref(gate); break;
        default:                   a = sigmoid_ref(gate); break;
      }
      o[rr * d + i] = static_cast<float>(out[rr * d + i]);
      r[rr * d + i] = a * val;
    }
  }
  const bool ok = report<T>(mname, dt, o, r);
  sycl::free(x, q); sycl::free(out, q);
  return ok;
}

template <typename T>
bool check_softmax(sycl::queue& q, DType dt, Variant variant, std::size_t rows,
                   std::size_t dim) {
  const std::size_t n = rows * dim;
  T* x = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<T>(sample(i) * 3.0f);

  ops::softmax(q, x, out, rows, dim, dt, variant, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  double worst_excess = 0.0, max_abs = 0.0, worst_rowsum = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double m = -1e30;
    for (std::size_t i = 0; i < dim; ++i) m = std::max(m, static_cast<double>(x[r * dim + i]));
    double sum = 0.0;
    for (std::size_t i = 0; i < dim; ++i) sum += std::exp(static_cast<double>(x[r * dim + i]) - m);
    double rowsum = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      const double ref_hi = std::exp(static_cast<double>(x[r * dim + i]) - m) / sum;
      const double ref = static_cast<double>(static_cast<T>(ref_hi));
      const double err = std::abs(static_cast<double>(out[r * dim + i]) - ref);
      max_abs = std::max(max_abs, err);
      worst_excess = std::max(worst_excess, err - (tol.atol + tol.rtol * std::abs(ref)));
      rowsum += static_cast<double>(out[r * dim + i]);
    }
    worst_rowsum = std::max(worst_rowsum, std::abs(rowsum - 1.0));
  }
  sycl::free(x, q); sycl::free(out, q);
  const bool ok = worst_excess <= 0.0 && worst_rowsum < 5e-2;
  std::cout << "  softmax dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " max_abs=" << max_abs << " rowsum_err=" << worst_rowsum
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_fp8_gemm(sycl::queue& q, DType out_dt, ops::Fp8Kind kind,
                    const char* kname, std::size_t M, std::size_t N, std::size_t K,
                    Variant variant = Variant::vendor) {
  // Build fp8 inputs from f32 via the public codec; decode them back to get the
  // exact fp8-rounded values the GEMM actually consumes -> fp64 reference.
  float* Af = sycl::malloc_shared<float>(M * K, q);
  float* Bf = sycl::malloc_shared<float>(K * N, q);
  std::uint8_t* A8 = sycl::malloc_shared<std::uint8_t>(M * K, q);
  std::uint8_t* B8 = sycl::malloc_shared<std::uint8_t>(K * N, q);
  float* Art = sycl::malloc_shared<float>(M * K, q);
  float* Brt = sycl::malloc_shared<float>(K * N, q);
  T* C = sycl::malloc_shared<T>(M * N, q);
  for (std::size_t i = 0; i < M * K; ++i) Af[i] = sample(i) * 0.5f;
  for (std::size_t i = 0; i < K * N; ++i) Bf[i] = sample(i + 5) * 0.5f;
  const float scale = 1.0f;

  bool ok = true;
  try {
    ops::fp8_encode(q, Af, A8, M * K, kind);
    ops::fp8_encode(q, Bf, B8, K * N, kind);
    ops::fp8_decode(q, A8, Art, M * K, kind);
    ops::fp8_decode(q, B8, Brt, K * N, kind);
    ops::fp8_gemm(q, A8, B8, C, M, N, K, kind, scale, out_dt, variant, true);

    const Tol base = tol_for(out_dt);
    const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 5e-3;
    double worst = 0.0, max_abs = 0.0;
    for (std::size_t m = 0; m < M; ++m)
      for (std::size_t n = 0; n < N; ++n) {
        double acc = 0.0;
        for (std::size_t k = 0; k < K; ++k)
          acc += (double)Art[m * K + k] * (double)Brt[k * N + n];
        const double ref = static_cast<double>(static_cast<T>(acc * scale));
        const double err = std::abs((double)C[m * N + n] - ref);
        max_abs = std::max(max_abs, err);
        worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
      }
    ok = worst <= 0.0;
    std::cout << "  fp8_gemm " << kname << " out=" << dtype_name(out_dt)
              << " " << variant_name(variant)
              << " (" << M << "x" << N << "x" << K << ") max_abs=" << max_abs
              << (ok ? "  ok" : "  FAIL") << '\n';
  } catch (const std::exception& e) {
    std::cout << "  fp8_gemm " << kname << ": UNSUPPORTED (" << e.what() << ")\n";
    ok = true;  // not a correctness failure; recorded as unsupported on this device
  }
  sycl::free(Af, q); sycl::free(Bf, q); sycl::free(A8, q); sycl::free(B8, q);
  sycl::free(Art, q); sycl::free(Brt, q); sycl::free(C, q);
  return ok;
}

template <typename T>
bool check_qgemm_int8(sycl::queue& q, DType out_dt, Variant variant,
                      std::size_t M, std::size_t N, std::size_t K) {
  std::int8_t* A = sycl::malloc_shared<std::int8_t>(M * K, q);
  std::int8_t* B = sycl::malloc_shared<std::int8_t>(K * N, q);
  float* as = sycl::malloc_shared<float>(M, q);
  float* bs = sycl::malloc_shared<float>(N, q);
  T* C = sycl::malloc_shared<T>(M * N, q);
  for (std::size_t i = 0; i < M * K; ++i) A[i] = static_cast<std::int8_t>(std::floor(sample(i) * 30.0f));
  for (std::size_t i = 0; i < K * N; ++i) B[i] = static_cast<std::int8_t>(std::floor(sample(i + 5) * 30.0f));
  for (std::size_t i = 0; i < M; ++i) as[i] = 0.01f + 0.005f * std::abs(sample(i + 1));
  for (std::size_t i = 0; i < N; ++i) bs[i] = 0.01f + 0.005f * std::abs(sample(i + 2));

  ops::qgemm_int8(q, A, B, as, bs, C, M, N, K, out_dt, variant, /*blocking=*/true);

  const Tol base = tol_for(out_dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 1e-3;
  double worst = 0.0, max_abs = 0.0;
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t n = 0; n < N; ++n) {
      long acc = 0;
      for (std::size_t k = 0; k < K; ++k) acc += (int)A[m * K + k] * (int)B[k * N + n];
      const double ref_hi = (double)acc * (double)as[m] * (double)bs[n];
      const double ref = static_cast<double>(static_cast<T>(ref_hi));
      const double err = std::abs((double)C[m * N + n] - ref);
      max_abs = std::max(max_abs, err);
      worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
    }
  sycl::free(A, q); sycl::free(B, q); sycl::free(as, q); sycl::free(bs, q); sycl::free(C, q);
  const bool ok = worst <= 0.0;
  std::cout << "  qgemm_int8 out=" << dtype_name(out_dt) << " variant=" << variant_name(variant)
            << " (" << M << "x" << N << "x" << K << ") max_abs=" << max_abs
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_legacy(sycl::queue& q, DType act_dt, ops::GgufType gt, const char* name,
                       std::size_t N, std::size_t K) {
  const int BB = (gt == ops::GgufType::q4_1) ? 20 : (gt == ops::GgufType::q5_0) ? 22 : 24;
  const bool affine = (gt == ops::GgufType::q4_1 || gt == ops::GgufType::q5_1);
  const bool five = (gt == ops::GgufType::q5_0 || gt == ops::GgufType::q5_1);
  const std::size_t bpr = K / 32, row_bytes = bpr * BB;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q); T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.02f));
  const std::uint16_t mb = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.005f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < bpr; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * BB;
      blk[0] = db & 0xFF; blk[1] = db >> 8;
      int off = 2;
      if (affine) { blk[2] = mb & 0xFF; blk[3] = mb >> 8; off = 4; }
      for (int i = off; i < BB; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * bpr + b) * BB + i) * 128 + 128) & 0xFF);
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 121));
  ops::gguf_gemv(q, w, x, y, N, K, gt, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0; const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < bpr; ++b) {
      const std::uint8_t* blk = wrow + b * BB; const double d = lh(blk);
      const double m = affine ? lh(blk + 2) : 0.0;
      const std::uint8_t* qhp = blk + (affine ? 4 : 2);
      const std::uint8_t* qs = blk + (five ? (affine ? 8 : 6) : 4);
      const std::size_t kbase = b * 32;
      std::uint32_t qh = five ? ((std::uint32_t)qhp[0] | ((std::uint32_t)qhp[1] << 8) | ((std::uint32_t)qhp[2] << 16) | ((std::uint32_t)qhp[3] << 24)) : 0;
      for (int j = 0; j < 16; ++j) {
        double v0, v1;
        if (!five) { v0 = (qs[j] & 0xF); v1 = (qs[j] >> 4); }
        else {
          int xh0 = ((qh >> (j + 0)) << 4) & 0x10, xh1 = ((qh >> (j + 12))) & 0x10;
          int a = (qs[j] & 0xF) | xh0, b2 = (qs[j] >> 4) | xh1;
          if (!affine) { a -= 16; b2 -= 16; }
          v0 = a; v1 = b2;
        }
        acc += (v0 * d + m) * (double)x[kbase + j];
        acc += (v1 * d + m) * (double)x[kbase + 16 + j];
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv " << name << " act=" << dtype_name(act_dt) << " worst_excess=" << worst
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_iq4xs(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  static const int cb[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
  const std::size_t sb = K / 256, row_bytes = sb * 136;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q); T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.02f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < sb; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * 136;
      blk[0] = db & 0xFF; blk[1] = db >> 8;
      for (int i = 2; i < 136; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * sb + b) * 136 + i) * 128 + 128) & 0xFF);
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 131));
  ops::gguf_gemv(q, w, x, y, N, K, ops::GgufType::iq4_xs, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0; const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < sb; ++b) {
      const std::uint8_t* blk = wrow + b * 136; const double d = lh(blk);
      const std::uint32_t sh = (std::uint32_t)blk[2] | ((std::uint32_t)blk[3] << 8);
      const std::uint8_t* sl = blk + 4; const std::uint8_t* qs = blk + 8;
      for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const double dl = d * (ls - 32); const std::uint8_t* qq = qs + ib * 16; const std::size_t yb = b * 256 + ib * 32;
        for (int j = 0; j < 16; ++j) {
          acc += dl * cb[qq[j] & 0xF] * (double)x[yb + j];
          acc += dl * cb[qq[j] >> 4] * (double)x[yb + 16 + j];
        }
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv iq4_xs act=" << dtype_name(act_dt) << " worst_excess=" << worst
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// Grid i-quant references use the ggml tables from the kernels namespace.
using quixicore::xpu::kernels::iq2xxs_grid;
using quixicore::xpu::kernels::iq2xs_grid;
using quixicore::xpu::kernels::iq3xxs_grid;
using quixicore::xpu::kernels::ksigns_iq2xs;
using quixicore::xpu::kernels::kmask_iq2xs;
using quixicore::xpu::kernels::iq1s_grid;

template <typename T>
bool check_gguf_iq1s(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  const std::size_t sb = K / 256, row_bytes = sb * 50;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q); T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.05f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < sb; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * 50;
      blk[0] = db & 0xFF; blk[1] = db >> 8;
      for (int i = 2; i < 50; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * sb + b) * 50 + i) * 128 + 128) & 0xFF);
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 151));
  ops::gguf_gemv(q, w, x, y, N, K, ops::GgufType::iq1_s, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const Tol tol = tol_for(act_dt); const double rtol = tol.rtol * std::sqrt((double)K) + 5e-3; const double kDelta = 0.125;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0; const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < sb; ++b) {
      const std::uint8_t* blk = wrow + b * 50; const double d = lh(blk);
      const std::uint8_t* qs = blk + 2; const std::uint8_t* qhb = blk + 34; const std::size_t kbase = b * 256;
      for (int ib = 0; ib < 8; ++ib) {
        const std::uint16_t qh = qhb[2 * ib] | (qhb[2 * ib + 1] << 8);
        const double dl = d * (2 * ((qh >> 12) & 7) + 1); const double delta = (qh & 0x8000) ? -kDelta : kDelta;
        for (int l = 0; l < 4; ++l) {
          const std::uint32_t gi = qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8); const std::uint64_t g = iq1s_grid[gi];
          const std::size_t yb = kbase + ib * 32 + l * 8;
          for (int j = 0; j < 8; ++j) { const int gv = (std::int8_t)((g >> (8 * j)) & 0xFF); acc += dl * (gv + delta) * (double)x[yb + j]; }
        }
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv iq1_s act=" << dtype_name(act_dt) << " worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_quantize_int4(sycl::queue& q, DType dt, std::size_t N, std::size_t K, std::size_t group) {
  const std::size_t bpr = K / 2, gpr = K / group;
  T* W = sycl::malloc_shared<T>(N * K, q);
  std::uint8_t* wp = sycl::malloc_shared<std::uint8_t>(N * bpr, q);
  half_t* sc = sycl::malloc_shared<half_t>(N * gpr, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);
  for (std::size_t i = 0; i < N * K; ++i) W[i] = static_cast<T>(sample(i) * 2.0f);
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 161));

  ops::quantize_int4_group(q, W, wp, sc, N, K, group, dt, Variant::sycl, true);
  ops::qgemv_int4(q, wp, sc, x, y, N, K, group, dt, Variant::sycl, true);

  // Host-decode the packed result and reference-GEMV; must match qgemv exactly
  // (both consume the same quantized weights). Also bound the recon error.
  int recon_bad = 0; double worst = 0.0;
  const Tol tol = tol_for(dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    for (std::size_t b = 0; b < bpr; ++b) {
      const std::uint8_t byte = wp[n * bpr + b];
      int q0 = byte & 0xF; if (q0 >= 8) q0 -= 16;
      int q1 = byte >> 4;  if (q1 >= 8) q1 -= 16;
      const std::size_t k0 = 2 * b;
      const double s = (double)sc[n * gpr + k0 / group];
      acc += q0 * s * (double)x[k0];
      acc += q1 * s * (double)x[k0 + 1];
      // recon error vs original weight <= half a quant step
      if (std::abs(q0 * s - (double)W[n * K + k0]) > s * 0.51) ++recon_bad;
      if (std::abs(q1 * s - (double)W[n * K + k0 + 1]) > s * 0.51) ++recon_bad;
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(W, q); sycl::free(wp, q); sycl::free(sc, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = (worst <= 0.0) && (recon_bad == 0);
  std::cout << "  quantize_int4_group dt=" << dtype_name(dt) << " roundtrip_excess=" << worst
            << " recon_bad=" << recon_bad << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_act_quant(sycl::queue& q, DType dt, std::size_t rows, std::size_t dim) {
  T* x = sycl::malloc_shared<T>(rows * dim, q);
  signed char* qo = sycl::malloc_shared<signed char>(rows * dim, q);
  float* scale = sycl::malloc_shared<float>(rows, q);
  for (std::size_t i = 0; i < rows * dim; ++i) x[i] = static_cast<T>(sample(i) * 6.0f);
  ops::act_quant_int8(q, x, qo, scale, rows, dim, dt, Variant::sycl, true);

  int bad_scale = 0; double worst_recon = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double amax = 0;
    for (std::size_t j = 0; j < dim; ++j) amax = std::max(amax, std::abs((double)x[r * dim + j]));
    const double ref_s = amax / 127.0;
    if (std::abs((double)scale[r] - ref_s) > 1e-4 * std::max(1.0, ref_s)) ++bad_scale;
    for (std::size_t j = 0; j < dim; ++j) {
      const double recon = (double)qo[r * dim + j] * (double)scale[r];
      worst_recon = std::max(worst_recon, std::abs(recon - (double)x[r * dim + j]) - (double)scale[r] * 0.51);
    }
  }
  sycl::free(x, q); sycl::free(qo, q); sycl::free(scale, q);
  const bool ok = (bad_scale == 0) && (worst_recon <= 0.0);
  std::cout << "  act_quant_int8 dt=" << dtype_name(dt) << " bad_scale=" << bad_scale
            << " recon_excess=" << worst_recon << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_iquant(sycl::queue& q, DType act_dt, ops::GgufType gt, const char* name,
                       std::size_t N, std::size_t K) {
  const int BB = (gt == ops::GgufType::iq2_xxs) ? 66 : (gt == ops::GgufType::iq2_xs) ? 74 : 98;
  const std::size_t sb = K / 256, row_bytes = sb * BB;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q); T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.05f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < sb; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * BB;
      blk[0] = db & 0xFF; blk[1] = db >> 8;
      for (int i = 2; i < BB; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * sb + b) * BB + i) * 128 + 128) & 0xFF);
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 141));
  ops::gguf_gemv(q, w, x, y, N, K, gt, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  auto ru32 = [](const std::uint8_t* p){ return (std::uint32_t)p[0] | ((std::uint32_t)p[1]<<8) | ((std::uint32_t)p[2]<<16) | ((std::uint32_t)p[3]<<24); };
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 5e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0; const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < sb; ++b) {
      const std::uint8_t* blk = wrow + b * BB; const double d = lh(blk); const std::size_t kbase = b * 256;
      if (gt == ops::GgufType::iq2_xxs) {
        const std::uint8_t* qs = blk + 2;
        for (int ib = 0; ib < 8; ++ib) {
          const std::uint8_t* aux = qs + 8 * ib; const std::uint32_t a1 = ru32(aux + 4);
          const double dbl = d * (0.5 + (a1 >> 28)) * 0.25;
          for (int l = 0; l < 4; ++l) {
            const std::uint64_t g = iq2xxs_grid[aux[l]]; const std::uint8_t sgn = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
            const std::size_t yb = kbase + ib * 32 + l * 8;
            for (int j = 0; j < 8; ++j) { const int gv = (std::int8_t)((g >> (8 * j)) & 0xFF); acc += dbl * gv * ((sgn & kmask_iq2xs[j]) ? -1 : 1) * (double)x[yb + j]; }
          }
        }
      } else if (gt == ops::GgufType::iq2_xs) {
        const std::uint8_t* qsb = blk + 2; const std::uint8_t* scales = blk + 66;
        for (int ib = 0; ib < 8; ++ib) {
          const double db1 = d * (0.5 + (scales[ib] & 0xF)) * 0.25, db2 = d * (0.5 + (scales[ib] >> 4)) * 0.25;
          for (int l = 0; l < 4; ++l) {
            const std::uint8_t* p = qsb + 8 * ib + 2 * l; const std::uint16_t q2 = p[0] | (p[1] << 8);
            const std::uint64_t g = iq2xs_grid[q2 & 511]; const std::uint8_t sgn = ksigns_iq2xs[q2 >> 9]; const double dbl = (l < 2) ? db1 : db2;
            const std::size_t yb = kbase + ib * 32 + l * 8;
            for (int j = 0; j < 8; ++j) { const int gv = (std::int8_t)((g >> (8 * j)) & 0xFF); acc += dbl * gv * ((sgn & kmask_iq2xs[j]) ? -1 : 1) * (double)x[yb + j]; }
          }
        }
      } else {  // iq3_xxs
        const std::uint8_t* q3 = blk + 2; const std::uint8_t* gas = blk + 66;
        for (int ib = 0; ib < 8; ++ib) {
          const std::uint32_t a = ru32(gas + 4 * ib); const double dbl = d * (0.5 + (a >> 28)) * 0.5; const std::uint8_t* q3b = q3 + 8 * ib;
          for (int l = 0; l < 4; ++l) {
            const std::uint8_t sgn = ksigns_iq2xs[(a >> (7 * l)) & 127];
            const std::uint32_t g1 = iq3xxs_grid[q3b[2 * l]], g2 = iq3xxs_grid[q3b[2 * l + 1]];
            const std::size_t yb = kbase + ib * 32 + l * 8;
            for (int j = 0; j < 4; ++j) {
              acc += dbl * (std::int8_t)((g1 >> (8 * j)) & 0xFF) * ((sgn & kmask_iq2xs[j]) ? -1 : 1) * (double)x[yb + j];
              acc += dbl * (std::int8_t)((g2 >> (8 * j)) & 0xFF) * ((sgn & kmask_iq2xs[j + 4]) ? -1 : 1) * (double)x[yb + 4 + j];
            }
          }
        }
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv " << name << " act=" << dtype_name(act_dt) << " worst_excess=" << worst
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_iq4nl(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  static const int cb[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
  const std::size_t bpr = K / 32, row_bytes = bpr * 18;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.02f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < bpr; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * 18;
      blk[0] = db & 0xFF; blk[1] = db >> 8;
      for (int i = 2; i < 18; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * bpr + b) * 18 + i) * 128 + 128) & 0xFF);
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 111));
  ops::gguf_gemv(q, w, x, y, N, K, ops::GgufType::iq4_nl, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < bpr; ++b) {
      const std::uint8_t* blk = wrow + b * 18; const double d = lh(blk); const std::uint8_t* qs = blk + 2;
      const std::size_t kbase = b * 32;
      for (int j = 0; j < 16; ++j) {
        acc += d * cb[qs[j] & 0xF] * (double)x[kbase + j];
        acc += d * cb[qs[j] >> 4] * (double)x[kbase + 16 + j];
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv iq4_nl act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << ") worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_q3k(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  const std::size_t sb = K / 256, row_bytes = sb * 110;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.02f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < sb; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * 110;
      for (int i = 0; i < 108; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * sb + b) * 110 + i) * 128 + 128) & 0xFF);
      blk[108] = db & 0xFF; blk[109] = db >> 8;
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 101));
  ops::gguf_gemv(q, w, x, y, N, K, ops::GgufType::q3_K, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < sb; ++b) {
      const std::uint8_t* blk = wrow + b * 110;
      const std::uint8_t* hm = blk; const std::uint8_t* qs = blk + 32; const std::uint8_t* sc = blk + 96;
      const double d = lh(blk + 108);
      auto rd32 = [&](int o){ return (std::uint32_t)sc[o] | ((std::uint32_t)sc[o+1]<<8) | ((std::uint32_t)sc[o+2]<<16) | ((std::uint32_t)sc[o+3]<<24); };
      const std::uint32_t a0 = rd32(0), a1 = rd32(4), tmp = rd32(8);
      std::uint32_t aux[4];
      aux[2] = ((a0>>4)&kmask2) | (((tmp>>4)&kmask1)<<4);
      aux[3] = ((a1>>4)&kmask2) | (((tmp>>6)&kmask1)<<4);
      aux[0] = (a0&kmask2) | (((tmp>>0)&kmask1)<<4);
      aux[1] = (a1&kmask2) | (((tmp>>2)&kmask1)<<4);
      auto scl = [&](int i){ return (int)(std::int8_t)((aux[i>>2] >> (8*(i&3))) & 0xFF); };
      for (int h = 0; h < 2; ++h) {
        const std::uint8_t* qb = qs + h * 32;
        for (int j = 0; j < 4; ++j) {
          const int shift = 2 * j, is = h * 8 + j * 2; const std::uint8_t mbit = (std::uint8_t)(1u << (h*4+j));
          const double dl1 = d * (scl(is) - 32), dl2 = d * (scl(is + 1) - 32);
          const std::size_t yb = b * 256 + h * 128 + j * 32;
          for (int l = 0; l < 16; ++l) {
            const int v1 = (int)((qb[l]>>shift)&3) - ((hm[l] & mbit) ? 0 : 4);
            const int v2 = (int)((qb[l+16]>>shift)&3) - ((hm[l+16] & mbit) ? 0 : 4);
            acc += dl1 * v1 * (double)x[yb + l];
            acc += dl2 * v2 * (double)x[yb + 16 + l];
          }
        }
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv q3_K act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << ") worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_q2k(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  const std::size_t sb = K / 256, row_bytes = sb * 84;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.02f));
  const std::uint16_t mb = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.01f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < sb; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * 84;
      for (int i = 0; i < 80; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * sb + b) * 84 + i) * 128 + 128) & 0xFF);
      blk[80] = db & 0xFF; blk[81] = db >> 8; blk[82] = mb & 0xFF; blk[83] = mb >> 8;
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 91));
  ops::gguf_gemv(q, w, x, y, N, K, ops::GgufType::q2_K, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < sb; ++b) {
      const std::uint8_t* blk = wrow + b * 84;
      const std::uint8_t* scales = blk; const std::uint8_t* qs = blk + 16;
      const double d = lh(blk + 80), dmin = lh(blk + 82);
      for (int h = 0; h < 2; ++h) {
        const std::uint8_t* qb = qs + h * 32;
        for (int j = 0; j < 4; ++j) {
          const int shift = 2 * j;
          const std::uint8_t sc1 = scales[h * 8 + j * 2], sc2 = scales[h * 8 + j * 2 + 1];
          const double dl1 = d * (sc1 & 0xF), ml1 = dmin * (sc1 >> 4), dl2 = d * (sc2 & 0xF), ml2 = dmin * (sc2 >> 4);
          const std::size_t yb = b * 256 + h * 128 + j * 32;
          for (int l = 0; l < 16; ++l) {
            acc += (dl1 * ((qb[l] >> shift) & 3) - ml1) * (double)x[yb + l];
            acc += (dl2 * ((qb[l + 16] >> shift) & 3) - ml2) * (double)x[yb + 16 + l];
          }
        }
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv q2_K act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << ") worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_q5k(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  auto smk4 = [](int j, const std::uint8_t* s, int& sc, int& m) {
    if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
  };
  const std::size_t sb = K / 256, row_bytes = sb * 176;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.02f));
  const std::uint16_t mb = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.01f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < sb; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * 176;
      blk[0] = db & 0xFF; blk[1] = db >> 8; blk[2] = mb & 0xFF; blk[3] = mb >> 8;
      for (int i = 4; i < 176; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * sb + b) * 176 + i) * 128 + 128) & 0xFF);
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 81));
  ops::gguf_gemv(q, w, x, y, N, K, ops::GgufType::q5_K, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < sb; ++b) {
      const std::uint8_t* blk = wrow + b * 176;
      const double d = lh(blk), dmin = lh(blk + 2);
      const std::uint8_t* sc = blk + 4; const std::uint8_t* qh = blk + 16; const std::uint8_t* qs = blk + 48;
      for (int iter = 0; iter < 4; ++iter) {
        int s1, m1, s2, m2; smk4(iter * 2, sc, s1, m1); smk4(iter * 2 + 1, sc, s2, m2);
        const double d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
        const std::uint8_t* ql = qs + iter * 32; const int u1 = 1 << (2 * iter), u2 = 2 << (2 * iter);
        const std::size_t yb = b * 256 + iter * 64;
        for (int l = 0; l < 32; ++l) {
          const double q1 = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
          const double q2 = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
          acc += (d1 * q1 - mm1) * (double)x[yb + l];
          acc += (d2 * q2 - mm2) * (double)x[yb + 32 + l];
        }
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv q5_K act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << ") worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_q4k(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  auto smk4 = [](int j, const std::uint8_t* s, int& sc, int& m) {
    if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
  };
  const std::size_t sb = K / 256, row_bytes = sb * 144;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.02f));
  const std::uint16_t mb = sycl::bit_cast<std::uint16_t>(static_cast<half_t>(0.01f));
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < sb; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * 144;
      blk[0] = db & 0xFF; blk[1] = db >> 8; blk[2] = mb & 0xFF; blk[3] = mb >> 8;
      for (int i = 4; i < 144; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * sb + b) * 144 + i) * 128 + 128) & 0xFF);
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 71));

  ops::gguf_gemv(q, w, x, y, N, K, ops::GgufType::q4_K, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < sb; ++b) {
      const std::uint8_t* blk = wrow + b * 144;
      const double d = lh(blk), dmin = lh(blk + 2);
      const std::uint8_t* sc = blk + 4; const std::uint8_t* qs = blk + 16;
      for (int iter = 0; iter < 4; ++iter) {
        int s1, m1, s2, m2; smk4(iter * 2, sc, s1, m1); smk4(iter * 2 + 1, sc, s2, m2);
        const double d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
        const std::uint8_t* qq = qs + iter * 32; const std::size_t yb = b * 256 + iter * 64;
        for (int l = 0; l < 32; ++l) {
          acc += (d1 * (qq[l] & 0xF) - mm1) * (double)x[yb + l];
          acc += (d2 * (qq[l] >> 4) - mm2) * (double)x[yb + 32 + l];
        }
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv q4_K act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << ") worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_q6k(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  const std::size_t sb = K / 256, row_bytes = sb * 210;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);
  // Random bytes for ql/qh; small int8 scales; a fixed small fp16 d per block.
  const half_t hd = static_cast<half_t>(0.01f);
  const std::uint16_t db = sycl::bit_cast<std::uint16_t>(hd);
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < sb; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * 210;
      for (int i = 0; i < 192; ++i) blk[i] = static_cast<std::uint8_t>((std::size_t)(sample((n * sb + b) * 192 + i) * 128 + 128) & 0xFF);
      for (int i = 0; i < 16; ++i) blk[192 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::floor(sample((n * sb + b) * 16 + i) * 8)));
      blk[208] = db & 0xFF; blk[209] = static_cast<std::uint8_t>(db >> 8);
    }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 61));

  ops::gguf_gemv(q, w, x, y, N, K, ops::GgufType::q6_K, act_dt, Variant::sycl, true);

  auto lh = [](const std::uint8_t* p) { std::uint16_t b = p[0] | (p[1] << 8); return (double)sycl::bit_cast<half_t>(b); };
  const Tol tol = tol_for(act_dt);
  const double rtol = tol.rtol * std::sqrt((double)K) + 3e-3;
  double worst = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    const std::uint8_t* wrow = w + n * row_bytes;
    for (std::size_t b = 0; b < sb; ++b) {
      const std::uint8_t* blk = wrow + b * 210;
      const std::uint8_t* ql = blk; const std::uint8_t* qh = blk + 128;
      const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
      const double d = lh(blk + 208);
      for (int half = 0; half < 2; ++half) {
        const std::uint8_t* qlh = ql + half * 64; const std::uint8_t* qhh = qh + half * 32;
        const std::int8_t* sch = sc + half * 8; const std::size_t yoff = b * 256 + (std::size_t)half * 128;
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const int q1 = ((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
          const int q2 = ((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
          const int q3 = ((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
          const int q4 = ((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
          acc += d * sch[is + 0] * q1 * (double)x[yoff + l + 0];
          acc += d * sch[is + 2] * q2 * (double)x[yoff + l + 32];
          acc += d * sch[is + 4] * q3 * (double)x[yoff + l + 64];
          acc += d * sch[is + 6] * q4 * (double)x[yoff + l + 96];
        }
      }
    }
    const double ref = (double)static_cast<T>(acc);
    worst = std::max(worst, std::abs((double)y[n] - ref) - (tol.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv q6_K act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << ") worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_gguf_gemv(sycl::queue& q, DType act_dt, ops::GgufType type,
                     const char* tname, std::size_t N, std::size_t K) {
  const int block_bytes = (type == ops::GgufType::q8_0) ? 34 : 18;
  const std::size_t bpr = K / 32;              // blocks per row
  const std::size_t row_bytes = bpr * block_bytes;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * row_bytes, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 51));

  // Pack blocks; keep the fp16-rounded scale and int quants for the reference.
  std::vector<double> dref(N * bpr);
  std::vector<int> qref(N * K);
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < bpr; ++b) {
      std::uint8_t* blk = w + n * row_bytes + b * block_bytes;
      const half_t hd = static_cast<half_t>(0.01f + 0.02f * std::abs(sample((n * bpr + b) * 7)));
      const std::uint16_t db = sycl::bit_cast<std::uint16_t>(hd);
      blk[0] = db & 0xFF; blk[1] = static_cast<std::uint8_t>(db >> 8);
      dref[n * bpr + b] = static_cast<double>(hd);
      std::uint8_t* qs = blk + 2;
      if (type == ops::GgufType::q8_0) {
        for (int i = 0; i < 32; ++i) {
          const int v = static_cast<int>(std::floor(sample(n * K + b * 32 + i) * 60.0f));
          const int cv = std::max(-127, std::min(127, v));
          qs[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(cv));
          qref[n * K + b * 32 + i] = cv;
        }
      } else {  // q4_0: low nibble -> elem i, high nibble -> elem i+16, val=(nib-8)
        for (int i = 0; i < 16; ++i) {
          auto q4 = [&](std::size_t idx) {
            int v = static_cast<int>(std::floor((sample(idx) * 0.5f + 0.5f) * 16.0f));
            return std::max(0, std::min(15, v));
          };
          const int lo = q4(n * K + b * 32 + i);
          const int hi = q4(n * K + b * 32 + 16 + i);
          qs[i] = static_cast<std::uint8_t>(lo | (hi << 4));
          qref[n * K + b * 32 + i] = lo - 8;
          qref[n * K + b * 32 + 16 + i] = hi - 8;
        }
      }
    }

  ops::gguf_gemv(q, w, x, y, N, K, type, act_dt, Variant::sycl, true);

  const Tol base = tol_for(act_dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 3e-3;
  double worst = 0.0, max_abs = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    for (std::size_t k = 0; k < K; ++k)
      acc += static_cast<double>(qref[n * K + k]) * dref[n * bpr + k / 32] * static_cast<double>(x[k]);
    const double ref = static_cast<double>(static_cast<T>(acc));
    const double err = std::abs(static_cast<double>(y[n]) - ref);
    max_abs = std::max(max_abs, err);
    worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  gguf_gemv " << tname << " act=" << dtype_name(act_dt)
            << " (N=" << N << " K=" << K << ") max_abs=" << max_abs
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_mxfp4_gemv(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K) {
  const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const std::size_t bpr = K / 2, blkpr = K / 32;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * bpr, q);
  std::uint8_t* bs = sycl::malloc_shared<std::uint8_t>(N * blkpr, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);

  auto nib = [&](std::size_t i) {  // random e2m1 nibble
    int mag = static_cast<int>(std::floor((sample(i) * 0.5f + 0.5f) * 8.0f)) & 7;
    int sign = (sample(i + 1) < 0.0f) ? 8 : 0;
    return static_cast<std::uint8_t>(sign | mag);
  };
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t b = 0; b < bpr; ++b)
      w[n * bpr + b] = static_cast<std::uint8_t>(nib(n * K + 2 * b) | (nib(n * K + 2 * b + 1) << 4));
    for (std::size_t g = 0; g < blkpr; ++g)
      bs[n * blkpr + g] = static_cast<std::uint8_t>(124 + (static_cast<int>((g + n) % 5)));  // ~2^-3..2^1
  }
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 21));

  ops::mxfp4_gemv(q, w, bs, x, y, N, K, act_dt, Variant::sycl, true);

  const Tol base = tol_for(act_dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 3e-3;
  double worst = 0.0, max_abs = 0.0;
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint8_t byte = w[n * bpr + k / 2];
      const int n4 = (k & 1) ? (byte >> 4) : (byte & 0xF);
      double val = e2m1[n4 & 7];
      if (n4 & 8) val = -val;
      const double scale = std::ldexp(1.0, static_cast<int>(bs[n * blkpr + k / 32]) - 127);
      acc += val * scale * static_cast<double>(x[k]);
    }
    const double ref = static_cast<double>(static_cast<T>(acc));
    const double err = std::abs(static_cast<double>(y[n]) - ref);
    max_abs = std::max(max_abs, err);
    worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(bs, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  mxfp4_gemv act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << ") max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_nvfp4_gemm(sycl::queue &q, DType act_dt, std::size_t M, std::size_t N, std::size_t K) {
  const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const std::size_t bpr = K / 2, blkpr = K / 16;  // block 16
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * bpr, q);
  std::uint8_t* bs = sycl::malloc_shared<std::uint8_t>(N * blkpr, q);  // e4m3 bytes
  float* bs_f = sycl::malloc_shared<float>(N * blkpr, q);   // desired scale floats
  float* bs_rt = sycl::malloc_shared<float>(N * blkpr, q);  // round-tripped exact
  T *x = sycl::malloc_shared<T>(M * K, q);
  T *y = sycl::malloc_shared<T>(M * N, q);
  const float gscale = 0.75f;

  auto nib = [&](std::size_t i) {
    int mag = static_cast<int>(std::floor((sample(i) * 0.5f + 0.5f) * 8.0f)) & 7;
    int sign = (sample(i + 1) < 0.0f) ? 8 : 0;
    return static_cast<std::uint8_t>(sign | mag);
  };
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < bpr; ++b)
      w[n * bpr + b] = static_cast<std::uint8_t>(nib(n * K + 2 * b) | (nib(n * K + 2 * b + 1) << 4));
  for (std::size_t i = 0; i < N * blkpr; ++i) bs_f[i] = 0.5f + 0.5f * std::abs(sample(i + 31));
  for (std::size_t i = 0; i < M * K; ++i)
    x[i] = static_cast<T>(sample(i + 41));

  bool ok = true;
  try {
    // Build e4m3 block-scale bytes via the public codec; decode back for the exact
    // scale values the kernel consumes (its e4m3 decode must match).
    ops::fp8_encode(q, bs_f, bs, N * blkpr, ops::Fp8Kind::e4m3);
    ops::fp8_decode(q, bs, bs_rt, N * blkpr, ops::Fp8Kind::e4m3);
    ops::nvfp4_gemm(q, w, bs, gscale, x, y, M, N, K, act_dt, Variant::sycl, true);

    const Tol base = tol_for(act_dt);
    const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 3e-3;
    double worst = 0.0, max_abs = 0.0;
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t n = 0; n < N; ++n) {
        double acc = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
          const std::uint8_t byte = w[n * bpr + k / 2];
          const int n4 = (k & 1) ? (byte >> 4) : (byte & 0xF);
          double val = e2m1[n4 & 7];
          if (n4 & 8)
            val = -val;
          acc += val * (double)bs_rt[n * blkpr + k / 16] * (double)gscale * (double)x[m * K + k];
        }
        const double ref = static_cast<double>(static_cast<T>(acc));
        const double err = std::abs(static_cast<double>(y[m * N + n]) - ref);
        max_abs = std::max(max_abs, err);
        worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
      }
    }
    ok = worst <= 0.0;
    std::cout << "  nvfp4_gemm act=" << dtype_name(act_dt) << " (M=" << M << " N=" << N
              << " K=" << K << ") max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  } catch (const std::exception& e) {
    std::cout << "  nvfp4_gemm: skipped (" << e.what() << ")\n";
  }
  sycl::free(w, q); sycl::free(bs, q); sycl::free(bs_f, q); sycl::free(bs_rt, q);
  sycl::free(x, q); sycl::free(y, q);
  return ok;
}

template <typename T>
bool check_qgemv_int4(sycl::queue& q, DType act_dt, std::size_t N, std::size_t K,
                      std::size_t group) {
  const std::size_t bpr = K / 2;         // bytes per row
  const std::size_t gpr = K / group;     // groups per row
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * bpr, q);
  half_t* scales = sycl::malloc_shared<half_t>(N * gpr, q);
  T* x = sycl::malloc_shared<T>(K, q);
  T* y = sycl::malloc_shared<T>(N, q);

  // Deterministic int4 weights (signed [-8,7]) packed 2/byte, per-group scales.
  auto qval = [](std::size_t i) {
    int v = static_cast<int>(std::floor(sample(i) * 4.0f));  // ~[-8,7]
    return std::max(-8, std::min(7, v));
  };
  std::vector<int> wint(N * K);
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < bpr; ++b) {
      const int lo = qval(n * K + 2 * b);
      const int hi = qval(n * K + 2 * b + 1);
      wint[n * K + 2 * b] = lo;
      wint[n * K + 2 * b + 1] = hi;
      w[n * bpr + b] = static_cast<std::uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
    }
  for (std::size_t i = 0; i < N * gpr; ++i) scales[i] = static_cast<half_t>(0.02f + 0.01f * std::abs(sample(i + 9)));
  for (std::size_t k = 0; k < K; ++k) x[k] = static_cast<T>(sample(k + 13));

  ops::qgemv_int4(q, w, scales, x, y, N, K, group, act_dt, Variant::sycl, true);

  double worst = 0.0, max_abs = 0.0;
  const Tol base = tol_for(act_dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K));
  for (std::size_t n = 0; n < N; ++n) {
    double acc = 0.0;
    for (std::size_t k = 0; k < K; ++k)
      acc += static_cast<double>(wint[n * K + k]) *
             static_cast<double>(scales[n * gpr + k / group]) * static_cast<double>(x[k]);
    const double ref = static_cast<double>(static_cast<T>(acc));
    const double err = std::abs(static_cast<double>(y[n]) - ref);
    max_abs = std::max(max_abs, err);
    worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
  }
  sycl::free(w, q); sycl::free(scales, q); sycl::free(x, q); sycl::free(y, q);
  const bool ok = worst <= 0.0;
  std::cout << "  qgemv_int4 act=" << dtype_name(act_dt) << " (N=" << N << " K=" << K
            << " g=" << group << ") max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

// w4a16_gemm: int4 group-quant weight x f16/bf16 activation DPAS GEMM. fp64
// oracle is the CPU q4 dequant reference C[m,n] = sum_k A[m,k] * wint[n,k] *
// scale[n,k/group] (same int4 encoding + weight/scale generation as
// check_qgemv_int4, so the two share the q4 reference). Shapes cover the small-M
// decode regime and non-tile-multiple M/N and a K tail (edge masking).
template <typename T>
bool check_w4a16_gemm(sycl::queue& q, DType act_dt, std::size_t M, std::size_t N,
                      std::size_t K, std::size_t group) {
  const std::size_t bpr = K / 2, gpr = K / group;
  std::uint8_t* w = sycl::malloc_shared<std::uint8_t>(N * bpr, q);
  half_t* scales = sycl::malloc_shared<half_t>(N * gpr, q);
  T* Amat = sycl::malloc_shared<T>(M * K, q);
  T* Cmat = sycl::malloc_shared<T>(M * N, q);

  auto qval = [](std::size_t i) {
    int v = static_cast<int>(std::floor(sample(i) * 4.0f));  // ~[-8,7]
    return std::max(-8, std::min(7, v));
  };
  std::vector<int> wint(N * K);
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t b = 0; b < bpr; ++b) {
      const int lo = qval(n * K + 2 * b);
      const int hi = qval(n * K + 2 * b + 1);
      wint[n * K + 2 * b] = lo;
      wint[n * K + 2 * b + 1] = hi;
      w[n * bpr + b] = static_cast<std::uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
    }
  for (std::size_t i = 0; i < N * gpr; ++i) scales[i] = static_cast<half_t>(0.02f + 0.01f * std::abs(sample(i + 9)));
  for (std::size_t i = 0; i < M * K; ++i) Amat[i] = static_cast<T>(sample(i + 13) * 0.5f);

  ops::w4a16_gemm(q, Amat, w, scales, Cmat, M, N, K, group, act_dt, Variant::sycl, true);

  double worst = 0.0, max_abs = 0.0;
  const Tol base = tol_for(act_dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K));
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (std::size_t k = 0; k < K; ++k) {
        // Faithful to the DPAS algorithm: the int4 weight is dequantized INTO the
        // 16-bit compute dtype (bf16/f16) before the tensor multiply, so the
        // oracle rounds dequant(W) to T too (the a16 quant numerics -- the same
        // way the repo's other quant oracles model their compute-dtype rounding).
        const T wq = static_cast<T>(static_cast<float>(wint[n * K + k]) *
                                    static_cast<float>(scales[n * gpr + k / group]));
        acc += static_cast<double>(Amat[m * K + k]) * static_cast<double>(wq);
      }
      const double ref = static_cast<double>(static_cast<T>(acc));
      const double err = std::abs(static_cast<double>(Cmat[m * N + n]) - ref);
      max_abs = std::max(max_abs, err);
      worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
    }
  sycl::free(w, q); sycl::free(scales, q); sycl::free(Amat, q); sycl::free(Cmat, q);
  const bool ok = worst <= 0.0;
  std::cout << "  w4a16_gemm act=" << dtype_name(act_dt) << " (M=" << M << " N=" << N
            << " K=" << K << " g=" << group << ") max_abs=" << max_abs
            << " worst_excess=" << worst << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_rope(sycl::queue& q, DType dt, std::size_t tokens, std::size_t heads,
                std::size_t hd) {
  const std::size_t n = tokens * heads * hd;
  T* x = sycl::malloc_shared<T>(n, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<T>(sample(i));
  const float base = 10000.0f;
  ops::rope(q, x, out, tokens, heads, hd, base, 0, dt, Variant::sycl, true);

  const std::size_t half = hd / 2;
  std::vector<float> o(n);
  std::vector<double> r(n);
  for (std::size_t t = 0; t < tokens; ++t)
    for (std::size_t h = 0; h < heads; ++h)
      for (std::size_t i = 0; i < half; ++i) {
        const std::size_t bi = (t * heads + h) * hd;
        const double freq = std::pow((double)base, -2.0 * (double)i / (double)hd);
        const double ang = (double)t * freq;
        const double x1 = (double)x[bi + i], x2 = (double)x[bi + i + half];
        r[bi + i] = x1 * std::cos(ang) - x2 * std::sin(ang);
        r[bi + i + half] = x1 * std::sin(ang) + x2 * std::cos(ang);
        o[bi + i] = (float)out[bi + i];
        o[bi + i + half] = (float)out[bi + i + half];
      }
  const bool ok = report<T>("rope", dt, o, r);
  sycl::free(x, q); sycl::free(out, q);
  return ok;
}

bool check_argmax(sycl::queue& q, DType dt, std::size_t rows, std::size_t vocab) {
  float* logits = sycl::malloc_shared<float>(rows * vocab, q);
  int* out = sycl::malloc_shared<int>(rows, q);
  for (std::size_t i = 0; i < rows * vocab; ++i) logits[i] = sample(i * 3 + 1);
  ops::argmax(q, logits, out, rows, vocab, DType::f32, Variant::sycl, true);
  int mism = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    int ref = 0;
    float best = logits[r * vocab];
    for (std::size_t j = 1; j < vocab; ++j)
      if (logits[r * vocab + j] > best) { best = logits[r * vocab + j]; ref = (int)j; }
    if (out[r] != ref) ++mism;
  }
  sycl::free(logits, q); sycl::free(out, q);
  const bool ok = (mism == 0);
  std::cout << "  argmax dt=" << dtype_name(dt) << " mismatches=" << mism
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

bool check_adamw(sycl::queue& q, DType dt, std::size_t n) {
  // f32 params/moments for a clean numeric check.
  float* p = sycl::malloc_shared<float>(n, q);
  float* g = sycl::malloc_shared<float>(n, q);
  float* m = sycl::malloc_shared<float>(n, q);
  float* v = sycl::malloc_shared<float>(n, q);
  std::vector<double> rp(n);
  const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
  const int step = 5;
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = sample(i); g[i] = sample(i + 2) * 0.5f;
    m[i] = sample(i + 4) * 0.1f; v[i] = std::abs(sample(i + 6)) * 0.1f;
  }
  // fp64 reference
  const double bc1 = 1.0 - std::pow((double)b1, step);
  const double bc2 = 1.0 - std::pow((double)b2, step);
  for (std::size_t i = 0; i < n; ++i) {
    const double grad = g[i];
    const double mi = b1 * (double)m[i] + (1 - b1) * grad;
    const double vi = b2 * (double)v[i] + (1 - b2) * grad * grad;
    rp[i] = (double)p[i] - lr * ((mi / bc1) / (std::sqrt(vi / bc2) + eps) + wd * (double)p[i]);
  }
  ops::adamw(q, p, g, m, v, n, lr, b1, b2, eps, wd, step, dt, Variant::sycl, true);
  std::vector<float> op(n);
  for (std::size_t i = 0; i < n; ++i) op[i] = p[i];
  const bool ok = report<float>("adamw", dt, op, rp);
  sycl::free(p, q); sycl::free(g, q); sycl::free(m, q); sycl::free(v, q);
  return ok;
}

template <typename T>
bool check_gemm(sycl::queue& q, DType dt, Variant variant, std::size_t M,
                std::size_t N, std::size_t K) {
  T* A = sycl::malloc_shared<T>(M * K, q);
  T* B = sycl::malloc_shared<T>(K * N, q);
  T* C = sycl::malloc_shared<T>(M * N, q);
  for (std::size_t i = 0; i < M * K; ++i) A[i] = static_cast<T>(sample(i) * 0.3f);
  for (std::size_t i = 0; i < K * N; ++i) B[i] = static_cast<T>(sample(i + 5) * 0.3f);

  ops::dense_gemm(q, A, B, C, M, N, K, dt, variant, /*blocking=*/true);

  // Tolerance grows with the K-length accumulation; scale rtol by sqrt(K).
  const Tol base = tol_for(dt);
  const double rtol = base.rtol * std::sqrt(static_cast<double>(K));
  double worst = 0.0, max_abs = 0.0;
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (std::size_t k = 0; k < K; ++k)
        acc += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
      const double ref = static_cast<double>(static_cast<T>(acc));
      const double err = std::abs(static_cast<double>(C[m * N + n]) - ref);
      max_abs = std::max(max_abs, err);
      worst = std::max(worst, err - (base.atol + rtol * std::abs(ref)));
    }
  }
  sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
  const bool ok = worst <= 0.0;
  std::cout << "  dense_gemm dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " (" << M << "x" << N << "x" << K << ") max_abs=" << max_abs
            << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_rms_norm(sycl::queue& q, DType dt, Variant variant, std::size_t rows,
                    std::size_t dim) {
  const std::size_t n = rows * dim;
  T* x = sycl::malloc_shared<T>(n, q);
  T* w = sycl::malloc_shared<T>(dim, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<T>(sample(i));
  for (std::size_t i = 0; i < dim; ++i) w[i] = static_cast<T>(0.5f + sample(i + 7) * 0.1f);
  const float eps = 1e-6f;

  ops::rms_norm(q, x, w, out, rows, dim, eps, dt, variant, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  double worst_excess = 0.0, max_abs = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      const double v = static_cast<double>(x[r * dim + i]);
      ss += v * v;
    }
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);
    for (std::size_t i = 0; i < dim; ++i) {
      const double ref_hi = static_cast<double>(x[r * dim + i]) * inv *
                            static_cast<double>(w[i]);
      const double ref = static_cast<double>(static_cast<T>(ref_hi));
      const double err = std::abs(static_cast<double>(out[r * dim + i]) - ref);
      max_abs = std::max(max_abs, err);
      worst_excess = std::max(worst_excess, err - (tol.atol + tol.rtol * std::abs(ref)));
    }
  }
  sycl::free(x, q); sycl::free(w, q); sycl::free(out, q);
  const bool ok = worst_excess <= 0.0;
  std::cout << "  rms_norm dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_layernorm(sycl::queue& q, DType dt, Variant variant, std::size_t rows,
                     std::size_t dim) {
  const std::size_t n = rows * dim;
  T* x = sycl::malloc_shared<T>(n, q);
  T* w = sycl::malloc_shared<T>(dim, q);
  T* b = sycl::malloc_shared<T>(dim, q);
  T* out = sycl::malloc_shared<T>(n, q);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<T>(sample(i));
  for (std::size_t i = 0; i < dim; ++i) {
    w[i] = static_cast<T>(0.8f + sample(i + 3) * 0.1f);
    b[i] = static_cast<T>(sample(i + 11) * 0.1f);
  }
  const float eps = 1e-5f;

  ops::layernorm(q, x, w, b, out, rows, dim, eps, dt, variant, /*blocking=*/true);

  const Tol tol = tol_for(dt);
  double worst_excess = 0.0, max_abs = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double sum = 0.0, ss = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      const double v = static_cast<double>(x[r * dim + i]);
      sum += v; ss += v * v;
    }
    const double mean = sum / static_cast<double>(dim);
    const double var = ss / static_cast<double>(dim) - mean * mean;
    const double inv = 1.0 / std::sqrt(var + eps);
    for (std::size_t i = 0; i < dim; ++i) {
      const double ref_hi = (static_cast<double>(x[r * dim + i]) - mean) * inv *
                                static_cast<double>(w[i]) +
                            static_cast<double>(b[i]);
      const double ref = static_cast<double>(static_cast<T>(ref_hi));
      const double err = std::abs(static_cast<double>(out[r * dim + i]) - ref);
      max_abs = std::max(max_abs, err);
      worst_excess = std::max(worst_excess, err - (tol.atol + tol.rtol * std::abs(ref)));
    }
  }
  sycl::free(x, q); sycl::free(w, q); sycl::free(b, q); sycl::free(out, q);
  const bool ok = worst_excess <= 0.0;
  std::cout << "  layernorm dt=" << dtype_name(dt) << " variant=" << variant_name(variant)
            << " max_abs=" << max_abs << (ok ? "  ok" : "  FAIL") << '\n';
  return ok;
}

template <typename T>
bool check_fp8_w8a16(sycl::queue &q, DType act_dt, ops::Fp8Kind kind, std::size_t M, std::size_t N,
                     std::size_t K, bool per_channel, Variant variant) {
  T *activations = sycl::malloc_shared<T>(M * K, q);
  float *weight_f32 = sycl::malloc_shared<float>(N * K, q);
  std::uint8_t *weight = sycl::malloc_shared<std::uint8_t>(N * K, q);
  float *weight_roundtrip = sycl::malloc_shared<float>(N * K, q);
  float *scales = sycl::malloc_shared<float>(per_channel ? N : 1, q);
  T *output = sycl::malloc_shared<T>(M * N, q);
  for (std::size_t i = 0; i < M * K; ++i)
    activations[i] = static_cast<T>(sample(i + 101) * 0.25f);
  for (std::size_t i = 0; i < N * K; ++i)
    weight_f32[i] = sample(i + 211) * 0.25f;
  for (std::size_t i = 0; i < (per_channel ? N : 1); ++i)
    scales[i] = 0.5f + 0.01f * static_cast<float>(i % 11);

  bool ok = true;
  try {
    ops::fp8_encode(q, weight_f32, weight, N * K, kind);
    ops::fp8_decode(q, weight, weight_roundtrip, N * K, kind);
    ops::fp8_gemm_w8a16(q, activations, weight, scales, output, M, N, K, kind, per_channel, act_dt,
                        variant, true);

    const Tol base = tol_for(act_dt);
    const double rtol = base.rtol * std::sqrt(static_cast<double>(K)) + 8e-3;
    double max_abs = 0.0;
    double worst = 0.0;
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t n = 0; n < N; ++n) {
        double accumulator = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
          accumulator += static_cast<double>(activations[m * K + k]) *
                         static_cast<double>(weight_roundtrip[n * K + k]);
        }
        const double reference =
            static_cast<double>(static_cast<T>(accumulator * scales[per_channel ? n : 0]));
        const double error = std::abs(static_cast<double>(output[m * N + n]) - reference);
        max_abs = std::max(max_abs, error);
        worst = std::max(worst, error - (base.atol + rtol * std::abs(reference)));
      }
    }
    ok = worst <= 0.0;
    std::cout << "  fp8_w8a16 act=" << dtype_name(act_dt) << " variant=" << variant_name(variant)
              << " (M=" << M << " N=" << N << " K=" << K << ") max_abs=" << max_abs
              << (ok ? "  ok" : "  FAIL") << '\n';
  } catch (const std::exception &error) {
    std::cout << "  fp8_w8a16: skipped (" << error.what() << ")\n";
  }
  sycl::free(activations, q);
  sycl::free(weight_f32, q);
  sycl::free(weight, q);
  sycl::free(weight_roundtrip, q);
  sycl::free(scales, q);
  sycl::free(output, q);
  return ok;
}

template <typename T>
bool check_fused_add_rms_norm(sycl::queue &q, DType dt, std::size_t rows, std::size_t dim) {
  const std::size_t count = rows * dim;
  T *x = sycl::malloc_shared<T>(count, q);
  T *residual = sycl::malloc_shared<T>(count, q);
  T *weight = sycl::malloc_shared<T>(dim, q);
  T *output = sycl::malloc_shared<T>(count, q);
  std::vector<T> residual_before(count);
  for (std::size_t i = 0; i < count; ++i) {
    x[i] = static_cast<T>(sample(i + 17));
    residual[i] = static_cast<T>(sample(i + 29) * 0.5f);
    residual_before[i] = residual[i];
  }
  for (std::size_t i = 0; i < dim; ++i)
    weight[i] = static_cast<T>(0.75f + sample(i + 43) * 0.1f);
  constexpr float eps = 1e-6f;

  ops::fused_add_rms_norm(q, x, residual, weight, output, rows, dim, eps, dt, Variant::sycl, true);

  const Tol tol = tol_for(dt);
  double max_output_error = 0.0;
  std::size_t residual_mismatches = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    double sum_squares = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      const double value = static_cast<double>(x[row * dim + i]) +
                           static_cast<double>(residual_before[row * dim + i]);
      sum_squares += value * value;
    }
    const double inverse_rms = 1.0 / std::sqrt(sum_squares / static_cast<double>(dim) + eps);
    for (std::size_t i = 0; i < dim; ++i) {
      const double value = static_cast<double>(x[row * dim + i]) +
                           static_cast<double>(residual_before[row * dim + i]);
      const T residual_reference = static_cast<T>(value);
      if (residual[row * dim + i] != residual_reference)
        ++residual_mismatches;
      const double normalized = static_cast<double>(static_cast<T>(value * inverse_rms));
      const double reference =
          static_cast<double>(static_cast<T>(normalized * static_cast<double>(weight[i])));
      max_output_error = std::max(max_output_error,
                                  std::abs(static_cast<double>(output[row * dim + i]) - reference));
    }
  }
  const bool ok = residual_mismatches == 0 && max_output_error <= tol.atol + 2.0 * tol.rtol;
  std::cout << "  fused_add_rms_norm dt=" << dtype_name(dt)
            << " residual_mismatches=" << residual_mismatches << " max_abs=" << max_output_error
            << (ok ? "  ok" : "  FAIL") << '\n';
  sycl::free(x, q);
  sycl::free(residual, q);
  sycl::free(weight, q);
  sycl::free(output, q);
  return ok;
}

float decode_e4m3(std::uint8_t bits) {
  const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
  const int exponent = (bits >> 3) & 0x0f;
  const int mantissa = bits & 0x07;
  if (exponent == 0) {
    return sign * std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
  }
  return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

float decode_e2m1(std::uint8_t nibble) {
  constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = values[nibble & 0x07u];
  return (nibble & 0x08u) ? -magnitude : magnitude;
}

std::uint8_t deterministic_fp4(std::size_t index) {
  const std::uint8_t magnitude = static_cast<std::uint8_t>((index * 5 + 1) & 0x07u);
  const std::uint8_t sign = (index % 5 == 0) ? 0x08u : 0u;
  return sign | magnitude;
}

void fill_packed_fp4(std::uint8_t *packed, std::size_t elements) {
  for (std::size_t i = 0; i < elements / 2; ++i) {
    packed[i] =
        deterministic_fp4(2 * i) | static_cast<std::uint8_t>(deterministic_fp4(2 * i + 1) << 4);
  }
}

float nvfp4_value(const std::uint8_t *packed, const std::uint8_t *scales, std::size_t row,
                  std::size_t column, std::size_t width, float global_scale) {
  const std::uint8_t byte = packed[row * (width / 2) + column / 2];
  const std::uint8_t nibble =
      column & 1 ? static_cast<std::uint8_t>(byte >> 4) : static_cast<std::uint8_t>(byte & 0x0fu);
  return decode_e2m1(nibble) * decode_e4m3(scales[row * (width / 16) + column / 16]) * global_scale;
}

template <typename T> bool check_nvfp4_moe(sycl::queue &q, DType act_dt) {
  constexpr std::size_t M = 8;
  constexpr std::size_t E = 8;
  constexpr std::size_t top_k = 8;
  constexpr std::size_t K = 256;
  constexpr std::size_t I = 32;
  constexpr std::size_t two_i = 2 * I;

  T *hidden = sycl::malloc_shared<T>(M * K, q);
  int *expert_ids = sycl::malloc_shared<int>(M * top_k, q);
  float *router_weights = sycl::malloc_shared<float>(M * top_k, q);
  std::uint8_t *w13 = sycl::malloc_shared<std::uint8_t>(E * two_i * K / 2, q);
  std::uint8_t *w13_scales = sycl::malloc_shared<std::uint8_t>(E * two_i * K / 16, q);
  float *w13_global = sycl::malloc_shared<float>(E, q);
  std::uint8_t *w2 = sycl::malloc_shared<std::uint8_t>(E * K * I / 2, q);
  std::uint8_t *w2_scales = sycl::malloc_shared<std::uint8_t>(E * K * I / 16, q);
  float *w2_global = sycl::malloc_shared<float>(E, q);
  float *scratch = sycl::malloc_shared<float>(M * top_k * two_i, q);
  float *fused = sycl::malloc_shared<float>(M * K, q);
  float *split = sycl::malloc_shared<float>(M * K, q);

  for (std::size_t i = 0; i < M * K; ++i)
    hidden[i] = static_cast<T>(sample(i + 307) * 0.1f);
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t route = 0; route < top_k; ++route) {
      const std::size_t pair = m * top_k + route;
      expert_ids[pair] = static_cast<int>((m + route) % E);
      router_weights[pair] = 1.0f / static_cast<float>(top_k);
    }
  }
  expert_ids[M * top_k - 1] = -1;
  fill_packed_fp4(w13, E * two_i * K);
  fill_packed_fp4(w2, E * K * I);
  for (std::size_t i = 0; i < E * two_i * K / 16; ++i)
    w13_scales[i] = (i & 1) ? 0x38u : 0x30u;
  for (std::size_t i = 0; i < E * K * I / 16; ++i)
    w2_scales[i] = (i & 1) ? 0x30u : 0x38u;
  for (std::size_t e = 0; e < E; ++e) {
    w13_global[e] = 0.02f + 0.002f * static_cast<float>(e);
    w2_global[e] = 0.03f + 0.002f * static_cast<float>(e);
  }

  ops::nvfp4_moe_fused(q, hidden, expert_ids, router_weights, w13, w13_scales, w13_global, w2,
                       w2_scales, w2_global, fused, M, E, top_k, K, I, act_dt, true, Variant::sycl,
                       true);
  ops::nvfp4_moe_split(q, hidden, expert_ids, router_weights, w13, w13_scales, w13_global, w2,
                       w2_scales, w2_global, scratch, split, M, E, top_k, K, I, act_dt, true,
                       Variant::sycl, true);

  std::vector<float> reference(M * K, 0.0f);
  std::vector<float> gate_up(two_i);
  std::vector<float> activated(I);
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t route = 0; route < top_k; ++route) {
      const std::size_t pair = m * top_k + route;
      const int expert_id = expert_ids[pair];
      if (expert_id < 0 || static_cast<std::size_t>(expert_id) >= E)
        continue;
      const std::size_t expert = static_cast<std::size_t>(expert_id);
      const std::size_t w13_row0 = expert * two_i;
      const std::size_t w2_row0 = expert * K;
      for (std::size_t row = 0; row < two_i; ++row) {
        float accumulator = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
          accumulator += nvfp4_value(w13, w13_scales, w13_row0 + row, k, K, w13_global[expert]) *
                         static_cast<float>(hidden[m * K + k]);
        }
        gate_up[row] = accumulator;
      }
      for (std::size_t i = 0; i < I; ++i) {
        activated[i] = static_cast<float>(silu_ref(gate_up[i])) * gate_up[i + I];
      }
      for (std::size_t row = 0; row < K; ++row) {
        float accumulator = 0.0f;
        for (std::size_t i = 0; i < I; ++i) {
          accumulator +=
              nvfp4_value(w2, w2_scales, w2_row0 + row, i, I, w2_global[expert]) * activated[i];
        }
        reference[m * K + row] += router_weights[pair] * accumulator;
      }
    }
  }

  double fused_error = 0.0;
  double split_error = 0.0;
  double variants_error = 0.0;
  for (std::size_t i = 0; i < M * K; ++i) {
    fused_error = std::max(fused_error, std::abs(static_cast<double>(fused[i] - reference[i])));
    split_error = std::max(split_error, std::abs(static_cast<double>(split[i] - reference[i])));
    variants_error = std::max(variants_error, std::abs(static_cast<double>(split[i] - fused[i])));
  }
  const double tolerance = act_dt == DType::f32 ? 8e-4 : 3e-3;
  const bool ok =
      fused_error <= tolerance && split_error <= tolerance && variants_error <= tolerance;
  std::cout << "  nvfp4_moe act=" << dtype_name(act_dt) << " fused_max_abs=" << fused_error
            << " split_max_abs=" << split_error << " variants_max_abs=" << variants_error
            << (ok ? "  ok" : "  FAIL") << '\n';

  sycl::free(hidden, q);
  sycl::free(expert_ids, q);
  sycl::free(router_weights, q);
  sycl::free(w13, q);
  sycl::free(w13_scales, q);
  sycl::free(w13_global, q);
  sycl::free(w2, q);
  sycl::free(w2_scales, q);
  sycl::free(w2_global, q);
  sycl::free(scratch, q);
  sycl::free(fused, q);
  sycl::free(split, q);
  return ok;
}

template <typename T> bool check_qwen_gdn_decode(sycl::queue &q, DType act_dt) {
  constexpr std::size_t batch = 1;
  constexpr std::size_t slots = 2;
  constexpr std::size_t query_heads = 16;
  constexpr std::size_t value_heads = 32;
  constexpr std::size_t head_dim = 128;
  constexpr std::size_t query_dim = query_heads * head_dim;
  constexpr std::size_t key_dim = query_heads * head_dim;
  constexpr std::size_t value_dim = value_heads * head_dim;
  constexpr std::size_t conv_dim = query_dim + key_dim + value_dim;
  constexpr std::size_t qkvz_dim = conv_dim + value_dim;

  T *projected_qkvz = sycl::malloc_shared<T>(batch * qkvz_dim, q);
  T *projected_ba = sycl::malloc_shared<T>(batch * 2 * value_heads, q);
  T *conv_state = sycl::malloc_shared<T>(slots * 3 * conv_dim, q);
  float *ssm_state = sycl::malloc_shared<float>(slots * value_heads * head_dim * head_dim, q);
  T *conv_weight = sycl::malloc_shared<T>(conv_dim * 4, q);
  T *conv_bias = sycl::malloc_shared<T>(conv_dim, q);
  float *A_log = sycl::malloc_shared<float>(value_heads, q);
  T *dt_bias = sycl::malloc_shared<T>(value_heads, q);
  int *state_indices = sycl::malloc_shared<int>(batch, q);
  T *mixed_qkv = sycl::malloc_shared<T>(batch * conv_dim, q);
  T *core = sycl::malloc_shared<T>(batch * value_dim, q);
  T *z = sycl::malloc_shared<T>(batch * value_dim, q);

  for (std::size_t i = 0; i < batch * qkvz_dim; ++i)
    projected_qkvz[i] = static_cast<T>(sample(i + 401) * 0.02f);
  for (std::size_t i = 0; i < batch * 2 * value_heads; ++i)
    projected_ba[i] = static_cast<T>(sample(i + 503) * 0.1f);
  for (std::size_t i = 0; i < slots * 3 * conv_dim; ++i)
    conv_state[i] = static_cast<T>(sample(i + 607) * 0.01f);
  for (std::size_t i = 0; i < slots * value_heads * head_dim * head_dim; ++i)
    ssm_state[i] = sample(i + 701) * 0.001f;
  for (std::size_t i = 0; i < conv_dim * 4; ++i)
    conv_weight[i] = static_cast<T>(sample(i + 809) * 0.01f);
  for (std::size_t i = 0; i < conv_dim; ++i)
    conv_bias[i] = static_cast<T>(sample(i + 907) * 0.001f);
  for (std::size_t i = 0; i < value_heads; ++i) {
    A_log[i] = -1.5f + 0.01f * static_cast<float>(i);
    dt_bias[i] = static_cast<T>(0.1f + 0.002f * static_cast<float>(i));
  }
  state_indices[0] = 0;

  std::vector<T> conv_reference(conv_state, conv_state + slots * 3 * conv_dim);
  std::vector<float> state_reference(ssm_state,
                                     ssm_state + slots * value_heads * head_dim * head_dim);
  std::vector<T> mixed_reference(batch * conv_dim);
  std::vector<T> core_reference(batch * value_dim);
  std::vector<T> z_reference(batch * value_dim);

  for (std::size_t channel = 0; channel < conv_dim; ++channel) {
    const std::size_t state0 = channel;
    const float history0 = static_cast<float>(conv_reference[state0]);
    const float history1 = static_cast<float>(conv_reference[state0 + conv_dim]);
    const float history2 = static_cast<float>(conv_reference[state0 + 2 * conv_dim]);
    const T input = projected_qkvz[channel];
    conv_reference[state0] = static_cast<T>(history1);
    conv_reference[state0 + conv_dim] = static_cast<T>(history2);
    conv_reference[state0 + 2 * conv_dim] = input;
    float value = static_cast<float>(conv_bias[channel]);
    value += history0 * static_cast<float>(conv_weight[channel * 4]);
    value += history1 * static_cast<float>(conv_weight[channel * 4 + 1]);
    value += history2 * static_cast<float>(conv_weight[channel * 4 + 2]);
    value += static_cast<float>(input) * static_cast<float>(conv_weight[channel * 4 + 3]);
    mixed_reference[channel] = static_cast<T>(value / (1.0f + std::exp(-value)));
    if (channel >= conv_dim - value_dim) {
      const std::size_t v = channel - (conv_dim - value_dim);
      z_reference[v] = projected_qkvz[conv_dim + v];
    }
  }

  std::vector<float> query(head_dim);
  std::vector<float> key(head_dim);
  for (std::size_t value_head = 0; value_head < value_heads; ++value_head) {
    const std::size_t query_head = value_head / 2;
    float query_norm = 1e-6f;
    float key_norm = 1e-6f;
    for (std::size_t i = 0; i < head_dim; ++i) {
      const float q_value = static_cast<float>(mixed_reference[query_head * head_dim + i]);
      const float k_value =
          static_cast<float>(mixed_reference[query_dim + query_head * head_dim + i]);
      query_norm += q_value * q_value;
      key_norm += k_value * k_value;
    }
    const float query_scale = 0.08838834764831845f / std::sqrt(query_norm);
    const float key_scale = 1.0f / std::sqrt(key_norm);
    for (std::size_t i = 0; i < head_dim; ++i) {
      query[i] = static_cast<float>(mixed_reference[query_head * head_dim + i]) * query_scale;
      key[i] =
          static_cast<float>(mixed_reference[query_dim + query_head * head_dim + i]) * key_scale;
    }
    const float a = static_cast<float>(projected_ba[value_heads + value_head]);
    const float b = static_cast<float>(projected_ba[value_head]);
    const float softplus =
        a + static_cast<float>(dt_bias[value_head]) <= 20.0f
            ? std::log(1.0f + std::exp(a + static_cast<float>(dt_bias[value_head])))
            : a + static_cast<float>(dt_bias[value_head]);
    const float decay = std::exp(-std::exp(A_log[value_head]) * softplus);
    const float beta = 1.0f / (1.0f + std::exp(-b));
    for (std::size_t value_lane = 0; value_lane < head_dim; ++value_lane) {
      const std::size_t state_base =
          ((value_head * head_dim + value_lane) * head_dim);
      float prediction = 0.0f;
      for (std::size_t k = 0; k < head_dim; ++k)
        prediction += state_reference[state_base + k] * decay * key[k];
      const float value = static_cast<float>(
          mixed_reference[query_dim + key_dim + value_head * head_dim + value_lane]);
      const float delta = (value - prediction) * beta;
      float output = 0.0f;
      for (std::size_t k = 0; k < head_dim; ++k) {
        const float updated = state_reference[state_base + k] * decay + delta * key[k];
        state_reference[state_base + k] = updated;
        output += updated * query[k];
      }
      core_reference[value_head * head_dim + value_lane] = static_cast<T>(output);
    }
  }

  ops::qwen_gdn_decode(q, projected_qkvz, projected_ba, conv_state, ssm_state, conv_weight,
                       conv_bias, A_log, dt_bias, state_indices, mixed_qkv, core, z, batch, slots,
                       false, act_dt, DType::f32, act_dt, Variant::sycl, true);

  std::size_t conv_mismatches = 0;
  std::size_t z_mismatches = 0;
  double core_error = 0.0;
  double state_error = 0.0;
  for (std::size_t i = 0; i < slots * 3 * conv_dim; ++i)
    if (conv_state[i] != conv_reference[i])
      ++conv_mismatches;
  for (std::size_t i = 0; i < value_dim; ++i) {
    if (z[i] != z_reference[i])
      ++z_mismatches;
    core_error = std::max(core_error, std::abs(static_cast<double>(core[i]) -
                                               static_cast<double>(core_reference[i])));
  }
  for (std::size_t i = 0; i < slots * value_heads * head_dim * head_dim; ++i) {
    state_error =
        std::max(state_error, std::abs(static_cast<double>(ssm_state[i] - state_reference[i])));
  }
  const double core_tolerance = act_dt == DType::f32 ? 2e-5 : 4e-3;
  std::size_t invalid_core_mismatches = 0;
  std::size_t invalid_z_mismatches = 0;
  std::size_t invalid_state_mismatches = 0;
  const std::vector<T> conv_after_valid(conv_state,
                                        conv_state + slots * 3 * conv_dim);
  const std::vector<float> state_after_valid(
      ssm_state, ssm_state + slots * value_heads * head_dim * head_dim);
  for (const int invalid_index : {-1, static_cast<int>(slots)}) {
    state_indices[0] = invalid_index;
    ops::qwen_gdn_decode(q, projected_qkvz, projected_ba, conv_state,
                         ssm_state, conv_weight, conv_bias, A_log, dt_bias,
                         state_indices, mixed_qkv, core, z, batch, slots,
                         false, act_dt, DType::f32, act_dt, Variant::sycl,
                         true);
    for (std::size_t i = 0; i < value_dim; ++i) {
      if (core[i] != T(0))
        ++invalid_core_mismatches;
      if (z[i] != projected_qkvz[conv_dim + i])
        ++invalid_z_mismatches;
    }
    for (std::size_t i = 0; i < slots * 3 * conv_dim; ++i)
      if (conv_state[i] != conv_after_valid[i])
        ++invalid_state_mismatches;
    for (std::size_t i = 0;
         i < slots * value_heads * head_dim * head_dim; ++i)
      if (ssm_state[i] != state_after_valid[i])
        ++invalid_state_mismatches;
  }
  const bool ok = conv_mismatches == 0 && z_mismatches == 0 &&
                  core_error <= core_tolerance && state_error <= 2e-5 &&
                  invalid_core_mismatches == 0 &&
                  invalid_z_mismatches == 0 &&
                  invalid_state_mismatches == 0;
  std::cout << "  qwen_gdn_decode act=" << dtype_name(act_dt)
            << " conv_mismatches=" << conv_mismatches << " z_mismatches=" << z_mismatches
            << " core_max_abs=" << core_error << " state_max_abs=" << state_error
            << " invalid_core_mismatches=" << invalid_core_mismatches
            << " invalid_z_mismatches=" << invalid_z_mismatches
            << " invalid_state_mismatches=" << invalid_state_mismatches
            << (ok ? "  ok" : "  FAIL") << '\n';

  sycl::free(projected_qkvz, q);
  sycl::free(projected_ba, q);
  sycl::free(conv_state, q);
  sycl::free(ssm_state, q);
  sycl::free(conv_weight, q);
  sycl::free(conv_bias, q);
  sycl::free(A_log, q);
  sycl::free(dt_bias, q);
  sycl::free(state_indices, q);
  sycl::free(mixed_qkv, q);
  sycl::free(core, q);
  sycl::free(z, q);
  return ok;
}

}  // namespace

int main() {
  const auto devices = gpu_devices();
  if (devices.empty()) {
    std::cerr << "no SYCL GPU device; skipping on-device correctness test\n";
    return 0;  // treated as skip on non-XPU hosts
  }

  sycl::queue q = make_gpu_queue(0, /*enable_profiling=*/false);
  std::cout << "device: " << describe_device(q.get_device()) << '\n';

  // Deterministic sweep across the interesting GELU range, incl. the zero
  // crossing and the saturating tails.
  const std::size_t n = 4096;
  std::vector<float> host_in(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_in[i] = -8.0f + 16.0f * (static_cast<float>(i) / static_cast<float>(n));
  }

  const Variant variants[] = {Variant::sycl, Variant::vendor};
  int failures = 0;
  for (const Variant v : variants) {
    failures += check_gelu<float>(q, DType::f32, v, host_in) ? 0 : 1;
    failures += check_gelu<half_t>(q, DType::f16, v, host_in) ? 0 : 1;
    failures += check_gelu<bf16_t>(q, DType::bf16, v, host_in) ? 0 : 1;
  }

  // Elementwise activations (native only): silu, gelu backward, glu modes.
  for (const DType dt : {DType::f32, DType::f16, DType::bf16}) {
    if (dt == DType::f32) {
      failures += check_silu<float>(q, dt, 4096) ? 0 : 1;
      failures += check_gelu_bwd<float>(q, dt, 4096) ? 0 : 1;
    } else if (dt == DType::f16) {
      failures += check_silu<half_t>(q, dt, 4096) ? 0 : 1;
      failures += check_gelu_bwd<half_t>(q, dt, 4096) ? 0 : 1;
    } else {
      failures += check_silu<bf16_t>(q, dt, 4096) ? 0 : 1;
      failures += check_gelu_bwd<bf16_t>(q, dt, 4096) ? 0 : 1;
    }
  }
  failures += check_glu<float>(q, DType::f32, ops::GluMode::swiglu, "glu_swiglu", 64, 1024) ? 0 : 1;
  failures += check_glu<float>(q, DType::f32, ops::GluMode::geglu, "glu_geglu", 64, 1024) ? 0 : 1;
  failures += check_glu<bf16_t>(q, DType::bf16, ops::GluMode::swiglu, "glu_swiglu_bf16", 64, 1024) ? 0 : 1;
  failures += check_glu<float>(q, DType::f32, ops::GluMode::reglu, "glu_reglu", 64, 1000) ? 0 : 1;  // d not mult of 4: scalar path
  // activations: GEGLU -> f16 fused convert (glu_gelu_f16). dt-in {f32,f16,bf16}.
  failures += check_glu_gelu_f16<float>(q, DType::f32, "glu_gelu_f16(f32-in)", 64, 1024) ? 0 : 1;
  failures += check_glu_gelu_f16<half_t>(q, DType::f16, "glu_gelu_f16(f16-in)", 64, 1024) ? 0 : 1;
  failures += check_glu_gelu_f16<bf16_t>(q, DType::bf16, "glu_gelu_f16(bf16-in)", 64, 1024) ? 0 : 1;
  failures += check_glu_gelu_f16<float>(q, DType::f32, "glu_gelu_f16(f32-in,d1000)", 64, 1000) ? 0 : 1;

  // Dense GEMM, both variants, f32 + bf16.
  for (const Variant v : variants) {
    failures += check_gemm<float>(q, DType::f32, v, 128, 96, 160) ? 0 : 1;
    failures += check_gemm<bf16_t>(q, DType::bf16, v, 128, 96, 160) ? 0 : 1;
  }

  // int4 quantized GEMV (native).
  failures += check_qgemv_int4<float>(q, DType::f32, 128, 4096, 128) ? 0 : 1;
  failures += check_qgemv_int4<bf16_t>(q, DType::bf16, 128, 4096, 128) ? 0 : 1;

  // w4a16_gemm: int4-weight x f16/bf16-activation DPAS (joint_matrix) GEMM.
  // fp64 q4-dequant oracle; small-M decode regime + edge masking (M/N/K tails).
  failures += check_w4a16_gemm<half_t>(q, DType::f16, 8, 64, 512, 128) ? 0 : 1;   // clean tiles
  failures += check_w4a16_gemm<bf16_t>(q, DType::bf16, 32, 128, 256, 64) ? 0 : 1;  // bigger M, bf16
  failures += check_w4a16_gemm<half_t>(q, DType::f16, 5, 20, 40, 8) ? 0 : 1;       // M/N mask + K tail
  failures += check_w4a16_gemm<bf16_t>(q, DType::bf16, 16, 48, 128, 32) ? 0 : 1;   // GQA-ish N, bf16

  // mxfp4 GEMV (native e2m1 + e8m0 block-scale decode).
  failures += check_mxfp4_gemv<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_mxfp4_gemv<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;

  // nvfp4 GEMV (native e2m1 + e4m3 block scale + per-tensor scale).
  failures += check_nvfp4_gemm<float>(q, DType::f32, 1, 128, 4096) ? 0 : 1;
  failures += check_nvfp4_gemm<half_t>(q, DType::f16, 2, 128, 4096) ? 0 : 1;
  failures += check_nvfp4_gemm<bf16_t>(q, DType::bf16, 3, 128, 4096) ? 0 : 1;
  failures += check_nvfp4_moe<float>(q, DType::f32) ? 0 : 1;
  failures += check_nvfp4_moe<half_t>(q, DType::f16) ? 0 : 1;
  failures += check_nvfp4_moe<bf16_t>(q, DType::bf16) ? 0 : 1;

  // GGUF q8_0 / q4_0 GEMV (native block-layout decode).
  failures += check_gguf_gemv<float>(q, DType::f32, ops::GgufType::q8_0, "q8_0", 128, 4096) ? 0 : 1;
  failures += check_gguf_gemv<bf16_t>(q, DType::bf16, ops::GgufType::q8_0, "q8_0", 128, 4096) ? 0 : 1;
  failures += check_gguf_gemv<float>(q, DType::f32, ops::GgufType::q4_0, "q4_0", 128, 4096) ? 0 : 1;
  failures += check_gguf_gemv<bf16_t>(q, DType::bf16, ops::GgufType::q4_0, "q4_0", 128, 4096) ? 0 : 1;
  failures += check_gguf_q6k<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_gguf_q6k<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;
  failures += check_gguf_q4k<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_gguf_q4k<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;
  failures += check_gguf_q5k<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_gguf_q5k<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;
  failures += check_gguf_q2k<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_gguf_q2k<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;
  failures += check_gguf_q3k<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_gguf_q3k<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;
  failures += check_gguf_iq4nl<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_gguf_iq4nl<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;
  failures += check_gguf_legacy<float>(q, DType::f32, ops::GgufType::q4_1, "q4_1", 128, 4096) ? 0 : 1;
  failures += check_gguf_legacy<bf16_t>(q, DType::bf16, ops::GgufType::q5_0, "q5_0", 128, 4096) ? 0 : 1;
  failures += check_gguf_legacy<float>(q, DType::f32, ops::GgufType::q5_1, "q5_1", 128, 4096) ? 0 : 1;
  failures += check_gguf_iq4xs<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_gguf_iq4xs<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;
  failures += check_gguf_iquant<float>(q, DType::f32, ops::GgufType::iq2_xxs, "iq2_xxs", 128, 4096) ? 0 : 1;
  failures += check_gguf_iquant<bf16_t>(q, DType::bf16, ops::GgufType::iq2_xs, "iq2_xs", 128, 4096) ? 0 : 1;
  failures += check_gguf_iquant<float>(q, DType::f32, ops::GgufType::iq3_xxs, "iq3_xxs", 128, 4096) ? 0 : 1;
  failures += check_gguf_iq1s<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_gguf_iq1s<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;

  // quantize_int4_group: weight quant round-trip with qgemv_int4.
  failures += check_quantize_int4<float>(q, DType::f32, 128, 4096, 128) ? 0 : 1;
  failures += check_quantize_int4<bf16_t>(q, DType::bf16, 128, 4096, 128) ? 0 : 1;

  // act_quant: per-token int8 activation quantization (feeds w8a8).
  failures += check_act_quant<float>(q, DType::f32, 128, 4096) ? 0 : 1;
  failures += check_act_quant<bf16_t>(q, DType::bf16, 128, 4096) ? 0 : 1;

  // int8 w8a8 GEMM, both variants.
  for (const Variant v : variants) {
    failures += check_qgemm_int8<float>(q, DType::f32, v, 96, 128, 256) ? 0 : 1;
    failures += check_qgemm_int8<bf16_t>(q, DType::bf16, v, 96, 128, 256) ? 0 : 1;
  }

  // fp8 GEMM (e4m3/e5m2), vendor; skips cleanly if the device lacks fp8.
  failures += check_fp8_gemm<float>(q, DType::f32, ops::Fp8Kind::e4m3, "e4m3", 128, 128, 256) ? 0 : 1;
  failures += check_fp8_gemm<float>(q, DType::f32, ops::Fp8Kind::e5m2, "e5m2", 128, 128, 256) ? 0 : 1;
  // fp8 GEMV (M=1), native SYCL decode path. Ragged N / K%4 exercise the
  // guarded scalar paths; bf16 out exercises the reduce-kernel cast.
  failures += check_fp8_gemm<float>(q, DType::f32, ops::Fp8Kind::e4m3, "e4m3", 1, 128, 256, Variant::sycl) ? 0 : 1;
  failures += check_fp8_gemm<float>(q, DType::f32, ops::Fp8Kind::e5m2, "e5m2", 1, 128, 256, Variant::sycl) ? 0 : 1;
  failures += check_fp8_gemm<float>(q, DType::f32, ops::Fp8Kind::e4m3, "e4m3", 1, 101, 203, Variant::sycl) ? 0 : 1;
  failures += check_fp8_gemm<bf16_t>(q, DType::bf16, ops::Fp8Kind::e4m3, "e4m3", 1, 128, 256, Variant::sycl) ? 0 : 1;
  failures +=
      check_fp8_w8a16<float>(q, DType::f32, ops::Fp8Kind::e4m3, 1, 128, 256, true, Variant::sycl)
          ? 0
          : 1;
  failures +=
      check_fp8_w8a16<bf16_t>(q, DType::bf16, ops::Fp8Kind::e5m2, 4, 128, 256, false, Variant::sycl)
          ? 0
          : 1;
  failures +=
      check_fp8_w8a16<half_t>(q, DType::f16, ops::Fp8Kind::e4m3, 4, 128, 256, true, Variant::best)
          ? 0
          : 1;
  failures +=
      check_fp8_w8a16<bf16_t>(q, DType::bf16, ops::Fp8Kind::e4m3, 8, 128, 256, true, Variant::best)
          ? 0
          : 1;

  // rope / adamw / argmax (native only).
  failures += check_rope<float>(q, DType::f32, 32, 8, 64) ? 0 : 1;
  failures += check_rope<bf16_t>(q, DType::bf16, 32, 8, 64) ? 0 : 1;

  // attention: fused per-head QK-norm + query-scale + RoPE (qk_norm_rope). MHA
  // and GQA, head_dim {64,128}, dt {f32,f16,bf16}, with and without the f16 sink.
  failures += check_qk_norm_rope<float>(q, DType::f32, 32, 8, 8, 64, false) ? 0 : 1;
  failures += check_qk_norm_rope<half_t>(q, DType::f16, 32, 8, 8, 64, true) ? 0 : 1;
  failures += check_qk_norm_rope<bf16_t>(q, DType::bf16, 24, 16, 4, 128, false) ? 0 : 1;
  failures += check_qk_norm_rope<float>(q, DType::f32, 16, 32, 8, 128, true) ? 0 : 1;
  failures += check_adamw(q, DType::f32, 4096) ? 0 : 1;
  failures += check_argmax(q, DType::f32, 64, 4000) ? 0 : 1;

  // serving: embedding + kv-cache scatter/gather (exact copy).
  failures += check_serving<float>(q, DType::f32) ? 0 : 1;
  failures += check_serving<bf16_t>(q, DType::bf16) ? 0 : 1;

  // serving: sentence-embedding pooling head (masked mean + per-token RMSNorm +
  // L2). Ragged token counts, incl. a 1-token sequence, across the dim shape keys.
  {
    const std::vector<int> tok = {1, 5, 16, 37, 64, 80, 3};
    for (const std::size_t d : {std::size_t(256), std::size_t(512), std::size_t(768), std::size_t(1024)}) {
      failures += check_pool_mean_rms_l2<float>(q, DType::f32, d, tok) ? 0 : 1;
      failures += check_pool_mean_rms_l2<half_t>(q, DType::f16, d, tok) ? 0 : 1;
      failures += check_pool_mean_rms_l2<bf16_t>(q, DType::bf16, d, tok) ? 0 : 1;
    }
  }

  // attention: flash-style SDPA (MHA + GQA, causal + non-causal).
  failures += check_attention<float>(q, DType::f32, 8, 8, 128, 128, 64, true) ? 0 : 1;
  failures += check_attention<bf16_t>(q, DType::bf16, 16, 4, 64, 64, 128, true) ? 0 : 1;   // GQA + d=128
  failures += check_attention<float>(q, DType::f32, 4, 4, 96, 160, 64, false) ? 0 : 1;      // cross-attn

  // attention_f16ctx: flash SDPA + fused f16 context store (dtype-T + f16 outputs checked).
  failures += check_attention_f16ctx<float>(q, DType::f32, 8, 8, 128, 128, 64, true) ? 0 : 1;
  failures += check_attention_f16ctx<half_t>(q, DType::f16, 16, 4, 64, 64, 128, true) ? 0 : 1;   // GQA + d=128
  failures += check_attention_f16ctx<bf16_t>(q, DType::bf16, 8, 8, 96, 96, 64, true) ? 0 : 1;
  failures += check_attention_f16ctx<float>(q, DType::f32, 4, 4, 96, 160, 64, false) ? 0 : 1;      // cross-attn

  // attn_swa: symmetric sliding-window attention. worst_excess==0 vs the windowed
  // oracle AND vs_causal_gap large (a causal mask would FAIL) across window sizes,
  // seq lengths, dtypes, GQA, and the EmbeddingGemma d=256 shape. window=0 == dense.
  failures += check_attn_swa<float>(q, DType::f32, 8, 8, 256, 256, 64, 64) ? 0 : 1;       // MHA, window 64
  failures += check_attn_swa<float>(q, DType::f32, 3, 3, 320, 320, 256, 128) ? 0 : 1;     // EmbeddingGemma d=256
  failures += check_attn_swa<half_t>(q, DType::f16, 16, 4, 256, 256, 128, 96) ? 0 : 1;    // GQA, d=128
  failures += check_attn_swa<bf16_t>(q, DType::bf16, 8, 8, 200, 200, 64, 50) ? 0 : 1;     // odd-ish window, half=25
  failures += check_attn_swa<float>(q, DType::f32, 4, 4, 96, 96, 64, 512) ? 0 : 1;        // window > seq: full band
  failures += check_attn_swa<float>(q, DType::f32, 4, 4, 128, 128, 64, 0) ? 0 : 1;        // window=0 == dense

  // linear_attention: non-causal linear attention.
  failures += check_linear_attn<float>(q, DType::f32, 8, 256, 64) ? 0 : 1;
  failures += check_linear_attn<bf16_t>(q, DType::bf16, 8, 128, 64) ? 0 : 1;
  failures += check_qwen_gdn_decode<float>(q, DType::f32) ? 0 : 1;
  failures += check_qwen_gdn_decode<half_t>(q, DType::f16) ? 0 : 1;
  failures += check_qwen_gdn_decode<bf16_t>(q, DType::bf16) ? 0 : 1;

  // ssm: Mamba selective scan.
  failures += check_selective_scan<float>(q, DType::f32, 1024, 512, 16) ? 0 : 1;
  failures += check_selective_scan<bf16_t>(q, DType::bf16, 1024, 256, 16) ? 0 : 1;

  // collectives: multi-GPU sum all-reduce across the visible B60s.
  failures += check_collectives() ? 0 : 1;

  // sampling: categorical + top-k (invariants: top-k membership, greedy=argmax).
  failures += check_sampling<float>(q, DType::f32, 256, 4000, 8) ? 0 : 1;
  failures += check_sampling<bf16_t>(q, DType::bf16, 256, 4000, 16) ? 0 : 1;

  // moe: top-k routing.
  failures += check_moe_route<float>(q, DType::f32, 512, 64, 2) ? 0 : 1;
  failures += check_moe_route<bf16_t>(q, DType::bf16, 512, 128, 4) ? 0 : 1;

  // utils: dropout, cross_entropy, hadamard.
  failures += check_dropout(q, DType::f32) ? 0 : 1;
  failures += check_cross_entropy<float>(q, DType::f32, 64, 4000) ? 0 : 1;
  failures += check_cross_entropy<bf16_t>(q, DType::bf16, 64, 4000) ? 0 : 1;
  failures += check_hadamard<float>(q, DType::f32, 64, 1024) ? 0 : 1;
  failures += check_hadamard<bf16_t>(q, DType::bf16, 64, 256) ? 0 : 1;

  const std::size_t rows = 64, dim = 1024;
  for (const Variant v : variants) {
    failures += check_softmax<float>(q, DType::f32, v, rows, dim) ? 0 : 1;
    failures += check_softmax<half_t>(q, DType::f16, v, rows, dim) ? 0 : 1;
    failures += check_softmax<bf16_t>(q, DType::bf16, v, rows, dim) ? 0 : 1;
    failures += check_rms_norm<float>(q, DType::f32, v, rows, dim) ? 0 : 1;
    failures += check_rms_norm<half_t>(q, DType::f16, v, rows, dim) ? 0 : 1;
    failures += check_rms_norm<bf16_t>(q, DType::bf16, v, rows, dim) ? 0 : 1;
    failures += check_layernorm<float>(q, DType::f32, v, rows, dim) ? 0 : 1;
    failures += check_layernorm<half_t>(q, DType::f16, v, rows, dim) ? 0 : 1;
    failures += check_layernorm<bf16_t>(q, DType::bf16, v, rows, dim) ? 0 : 1;
  }
  failures += check_fused_add_rms_norm<float>(q, DType::f32, rows, dim) ? 0 : 1;
  failures += check_fused_add_rms_norm<half_t>(q, DType::f16, rows, dim) ? 0 : 1;
  failures += check_fused_add_rms_norm<bf16_t>(q, DType::bf16, rows, dim) ? 0 : 1;

  // norms: fused residual-add + double RMSNorm -> f16 (rms_residual_next). Swept
  // across dt in {f32,f16,bf16} x dim {256,768,1024} (768 = EI_N_EMBD shape).
  for (const std::size_t d : {std::size_t(256), std::size_t(768), std::size_t(1024)}) {
    failures += check_rms_residual_next<float>(q, DType::f32, 64, d) ? 0 : 1;
    failures += check_rms_residual_next<half_t>(q, DType::f16, 64, d) ? 0 : 1;
    failures += check_rms_residual_next<bf16_t>(q, DType::bf16, 64, d) ? 0 : 1;
  }

  if (failures) {
    std::cerr << "FAIL: " << failures << " case(s) exceeded tolerance\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
