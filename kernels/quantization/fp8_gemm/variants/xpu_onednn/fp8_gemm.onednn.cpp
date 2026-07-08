// fp8 GEMM (e4m3/e5m2), vendor variant via oneDNN matmul on Intel XMX. fp8 is a
// data encoding, not NVIDIA-exclusive silicon: this exercises the B60's fp8 path
// natively. Also exposes oneDNN reorder-based f32<->fp8 codecs so tests build
// fp8 inputs and references without a hand-written encoder. All gated behind
// oneDNN; whether the device actually builds an fp8 matmul is verified at run
// time (measure, don't assume).

#include "quantization/fp8_gemm/fp8_kernel.hpp"

#if defined(QUIXICORE_XPU_HAS_ONEDNN)

#include <mutex>
#include <sstream>
#include <unordered_map>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace quixicore::xpu::kernels {
namespace {

dnnl::memory::data_type fp8_dt(int kind) {
  return kind == 1 ? dnnl::memory::data_type::f8_e5m2
                   : dnnl::memory::data_type::f8_e4m3;
}
dnnl::memory::data_type out_dt_of(DType dt) {
  switch (dt) {
    case DType::f32: return dnnl::memory::data_type::f32;
    case DType::f16: return dnnl::memory::data_type::f16;
    case DType::bf16: return dnnl::memory::data_type::bf16;
  }
  return dnnl::memory::data_type::f32;
}

dnnl::engine make_engine(sycl::queue& q) {
  return dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
}

}  // namespace

// Engine + primitive cache. Measured on B60 (4096^3 e4m3): rebuilding the
// engine/primitive every call cost ~3 ms on a 3.8 ms matmul — nearly half the
// shipped fp8_gemm time was setup, not GEMM. Keyed per SYCL context (SYCL 2020
// guarantees std::hash for sycl::context) with the shape/attr key inside.
namespace {
struct EngCache {
  dnnl::engine eng;
  std::unordered_map<std::string, dnnl::matmul> prims;
};
std::mutex g_mu;
std::unordered_map<sycl::context, EngCache> g_cache;
}  // namespace

bool fp8_gemm_onednn(sycl::queue& q, const void* a, const void* b, void* c,
                     std::size_t M, std::size_t N, std::size_t K, int kind,
                     float scale, DType out_dt) {
  try {
    using tag = dnnl::memory::format_tag;
    const auto f8 = fp8_dt(kind);
    const dnnl::memory::desc a_md({(dnnl::memory::dim)M, (dnnl::memory::dim)K}, f8, tag::ab);
    const dnnl::memory::desc b_md({(dnnl::memory::dim)K, (dnnl::memory::dim)N}, f8, tag::ab);
    const dnnl::memory::desc c_md({(dnnl::memory::dim)M, (dnnl::memory::dim)N}, out_dt_of(out_dt), tag::ab);

    dnnl::engine eng;
    dnnl::matmul prim;
    {
      std::lock_guard<std::mutex> lk(g_mu);
      auto& ec = g_cache[q.get_context()];
      if (!ec.eng) ec.eng = make_engine(q);
      eng = ec.eng;
      std::ostringstream os;
      os << M << 'x' << N << 'x' << K << '/' << kind << '/' << (int)out_dt
         << '/' << sycl::bit_cast<std::uint32_t>(scale);
      auto it = ec.prims.find(os.str());
      if (it == ec.prims.end()) {
        dnnl::primitive_attr attr;
        // Measured on B60: f16 fpmath lets oneDNN up-convert fp8 to f16 and
        // take the XMX f16 route — 44.8 vs 35.8 TFLOP/s at 4096^3 (exact:
        // both fp8 kinds embed losslessly in f16, accumulation stays f32).
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
        if (scale != 1.0f) {
          dnnl::post_ops po;
          po.append_eltwise(dnnl::algorithm::eltwise_linear, scale, 0.0f);
          attr.set_post_ops(po);
        }
        dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md, attr);
        it = ec.prims.emplace(os.str(), dnnl::matmul(pd)).first;
      }
      prim = it->second;
    }
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
    auto usm = [&](const dnnl::memory::desc& m, void* p) {
      return dnnl::sycl_interop::make_memory(m, eng, dnnl::sycl_interop::memory_kind::usm, p);
    };
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, usm(a_md, const_cast<void*>(a))},
        {DNNL_ARG_WEIGHTS, usm(b_md, const_cast<void*>(b))},
        {DNNL_ARG_DST, usm(c_md, c)},
    };
    dnnl::sycl_interop::execute(prim, strm, args).wait();
    return true;
  } catch (const dnnl::error&) {
    return false;  // fp8 matmul unsupported on this device/library
  }
}

static void reorder_1d(sycl::queue& q, const void* in, dnnl::memory::data_type it,
                       void* out, dnnl::memory::data_type ot, std::size_t n) {
  dnnl::engine eng = make_engine(q);
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
  using tag = dnnl::memory::format_tag;
  const dnnl::memory::desc imd({(dnnl::memory::dim)n}, it, tag::a);
  const dnnl::memory::desc omd({(dnnl::memory::dim)n}, ot, tag::a);
  auto im = dnnl::sycl_interop::make_memory(imd, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(in));
  auto om = dnnl::sycl_interop::make_memory(omd, eng, dnnl::sycl_interop::memory_kind::usm, out);
  dnnl::reorder(im, om).execute(strm, im, om);
  strm.wait();
}

void fp8_from_f32(sycl::queue& q, const float* in, void* out_fp8, std::size_t n, int kind) {
  reorder_1d(q, in, dnnl::memory::data_type::f32, out_fp8, fp8_dt(kind), n);
}
void fp8_to_f32(sycl::queue& q, const void* in_fp8, float* out, std::size_t n, int kind) {
  reorder_1d(q, in_fp8, fp8_dt(kind), out, dnnl::memory::data_type::f32, n);
}
bool fp8_supported() { return true; }

}  // namespace quixicore::xpu::kernels

#endif  // QUIXICORE_XPU_HAS_ONEDNN
