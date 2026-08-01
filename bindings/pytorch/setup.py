"""Build the QuixiCore XPU PyTorch extension (tk_xpu).

Requires: `source /opt/intel/oneapi/setvars.sh` (icpx on PATH), the `sycl` CMake
preset already built (provides libquixicore_xpu_ops.a), and torch+xpu installed.

    cd bindings/pytorch && ../../.venv/bin/python setup.py build_ext --inplace
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, SyclExtension

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DNNL = os.environ.get("DNNLROOT", "/opt/intel/oneapi/dnnl/latest")
# The ops must be a SHARED library built with icpx -fsycl: a static .a linked into
# the extension does not get its SYCL device-image registration wired into the
# runtime ProgramManager (kernel-submit segfaults). Build it with:
#   cmake -S . -B build-sycl-shared -G "Unix Makefiles" -DCMAKE_CXX_COMPILER=icpx \
#         -DQUIXICORE_XPU_ENABLE_SYCL=ON -DBUILD_SHARED_LIBS=ON && cmake --build ...
OPS_DIR = os.path.join(REPO, "build-sycl-shared")

setup(
    name="tk_xpu",
    ext_modules=[
        SyclExtension(
            name="tk_xpu",
            sources=[os.path.join(HERE, "tk_xpu_ext.sycl")],
            include_dirs=[
                os.path.join(REPO, "include"),
                os.path.join(DNNL, "include"),
            ],
            library_dirs=[OPS_DIR, os.path.join(DNNL, "lib")],
            libraries=["quixicore_xpu_ops", "quixicore_xpu", "dnnl"],
            # rpath so the shared ops lib + oneDNN resolve at import without setvars.
            extra_link_args=[f"-Wl,-rpath,{OPS_DIR}", f"-Wl,-rpath,{os.path.join(DNNL, 'lib')}"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
