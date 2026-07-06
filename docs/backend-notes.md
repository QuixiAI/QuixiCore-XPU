# Backend Notes

QuixiCore XPU is the Intel GPU implementation of the QuixiCore contract. XPU
source may use SYCL, Level Zero, XeTLA, oneDNN, Triton XPU backend experiments,
and oneCCL when those choices remain behind the shared operation semantics.

The repository is currently a scaffold. New kernel work should start directly
in the semantic family layout documented in `docs/repository-structure.md`.
