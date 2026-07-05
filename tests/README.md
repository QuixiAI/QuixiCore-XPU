# Tests

The current test suite is a scaffold smoke test for backend metadata and contract
status reporting.

Run it with:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Future kernel tests should compare native XPU outputs against deterministic host
references and report the same tolerance fields used by the umbrella QuixiCore
contract.
