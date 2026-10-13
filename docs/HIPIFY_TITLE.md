# HIPIFY title lifecycle (offline candidate)

The unchanged reviewed CUDA fixture and unedited HIPIFY output retain the ordinary
`int main()` body. The compiler flag renames it to `hipify_client_main`; the title
wrapper calls that exact no-argument C++ signature. This profile uses the existing
generic metadata admission, owned runtime and finite shell lifecycle, with no new
HIP API. The single genuine compiler registration precedes main; its unregister
and complete State destruction must precede the POD finalizer and shell request.

The title observes one module/function, six completed launches, 662 active wrapping
integer sums and 2,238 full-buffer word comparisons. Lengths 1/73/257 each run twice
with changing inputs. Both inputs and seven-word guards on each end are compared.
`HIP_TITLE COMPUTE` precedes finalization; only the late finalizer may report closed
ownership. The early private startup installer must remain before `_init` when
native linking is added. Constructor messages before log redirection are not
replayed as application evidence: the wrapper records an actual later snapshot.

Eden requests filesystem capability and the ordinary configurable results directory
before an exclusive new `ps5-hip-gfx1013-hipify.log`. Neither `/download0` nor a
fixed `/data` result path is required. The existing shell probe remains unchanged.
Every error, uncertain ownership, missing unregister/destruction or persistence
failure quarantines without requesting shell exit. Host lifecycle tests use actual
compiler objects and real runtime ownership with explicit service/transport mocks;
they do not execute AMD ISA or qualify native title teardown.

```
python3 tests/hipify_title_test.py --reference-root ../.. \
  --probe-dir ../g4-hipify-runtime/results/runtime-v4 --output results/title-model-v1
python3 tests/hipify_title_test.py --check results/title-model-v1
```

The strict log checker is scoped to `gfx1013`, `hipify`, and shell-exit mode.
Generated HIP and original CUDA bytes must remain identical to the reviewed
compiler checkpoint. Native link/package and a frozen prospective console control
remain separate steps; no hardware result is claimed here.
