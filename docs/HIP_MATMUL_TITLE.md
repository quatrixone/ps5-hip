# Tiled matmul title candidate

The `hip_matmul` title uses the existing metadata-driven geometry/LDS runtime,
one genuine compiler registration, eight blocking launches and the unchanged
measured `blockDim` shader. Four matrix shapes run with two changed-input rounds;
982 active results use the exact integer-valued FP32 oracle documented in
HIP_TILED_MATMUL_CLIENT.md. The runtime observes actual registration/completion
counts; these checks belong to this diagnostic client and title, not admission.

A fresh `--title-entry --compiler-only` probe compiles the host entry explicitly
as `ps5hip_matmul_client_main`. The builder requires its pinned receipt, combined
object and embedded device bytes, then reuses the existing private CRT ordering,
RELRO/bundle bounds, helper elevation, log persistence and shell-exit protocol.
The wrapper accepts success only after the application oracle passes and after
genuine compiler unregister, State destruction and complete allocation closure.
Terminal errors quarantine without a shell request. All final observations read
process-lifetime POD, not a destroyed runtime object.

The host title model uses genuine compiler constructors and destructor calls,
actual runtime/ownership code, and synthetic transport/service callbacks. It
checks clean errors, uncertain ownership, missing/corrupt outputs, input/guard
mutations, failed/late registration, rejected/skipped unregister, late live data,
log persistence and shell errors. These models are not AMDGPU ISA emulation.

The isolated candidate remains offline until native link, signed-container and
full package extraction audits pass and a separate bounded experiment is reviewed.
No full G4, arbitrary GEMM accuracy or performance claim follows from this title.
Single-thread/module blocking-runtime limits and caller indexing responsibility
remain unchanged. Installed SDK/resource support is not a hermetic inventory.
