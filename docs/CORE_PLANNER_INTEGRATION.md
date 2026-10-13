# Checked planner integration

This work starts from `development/offline-bringup` at `e6725fd`, not the
repository's `main` branch. It imports only the cumulative pure planner changes
from private-buffer source `56dcbcf`: buffer scratch from `1e0428f`, LDS from
`27165fd`, and the structural eight-SGPR entry from `6f7d9a1`.

The allocator, data registry, native launch helper, queue, AGC boundary, genuine
HIP control and title lifecycle remain unchanged. No native fixture builder or
HIP admission is widened. InitialLaunch keeps its six-word storage and the
initial encoder still rejects eight-word/scratch/LDS states. The selected
eight-SGPR encoder assembles its separate two raw-base words locally.

The imported APIs are pure checked planning, not hardware or shader-safety
certificates. In particular, `flat_call_launch_rejection` cannot distinguish the
qualified private304 compiler graph from arbitrary generic/volatile shaders with
the same metadata. Exact compiler graph/provenance qualification stays outside
these APIs; the normal HIP runtime continues to reject this profile. Likewise,
dynamic LDS capacity does not imply support for hidden dynamic-size or shared
pointer arguments. Combined scratch+LDS and other unmeasured entry states reject.

Tests retain the established scratch/LDS/flat-entry negative matrices and decode
actual immutable O1/O2 objects. A new integration test verifies route separation
and a literal initial packet baseline. Current ownership/submission tests exercise
the modern cleanup behavior against the integrated library. No device code is
recompiled. The actual-private304 tests consume the selected artifacts from the
pinned compiler reproduction; structural288 tests remain explicitly unqualified.

This step does not integrate geometry, generic runtime dispatch selection,
multiple registered kernel support, or any new console package. Those remain
subsequent reviewable changes. No hardware capability or G3/G4 completion follows.
