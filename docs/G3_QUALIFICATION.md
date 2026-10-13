# Compute qualification status

This is the current evidence map for G3. A result applies to its frozen image,
compiler profile and tested shape. Host models do not execute GPU instructions;
structural loader acceptance does not qualify arbitrary compatible-looking code.

| Feature | Current evidence | Remaining scope |
| --- | --- | --- |
| Baseline arithmetic | G2 combined arithmetic image passed two clean cycles; `../evidence/native-g2-acceptance.json` | Other shapes, instructions and compiler profiles |
| Private calls | 304-byte private-call fixture produced 16,448 exact values; delayed completion evidence; `../evidence/native-private304-computation.json` | Original bounded cycle remains inconclusive; generic address spaces and broader private/call shapes |
| Compiler spilling | 4,096 exact outputs, modified scratch, intact guards and clean bounded cycle; `../evidence/native-register-spill.json` | Other spill layouts and optimization/resource combinations |
| Shared memory | Static, dynamic and mixed LDS fixtures each produced 257 exact group results; `../evidence/native-lds-static.json`, `../evidence/native-lds-dynamic.json`, `../evidence/native-lds-mixed.json` | Other sizes, shapes and optimization levels; static post-run audit is parent-only |
| Dependent dispatch and ownership | Three changed-input rounds over persistent buffers produced 771 exact final values; recovered result and delayed cleanup independently audited; `../evidence/native-visibility-computation.json` | Original bounded cycle remains inconclusive; asynchronous ordering, GPU-only publication and other ownership patterns |
| Wave32 operations | Full and partial native fixtures each passed 256 checked words and clean bounded cycles; independent audits clear; `../evidence/native-wave-full.json`, `../evidence/native-wave-partial.json` | Arbitrary active masks, broader shapes and compiler profiles |
| 2D/3D geometry | 2D/3D fixtures passed 384/1536 records and clean bounded cycles; independent audits clear; `../evidence/native-geometry-2d.json`, `../evidence/native-geometry-3d.json` | Other shapes and optimization profiles; arbitrary block shapes remain unsupported |
| Atomics | Six relaxed atomic fixture kernels packaged; `../evidence/offline-atomic-native-readiness.json` | Native results, current-substrate review, additional memory orders/scopes and publication semantics |
| FP32 device math | Fixed numerical corpus and native package prepared; `../evidence/offline-native-math-readiness.json` | Native results under the predeclared numerical policy; broader domains and modes |
| Addressing, FP16 and linking | Existing loader and pointer checks cover bounded subsets | Native large-offset/legal-alignment cases, FP16/integer boundaries, separate compilation, globals/constants and required relocations |
| Integrated runtime | Combined planners at `296222d`, owned preparation at `7aed16b`; 65 sanitizer tests,26 genuine vector host-model cases and review passed; `../evidence/offline-owned-native-integration.json` | Connect broader HIP launch paths and qualify combined runtime interactions; private-call admission remains excluded |

The visibility path completes each queue submission and includes CPU cache
maintenance. It does not establish GPU synchronization without CPU involvement.
The delayed private-call and visibility captures disclose unlocked/uncaptured
intervals and do not retroactively convert their original cycles to passes.

The geometry path admits measured 64-thread shapes with genuine full-header HIP
builtins and checked COv5 hidden arguments. It is an additional device-launch path,
not general multidimensional `hipLaunchKernel` compatibility. The original initial
launch API remains restricted to its earlier shape and resource contract.

G3 remains open. Its acceptance still requires the feature families and boundaries
in `IMPLEMENTATION_PLAN.md`, multiple compiler optimization profiles, and the
capabilities needed by G4 clients. Wave64, general ROCm binary compatibility,
general HIP application compatibility and framework readiness are not established
by these fixtures.
