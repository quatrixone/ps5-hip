# LDS qualification contract

Status: static LDS passed one native .40 cycle; dynamic and mixed LDS remain unrun.
Preflight and deployment were independently reviewed. Saved execution evidence was
parent-audited after a content filter interrupted independent post-run review.

The frozen candidate source is `27165fd51fb16798d6bc340b5f254c8298c501b5`.
Package identities are in `../evidence/offline-lds-native-readiness.json`; fresh
compiler reproduction is in `../evidence/offline-lds-reproduction-v2.json`.

## Computation and oracle

Each of 257 workgroups has 64 threads, spanning two wave32 waves. There are
16,448 input values and exactly **257 output sums**. For group `g` and lane `l`,
the input is `g * 4099 + l * 17 + (l >= 32 ? 65536 : 0)`. The expected reduction
is `g * 262336 + 2131424`, within uint32 range for every tested group. The upper
wave's extra contribution makes simply omitting that wave observable.

Every thread reaches the initial barrier and all six reduction barriers. Each
phase halves the active adders, from 32 to one, while the barriers remain outside
the lane conditional. The saved static gfx1013 O2 ISA contains LDS writes/reads,
restoration of the execution mask and seven barriers. This is compiler evidence,
not proof of hardware scheduling or visibility.

| Profile | Static bytes | Dynamic bytes | Allocated bytes | Oracle |
| --- | ---: | ---: | ---: | --- |
| static | 256 | 0 | 1024 | Reduction sum |
| dynamic | 0 | 1280 | 2048 | Same sum, stride 5 |
| mixed | 256 | 1280 | 2048 | Sum XOR `((g * 64 + 33) XOR 0x53a971bd)` |

The mixed case checks separation of compiler-placed dynamic storage from a static
tag array, including a tag loaded from lane 33. The offline padded-layout oracle
also models strides 1 and 16; these extra strides are not native-qualified cases.

## Required native evidence

Validate static first, then dynamic and mixed. For each exact frozen image require
257 exact output sums, all 16,448 inputs unchanged, full global allocation guards,
successful allocation closure and every finalization checkpoint. The expected
log must declare the exact allocation profile above. Queue timeout remains 2,000
ms; mismatches, missing evidence or uncertain completion cannot count as success.

Use the existing bounded lifecycle and artifact-identity contract only once its
new controls are ready and reviewed. Preserve the successful spill image/result.
Do not interpret host models, checksum identity or a partial log as a GPU pass.
Record natural title shutdown, service health and visible mount-baseline evidence
with the exact computation. Full G3 also needs its other feature families and
broader optimization/geometry/visibility cases; these three fixtures cannot close
the whole gate.

## Latest offline verification

Ten LDS host tests passed, including native-harness oracle tests. Fresh LLVM18
compilation at O1/O2 for gfx1013/gfx1030 reproduced all four existing code objects
byte-for-byte. All three expected-output binaries also reproduced exactly. The
frozen source remains clean. No code-object or oracle change was needed.

## Static observation

PID141/application `0xa018` produced 257 exact sums, with every input unchanged,
guards intact, allocations closed and all finalization checkpoints. It stopped
naturally before host cleanup. The bounded cycle recorded matching visible mount
identity multisets, healthy services and released locks. Parent audit rehashed all
25 raw files and reproduced the frozen acceptance result with networking blocked.
See `../evidence/native-lds-static.json` and
`../evidence/native-lds-static-deployment.json`. No broader optimization, geometry,
dynamic/mixed LDS or full G3 qualification follows from this exact fixture.

Dynamic execution controls have passed 25 offline regression methods and real
candidate verification with a synthetic test-only deployment binding. Their actual
deployment binding remains null until a separately verified deployment exists.
