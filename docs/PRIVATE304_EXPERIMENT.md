# Native private-memory call experiment

Status: exact private304 computation passed with separately audited delayed
completion evidence. The original bounded cycle remains inconclusive; full G3 is open.
The prerequisite arithmetic fixture passed G2 on .40; see
[G2 acceptance](../evidence/native-g2-acceptance.json).

The frozen candidate is `56dcbcf8f0b47808267064ec3553c17dd820018e`, image
`c539b22061334a283bae136f0c2d3404fd047cca38e9e8cf07594803d431db74`.
[Native readiness](../evidence/offline-private-buffer-native-readiness.json)
binds its package, experimental compiler reproduction, profile and host tests.
Use these exact bytes, not a rebuild with the installed compiler.

## Hypothesis and oracle

The selected ordinary O2 gfx1013 kernel can execute its retained nested calls
using buffer-backed private memory through the existing eight-SGPR native route.
Its root frame is 144 bytes and middle frame 160 bytes, totaling 304 bytes per
lane. The callee writes through a private pointer; the caller reloads and uses
the modified value. This tests compiler lowering, call state and scratch access
together, rather than inferring support from descriptor metadata.

Run the named `nested_private` entry under diagnostic profile
`private_calls_buffer`: 257 groups of 64 threads, wave32, 16,448 invocations.
Require all 16,448 exact outputs, unchanged inputs, modified scratch payload,
unchanged complete allocation guards and closed allocations. The guarded scratch
ring contains 13,107,200 payload bytes with a 10,240-byte wave stride and 64 KiB
prefix/suffix guards. The queue wait is bounded to 2,000 ms; timeout or quarantine
rejects the experiment and never permits freeing memory still owned by the GPU.

## Deployment and lifecycle

Use PPSA99014 on idle .40, firmware 12.70, under both owned shared locks and the
current runbook. Preserve the known arithmetic image and result. The deployed
ShadowMount version caches images by path: wait for explicit old-image unmount
after its source is renamed away before promoting the verified temporary image.
Do not force an unmount or change settings. Verify both the stored package and
the mounted executable/kernel bytes, as well as title metadata and mount link.

For ftpsrv, negotiate raw SELF transfer on the fresh FTP connection and require
the exact disabled-mode acknowledgement before comparing signed executable bytes.
The [pinned upstream handler](https://github.com/ps5-payload-dev/ftpsrv/blob/23737c4738d15b9b9b036665bbf008d8fd6d82d2/cmd.c)
toggles only connection-local conversion; its default SELF-to-ELF download can
produce different bytes and lengths. A mode mismatch is a verification failure,
not permission to widen the bound or accept a transformed executable.

Review deployment separately before making an execution control runnable. Bind
that control to the actual deployment receipt. Before launch, reverify installed
and mounted bytes, require an absent private result log, and take a fresh complete
positive title-mount baseline. The new image may use a different logical-device
slot from the arithmetic image; compare its own pre/post identities exactly.

Require a fresh PID/application identity matching the nonnegative launch return,
the exact application transcript and all finalization checkpoints, natural
self-stop before host cleanup, ordered OS process/sandbox/EndAppMount completion,
unchanged visible title-mount multiset, healthy declared services and release of
only owned locks. Observe within the reviewed 20-second host window. Each payload
phase is single-use; preserve every failed or incomplete observation.

Mismatch, fault, missing evidence, timeout, quarantine, changed mount identity,
or unhealthy services rejects acceptance. Diagnose failures offline before another
attempt; never automatically repeat a suspected panic.

## What a pass establishes

A pass supports this exact ordinary, buffer-only O2 private-call fixture and its
declared native setup. It does not qualify generic or volatile private pointers,
all compiler optimizations, arbitrary call graphs, flat-scratch addressing,
general G3 semantics or a full HIP runtime. Visible mount equivalence does not
prove global namespace completeness or release of unobserved references.

After a pass, record the exact capability and continue the remaining G3 cases,
including spills, LDS/barriers and other compiler/launch shapes. After a failure,
use the saved application, ISA and OS evidence to choose the next diagnostic;
do not change the oracle to accommodate the result.

## Observed result

The first launch produced all 16,448 exact outputs, modified scratch, unchanged
guards, closed allocations and full finalization, followed by natural PID123 /
application `0x6018` teardown. A shell-focus lag caused the original controller
to refuse cleanup before fetching its result; that cycle remains inconclusive.

A separately reviewed completion capture fetched the unchanged 642-byte result
without another GPU launch. The mounted candidate matched, saved Shadow history
extended without an intervening title/registration event, and a delayed snapshot
matched the original visible mount baseline. The post send occurred 452.68 seconds
after launch; locks and klog were not continuous across that interval. The later
close controller found the application already absent, rather than killing it.

Independent composed review accepts the exact computation and eventual visible
baseline only. It does not retroactively accept the original 20-second cycle or
qualify full G3. See `../evidence/native-private304-computation.json`. The next
frozen candidate is `register_spill`, with 52 real compiler spill/reload pairs and
4,096 exact outputs, followed by the static, dynamic and mixed LDS cases.
