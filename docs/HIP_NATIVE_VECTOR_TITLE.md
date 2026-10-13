# Bounded native HIP vector title

This title wraps the frozen genuine HIP vector control in the existing Eden and
finite shell-managed lifecycle. It remains one serialized gfx1013, wave32 client:
73 elements, two blocking launches, 288-byte kernargs, null stream and zero LDS.
It does not broaden the runtime API, qualify hardware, or close G4.

`hip_vector_title.cpp` reads and validates `/app0/results-dir.txt` before requesting
the pinned Eden filesystem capability from `/app0/lapy.elf`. It exclusively creates
`ps5-hip-gfx1013-hip_vector.log`, checks runtime primitives and runs the real full-header
client. Constructor registration still occurs before main without native allocation
or AGC initialization. Its early stdout records are not copied into the result log;
a distinctly labelled post-redirection snapshot checks one real module/function,
empty data ownership and no initialized backend instead.

The private startup copy installs the unchanged shell callback, then the title
finalizer, before `_init`. The real compiler constructor creates the runtime State
and registers its real module destructor. Successful main return therefore runs
these callbacks in reverse order: existing shutdown probes, compiler unregister,
State destruction, title finalizer, and the existing application-fini/shell request.
Neither the wrapper nor client manually unregisters a compiler module.

The State checks actual registry ranges, module/function counts, both completed
launches, empty configurations and absence of poison. It explicitly destroys the
idle queue before its backend. A first-declared sentinel marks completion only
after every other member is destroyed. The finalizer reads only a trivially
initialized external observation, never the destroyed State. It prints the final
proof only if all checks passed, checks stream errors and flush/fsync, and otherwise
waits for the bounded host controller without requesting shell exit. The client's
main success proves computation only; the strict log checker requires the later
unregister, destroyed-State proof and successful shell request return as well.

Clean client exceptions explicitly attempt closure of all three data allocations.
Terminal failure stops local teardown and retains unresolved GPU ownership; failed
safe cleanup is also terminal and cannot become title success. Nonzero client
return, partial registration, missing finalizer, failed persistence, failed close,
or a failed final observation all prevent the title from reaching shell success.
Early configuration/elevation/log-open failures return nonzero to the existing
shell guard. An existing result file is never overwritten.

`hip_vector_title_test.py` links the actual saved compiler object and executes its
constructors/destructor with host-only native-memory/backend/Eden/shell doubles.
It exercises normal ordering, partial/skipped/rejected unregister, late retained
data, submission errors/timeouts/unknown exceptions, output/input/guard corruption,
allocation/close errors, configuration/elevation/O_EXCL, persistence and shell errors.
The original transcripts retain explicit model audit lines; only those known lines
are removed for a separate production grammar check. Missing or mutated teardown
proof is rejected. The callbacks calculate vector outputs; they do not emulate
AMD ISA. Fixed test mappings use the documented ASan shadow-gap accommodation and
explicit full guards, not heap redzones or evidence about console allocator safety.

Packaging must bind the genuine embedded HIP object and linker bounds, the private
startup patch, exact helper and shell import providers. No external `canary.hsaco`
is executed by this profile. A package or passing host model does not establish
native execution, GPU correctness, process/layer release or service health; those
remain separate bounded hardware observations. No such observation is claimed here.

The offline native recipe is `tools/build_native_vector_title.py --reference-root
<pinned-local-input-repository>`. It requires the exact prepared SDK child link in
this checkout's real `.deps` directory and resolves HIP headers, the genuine saved
combined object, native tools, zlib and the Eden helper from that reference. It
copies the native source and elevation sources into a fresh output before making
the title-specific startup/linker changes. It never rebuilds shared dependencies.
For a development attempt, supply `--output build/native/gfx1013-hip_vector-<attempt>`;
the canonical output is reserved for qualification after source freeze. Every
attempt retains commands, stdout/stderr, input hashes and the failed stage.

The recipe checks the exact genuine compiler object/bundle, raw PIE and converted
ELF wrapper/constructor/RELRO/relocations, linked startup installer order, real
compiler `atexit` unregister reference, native import providers, signed-container
reconstruction and pinned clean-room libc companion. The prospective reconstruction
criterion requires equal lengths and every byte exact except the converter's one
final, unmapped, 24-byte SIE build-id note: the pinned signer intentionally omits
PT_NOTE, so that extent reconstructs as zero. Its header/name/location, disjointness
from all other stored ranges, original bytes and both whole-file hashes are recorded.
Any difference outside that exact note is rejected. Development attempt v2 remains
a preserved failure under the earlier whole-file equality criterion.
The stage retains linked audit
inputs outside the application and rechecks them, including a narrow comparison
of the pinned signer's plaintext FSELF segments against the audited ELF. Those
files provide local provenance; they are not installed in the application image.

After all source owners freeze the reviewed change, build the canonical output,
then run `tools/stage_native_title.py --target gfx1013 --kernel hip_vector
--results-dir <ordinary-absolute-directory>` and `tools/package_native_title.py
--stage <new-stage-directory>`. The existing generic packager runs UFS fsck and a
complete extracted-file comparison before publishing the image. The title contains
only the signed executable, helper, libc companion, metadata, icon and result-path
configuration; there is no external `canary.hsaco`. Frozen catalog/controller
evidence remains the authority for `verify_native_case.py` before a later console
run. Packaging itself never adds a candidate to that authority.

The recipe inventories consumed HIP headers, tool binaries and selected SDK
archives/stubs before and after use. It does not inventory every installed SDK
header or every toolchain support file and is not a hermetic SDK distribution.
Native startup/destructor behavior and filesystem permissions still require a
separate guarded console milestone.
