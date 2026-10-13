# Genuine four-route HIP compiler and runtime control

`tests/hip_routes.hip` is one full-header HIP translation unit with four real
kernel definitions and ordinary `<<<>>>` call sites. One compiler module registers
four distinct host tokens. The client repeats integer XOR, static LDS reduction,
forced spilling and measured 2D geometry on the same three 64 KiB allocations,
changing all 8,192 input words for the second sequence. It checks entire input,
output and tail allocations, including untouched guards. Its five CPU oracle
pages are heap-owned; no large main-thread stack allowance is assumed.

The bounded shapes are two groups of 64 for the first three entries, and
3x2x1 groups of 8x8x1 for geometry. Exact outputs are 128 XOR values, two unsigned
64-value sums, 8,192 copied words, and 384 sixteen-word geometry records plus
273 logical tail values and 111 unchanged sentinels. The geometry record stores
block index, local index, block/grid sizes, global coordinates and input value.
No runtime allocator infers these indexing bounds; the separate fixed adapter
enforces the client contract before native preparation.

The measured O2 gfx1013 entry profiles are integer 24-byte kernargs/0x8c, LDS
256 static bytes plus 16-byte kernargs/0x8c, spill 212 private bytes plus 16-byte
kernargs/0x8d, and geometry 296-byte kernargs/0x138c. All use properties 0x409,
wave32, no preload/resource3. The first development artifact correctly failed
the ISA gate: generic volatile spill arguments produced flat external accesses.
Only those known-global arguments now receive internal address-space-1 casts;
scratch remains buffer loads/stores. The public argument/stub ABI is unchanged.
This does not establish support for arbitrary generic or private pointers.

Run `tools/probe_hip_routes.py --reference-root <pinned-input-repository>
--output <fresh-directory>`. The recipe checks pinned HIP 6.2.4 headers and
workitem/constants, LLVM 18.1.8 tools, three genuine Linux/PS5/Linux-model combined
objects, descriptors/ISA, compiler wrapper and four constructor registrations.
Device-only wchar routing preserves the PS5 host default. Host transport models
exercise 19 cases, including changed-input and guard corruption, partial/late
registration failure, clean scratch setup failure, uncertain scratch ownership,
and completed cleanup failure. Uncertain paths must retain eight native model
allocations and reject later operations without releasing anything; completed
scratch cleanup failure retains four and reports GPU ownership resolved.
The late duplicate test proves its name lies in the authoritative linked span,
uses a genuine handle/token and requires finalization to remain false after all
eight earlier successes. Model callbacks calculate outputs; they do not execute
AMD ISA or prove cache behavior.

The final phase privately copies the pinned native-tool sources, links the real
PS5 combined object with the shared runtime and fixed adapter, converts the ELF,
and audits wrapper/bundle/RELRO/constructor boundaries and native imports. It
creates no signed title, package or console operation. Shared native exports,
selected SDK libraries, zlib artifacts, tools and consumed source identities are
checked before/after use. Installed SDK/resource support trees are not fully
inventoried; this is not a hermetic SDK distribution. `--compiler-only` and
`--host-only` are development stops and do not satisfy the complete saved-proof
checker.

Run `tests/hip_routes_recipe_test.py` for deliberate transcript mutations, or
`tests/hip_routes_recipe_test.py --check <completed-output>` to recheck retained
provenance and every model transcript without recompiling. A passing offline
proof does not qualify these shared sources on hardware, broaden the fixed
client into arbitrary HIP support, or close G4.
