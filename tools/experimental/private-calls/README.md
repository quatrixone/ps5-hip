# Experimental private-call compiler recipe

This directory preserves the bounded specialization plugin v3 and backend patch
v4 as reproducible research sources. **This compiler route and portable recipe
remain experimental and unfrozen.** The independent paired-IPRA-off measurement
is complete for all eight ordinary/volatile profiles; portable preparation and
measurement are separate from full backend acceptance. No receipt here claims
an accepted compiler or native execution.

V4 rejects `-amdgpu-experimental-prove-buffer-calls` when
`TM->Options.EnableIPRA` is true. Every clean-baseline, patched-default-off and
patched-enabled comparison must explicitly pass `-enable-ipra=false`; disabling
IPRA for only the candidate would invalidate the comparison. This is a compiler
invocation option, not a CMake build option. Default-off behavior remains a
separate paired control. Only the dedicated guard regressions intentionally
exercise default/IPRA-on configurations to require failure before proof.

All eight independently measured ordinary/volatile nested profiles (gfx1013 and
gfx1030, O1 and O2) require 304 private bytes: 144 root + 160 middle + 0 leaf.
The volatile mixed private/global root requires 144 bytes. Compared with the
stock IPRA-on nested route's 288 bytes, the middle frame increases from 144 to
160: ABI-preserved VGPR slots move the array origin and lane-save spill, with
16-byte frame rounding. The exact ordinary postcall window includes two data
loads, two ABI restore loads and `vmcnt(2)` before the observable sum.

The frozen native harness's 288-byte model describes its original stock fixture
and must not be relabeled for this new compiler route. Any future native route
needs separately derived frame, ring-sizing and resource qualification.

V2's actual enabled run reported the graph proven but retained two flat-scratch
initialization writes: `MachineRegisterInfo::isPhysRegUsed(FLAT_SCR)` includes
call-clobber register masks. V3 retained actual explicit/implicit register-use
checks while distinguishing masks after proof, but its actual matrix then failed
root-resource equivalence with default IPRA. The module-wide machine-function
barrier changes interprocedural register-allocation ordering. V4 adds the explicit
IPRA guard; it does not erase either failed result. Both earlier patch digests are
preserved in `pins.json`, and zero-initializer/resource-equivalence acceptance
must remain strict.

This is not the selected production compiler, an accepted replacement compiler,
a native package route, or evidence of PS5 execution. No production dependency
lock is changed. The real ordinary and volatile fixtures retain nested noinline
calls and generic source pointer types. No assembly rewrite, forged descriptor,
manual `amdgpu-calls` attribute, or forced inlining is part of this recipe.

## Source and tool identities

`pins.json` records the exact LLVM 18.1.8 source commit, original research paths
and SHA-256 digests, copied-file digests, and the narrow portability adaptations.
The plugin and patch are byte-identical to the original research artifacts. The
patch's historical DRAFT comment is retained as provenance. Historical auditors and the compact measurement are byte-exact references.
Versioned active adapters change receipt/path binding only; their provenance is
listed beside the originals. Old scripts, wrapper source and output receipts are
retained, and later separate acceptance work is not implied.

Use Linux/WSL Python 3, Git, CMake, Ninja, ccache, and LLVM **18.1.8** tools on PATH:
`clang++-18`, `opt-18`, `llvm-config-18`, `ld.lld-18`, `llvm-readobj-18`, and
`llvm-objdump-18`; LLVM development headers/libraries are needed for the plugin.
The recipe checks exact versions, records resolved executable and linked-library
hashes before/after, and verifies the actual HIP linker selected by the driver.
Local rebuilt tool hashes are observations, not assumed equal to the original
Ubuntu binaries. A different distribution/build must pass the same checks.

## Separate baseline and patched llc builds

These are explicit manual commands. Neither `prepare` nor `validate` builds,
installs, or selects llc. Use native Linux disk with room for two AMDGPU builds;
avoid a small `/tmp` tmpfs. Four compile jobs and one link job bound each build.
Run the two builds sequentially unless machine resources justify overlap.

```sh
# Set these to absolute paths; spaces are supported when quoted.
recipe="$PWD/tools/experimental/private-calls"
build_root="$HOME/ps5hip-private-compiler"
mkdir -p "$build_root"
git clone --filter=blob:none --no-checkout --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git "$build_root/source"
git -C "$build_root/source" sparse-checkout init --cone
git -C "$build_root/source" sparse-checkout set llvm cmake third-party
git -C "$build_root/source" checkout --detach 3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff
git -C "$build_root/source" worktree add --detach "$build_root/source-patched" 3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff
git -C "$build_root/source-patched" apply --check "$recipe/backend/buffer-call-proof-v4.patch"
git -C "$build_root/source-patched" apply "$recipe/backend/buffer-call-proof-v4.patch"

configure() {
  cmake -S "$1/llvm" -B "$2" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
    -DLLVM_TARGETS_TO_BUILD=AMDGPU -DLLVM_ENABLE_PROJECTS= -DLLVM_ENABLE_RUNTIMES= \
    -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_PARALLEL_COMPILE_JOBS=4 -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DLLVM_USE_LINKER=lld-18 -DLLVM_CCACHE_BUILD=ON -DLLVM_ENABLE_TERMINFO=OFF \
    '-DCMAKE_CXX_FLAGS=-include cstdint'
}
configure "$build_root/source" "$build_root/baseline"
ninja -C "$build_root/baseline" -j4 llc
configure "$build_root/source-patched" "$build_root/patched"
ninja -C "$build_root/patched" -j4 llc
```

The identical C++-only `-include cstdint` is a host portability workaround for
LLVM18 headers with the current libstdc++; it changes neither checked-out source
nor device compiler flags. Keep full commands, clean/patched Git identities,
configuration/cache and build logs separately. The validation wrapper records
binary identities but cannot prove how a supplied llc was built. Assertions must
remain enabled for proof diagnostics. The patch's flag is default off:
`-amdgpu-experimental-prove-buffer-calls`.

## Prepare, then validate each mode explicitly

```sh
out="$PWD/results/private compiler experiment"
python3 "$recipe/recipe.py" prepare --output "$out" --llvm-source "$build_root/source"
python3 "$recipe/recipe.py" measure --output "$out" --compiler "$build_root/baseline/bin/llc"
python3 "$recipe/recipe.py" validate --output "$out" --compiler "$build_root/baseline/bin/llc" --mode baseline
python3 "$recipe/recipe.py" validate --output "$out" --compiler "$build_root/patched/bin/llc" --mode patched-off
python3 "$recipe/recipe.py" guard-controls --output "$out" --baseline-compiler "$build_root/baseline/bin/llc" --patched-compiler "$build_root/patched/bin/llc"
# All selected callers explicitly use paired -enable-ipra=false.
python3 "$recipe/recipe.py" validate --output "$out" --compiler "$build_root/patched/bin/llc" --mode patched-on
# After those modes pass, retain explicit/implicit flat-register rejection:
python3 "$recipe/recipe.py" register-controls --output "$out" --baseline-compiler "$build_root/baseline/bin/llc" --patched-compiler "$build_root/patched/bin/llc"
# Only after every mode really succeeds:
python3 "$recipe/recipe.py" audit --output "$out" --allow-frame-move
```

Preparation requires a new output directory and clean pinned LLVM checkout. It
builds the small plugin, compiles the two real HIP sources at O1/O2 for gfx1013 and
gfx1030 (IR plus linked objects), checks actual linker selection, runs all five
plugin checks, generates the 24 ordinary/generic/volatile ordered/reordered IR
cases and negative/stress inputs, and runs synthetic checker tests. It preserves
commands, failures and hashes locally. Existing output is refused; do not retry a
failed operation in place. Use a new output tree after any source/tool change.
Python optimization (`python -O`/`PYTHONOPTIMIZE`) is forbidden because the
reviewed research checks use assertions.

`measure` independently compiles the eight fresh original ordinary/volatile
inputs with the clean baseline llc and explicit IPRA-off. Its CMake source path
must match the clean pinned checkout; executable, libraries and build-cache
identities are recorded and rechecked. This is provenance checking, not proof
that an arbitrary supplied binary was built honestly; preserve the build logs.
The reused analysis must reproduce **all** pinned semantic fields: descriptors,
frames, call edges, load/store effects, save/restore operands and observable
postcall windows. Only current compiler/input/raw-file identities and the labeled
historical IPRA-on comparison are separate provenance. No historical raw hash is
silently substituted for a new output. The generated binding rehashes all fresh
inputs and raw artifacts, and both active matrix and linked audits require it.
Measurement also runs frame and linked-graph corruption tests without invoking
the patched compiler. A new build/path requires its own fresh measurement.

Baseline validation also generates MIR mutations from that newly built baseline's
stress outputs. Each mode runs ordinary, generic, volatile, negative, spill,
late-callee-save, post-PEI reload, generation-ineligible and ten MIR controls.
Four additional register-operand controls explicitly compare the baseline and
enabled compiler; they remain a separate requested operation. Default-off must
match baseline bytes. Enabled positives require the complete
machine-function inventory before proof, resolved retained calls, unchanged
resources/eight-slot descriptor, buffer private accesses, and zero flat init.
The v2 diagnostic checker binds the entire present machine-function set to both
linked symbol tables. It allows only the two exact internal generic originals
that are absent from both and unreferenced outside their jointly removed bodies;
every retained function still needs a result. Its real-record mutation test runs
after fresh enabled validation, never against a fabricated positive result.
The ordinary post-audit independently allows only the exact initializer removal
and verified linked-call displacement changes. The explicit `--allow-frame-move`
option additionally permits one immediate s32 initialization to commute through
a bounded entry prefix of known instructions with no s32 dependency; no other
root/callee instruction, operand, frame or descriptor change is permitted.
Omitting the option retains the earlier strict scheduling comparison. The extended post-audit also
requires exact rejected-case bytes and the exact
four-instruction removal in positive alias, late-save and MIR controls. Its
alias-only rule permits two **literal**, internally ordered entry blocks to
commute after proving their register effects disjoint; it is not a general
instruction scheduler or normalization rule. Existing alias objects are also
linked separately, and byte-bound disassembly must resolve the exact direct call
target, argument source and return pairs. Matched byte/ISA corruption tests must
reject wrong displacements, carry and register pairs. The alias audit invokes
only linking and inspection of existing objects, not llc compilation. Its versioned
portable adapter requires the resolved linker and disassembler recorded during
preparation, rechecks their bytes and recorded libraries, and has no fixed-path
fallback. The original audit and parser remain preserved. Measurement and final
audit run the corresponding post-auditors' mutation tests. Full selected
acceptance still requires separately recorded build provenance; `run_all.py`
alone is not full compiler acceptance.

No assertion should be weakened to accommodate changed output. Source aliases
canonicalized away are safe positive controls; actual surviving MIR aliases are
negative controls. Unsupported dynamic alloca is a pre-proof failure, not a
successful proof rejection. Reloading MIR after the proof barrier must retain
conservative initialization because compiler-owned proof is not serialized.

## Semantic and hardware limits

The specialization plugin retains generic fallback and refuses unsupported
pointer use/ABI attributes. It has 64 new clones per module and eight descendants
per original generic function per pass invocation; it is not a general HIP
compiler transform. Volatile operation count is preserved, but stock LLVM18's
address-space inference keeps generic volatile accesses flat. The real volatile
and mixed private/global graphs remain negative controls. No volatile
address-space contract is invented here.

The backend proof is GFX10-specific, compiler-owned, and default off. It rejects
unknown calls/instructions, recursion, indirect/unresolved calls, bundles,
dynamic/realigned frames, actual flat/aperture register use and unsupported
semantic pseudos. V4 rejects proof with IPRA enabled. Paired no-IPRA measurements qualify only
these fixtures and compiler configuration; V2/V3 failures never imply V4 success.
Even a future passing selected matrix proves neither arbitrary-client compiler
correctness nor PS5 private-memory placement, SPI input setup, cache behavior,
scratch swizzling on hardware, launch success, or runtime support.

Wrapper guard tests run with `python3 "$recipe/test_recipe.py"` and invoke no
compiler.

The semantic-binding checks run with `python3 "$recipe/test_measurement.py"`;
they reject resource, call, effect, operand, wait and configuration mutations
without invoking a compiler.
