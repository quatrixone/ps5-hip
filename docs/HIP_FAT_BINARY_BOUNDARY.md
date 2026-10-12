# Bounded HIP fat-binary consumption and the next registration boundary

Offline design note, 2026-10-12. This records the narrow G4 implementation
boundary, not a native runtime, a completed G4 gate or GPU qualification.
See [the client inventory](HIP_CLIENT_REQUIREMENTS.md),
[the measured host ABI](HIP_HOST_ABI_SPIKE.md) and
[implementation-plan section 3.2](IMPLEMENTATION_PLAN.md#32-runtime-implementation-decision).

## Current reader

`include/ps5hip/fat_binary.h` declares
`read_fat_binary(const uint8_t *data, size_t size, const std::string &target)`.
The caller supplies an actual readable span and keeps it immutable during the
call. `src/fat_binary.cpp` accepts only the uncompressed Clang offload container,
at most 64 MiB and 64 directory entries, with bounded printable target IDs.
Selection is exact: `hipv4-amdgcn-amd-amdhsa--gfx1013` or
`hipv4-amdgcn-amd-amdhsa--gfx1030`. These are compiler target identities, not a
claim that either entire instruction set works on the PS5.

The reader checks every directory entry, payload extent, duplicate target and
nonempty payload overlap before returning. The selected payload must be nonempty.
It copies that payload into `BundledCodeObject::bytes`, invokes the existing
`inspect_code_object` on the owned copy, and requires the ELF metadata target to
agree with the selected directory target. Callers can then use the existing
`materialize_code_object` on those same bytes; reading a bundle does not itself
allocate GPU memory or materialize an executable GPU mapping.

Compressed `CCOB` containers, alternative target spellings, device-link formats
and newer compiler containers are outside this contract. In particular, the
Clang 24/HIP 7 fixture uses `hip-amdgcn-amd-amdhsa--gfx1013` and different ELF
target/descriptor information; it must not become accepted by relaxing a prefix
comparison. Empty host placeholders in the selected LLVM 18 fixtures do not
establish host binary compatibility.

There are no HIP registration exports in this unit. The actual input controls
are the 12 saved LLVM 18/ROCm 6.2.4 bundle sections: Linux and PS5 host emission,
both device targets, and launch/vector-add/dimensions fixtures under
`results/hip-host-abi-spike/rocm-6.2.4/`. Parser test results belong in their
validation receipt, not in this design note.

The existing inspector accepts an extracted bundle section directly:

```sh
ps5hip-inspect --bundle gfx1013 vector.bundle
```

The one-argument ELF inspection mode remains available. Both modes report
`hardware_validated=false`. For the real compiler-object tests, configure CMake
with `-DPS5HIP_HIP_ABI_FIXTURE_DIR=/path/to/rocm-6.2.4`. The fixture test verifies
the committed manifest and exact host/device hashes, extracts the unchanged
sections, and exercises both the reader and inspector. If the saved default
native ELF is a historical failed import control, set `PS5HIP_NATIVE_AGC_ELF` to
the verified provider-correct control explicitly; preserve the failed image.

## Why the compiler wrapper cannot supply its own bound

The pinned HIP revision is `d3b75fbb82113374e2d6103baa96a9c61425da48` and
CLR revision is `65d174c3e35423bf25c9c369513780c9d9ca760c`, from ROCm 6.2.4.
Their archives, generated version and compiler inputs are bound by
`evidence/offline-compatible-hip-headers.json`. All CLR paths below are relative
to that revision.

`hipamd/src/hip_platform.cpp:70-77` implements
`void **__hipRegisterFatBinary(const void *)`. It reads a wrapper containing
magic/version and a binary pointer; the ABI supplies **no readable length**.
The measured wrapper is 24 bytes, magic `0x48495046`, version 1, with its binary
pointer at offset 8 and a reserved word at offset 16. The native combined-object
control has an allocated, 4096-aligned `.hip_fatbin` section and a writable,
8-aligned `.hipFatBinSegment` containing a relocation for that pointer. For
example, the saved gfx1013 PS5 vector object's section sizes are 8136 and 24
bytes respectively, recorded in its `*-host-readobj.txt`.

The upstream parser is a useful format reference, not a drop-in bounded reader:
`hipamd/src/hip_code_object.cpp:528-569` walks cast directory pointers without
an input size and may stop when selected binaries have been found.
`getFatbinSize` at lines 603-617 derives a length from directory contents. That
derived number cannot prove the referenced memory is readable. Its file/memory
entry points at lines 496-524 do not give the parser an authoritative span.
Neither scanning for magic nor trusting an offset/size from the container solves
this problem. No claim is made here that every upstream parsing implementation
has been inventoried.

## Smallest complete linked-title binding

Keep the first registration control to one ordinary HIP translation unit, one
wrapper and one bundle. Establish bounds at link time before exporting the
compiler ABI:

1. Preserve `.hip_fatbin` with `KEEP`, its required alignment and hidden start/end
   symbols in the read-only image. Preserve `.hipFatBinSegment` with separate
   hidden bounds inside `.data.rel.ro` or an output section named
   `.data.rel.ro.hip_wrappers`.
2. Require the passed wrapper address to match the bounded single-wrapper slot
   before reading its 24 bytes. Validate magic/version/reserved fields. Require
   its relocated binary pointer to equal the start of the bounded single bundle;
   only then call `read_fat_binary` with that exact linker span.
3. Keep constructor-time registration metadata-only. Own the copied bytes and
   map the compiler's host function token to the registered device symbol.
   `hip_platform.cpp:80-105` confirms that host identity and device name are
   distinct inputs. Do not treat the host token as a GPU entry address.
4. Exercise actual compiler constructors and unregister using a linked Linux
   control, then verify native PIE and converted executable sections, relocations,
   load extents and startup arrays offline. A linked control must fail explicitly
   if it attempts an unimplemented launch; it must not use success-returning HIP
   stubs to simulate runtime completion.

The native-tool source identity is
`3f3ee69607013b345d2baa6d6a37c86745649a08`, exported under the matching
`.deps/native-tools-*` directory. Its `ps5-pie.ld:33-38` currently has ordinary
read-only/RELRO placement but no explicit HIP bounds. Lines 47-51 already keep
`.ctors`/`.ctors.*` in the initialization array; `app_crt.cpp:61-64` executes
that array. Constructor support therefore has an existing integration path.

The converter matters: `sce_module_writer.cpp:172-176` classifies RELRO by
output-section name, including `.data.rel.ro*`, GOT and initialization arrays.
A standalone section named `.hipFatBinSegment` with only a `:relro` linker
assignment is insufficient to establish that the converter will retain its
intended protection. The converter inspects allocated sections at lines 613-655
and copies their stored bytes at lines 884-891. Verify the converted result;
an intermediate link map alone is not native-load evidence. Hidden boundary
symbols should resolve through ordinary linked references, without depending on
runtime section headers or dynamic symbol lookup.

Reject wrong wrapper slots, partial wrappers, bad magic/version, relocated
pointers outside the bundle or into its interior, missing/duplicated controls,
unsupported directory/ELF targets and all malformed reader inputs. Test pointer
arithmetic overflow and boundary endpoints. For multiple translation units,
introduce exact per-input spans through a generated/linker catalog first:
aggregate section bounds prove readability but do not prevent one bundle's
directory from referring into a neighboring bundle. Do not silently broaden the
single-TU registration control into general module loading.

## Launch and lifetime obligations after registration

The measured ordinary stubs require `__hipPushCallConfiguration`,
`__hipPopCallConfiguration`, `hipLaunchKernel`, registration/unregistration and
`atexit`. Match the pinned header's actual C ABI, including by-value `dim3`;
do not reproduce its layout from an assumed declaration. Push/pop need nested,
thread-local state and deterministic underflow handling. The measured launch stub
discards the `hipLaunchKernel` return, making recorded last-error behavior
necessary rather than optional.

Copy argument values from the stub's `void **` array before it returns. Recover
pointer extents and owners from a real allocation registry, including checked
interior pointers, rather than inventing sizes. The existing initial launch
route can pack the measured 13 hidden geometry fields into 288-byte kernarg
storage for its uniform 64x1x1 profile. Other shapes, nondefault streams and
dynamic LDS remain errors until their connected runtime semantics exist.

Module/code, kernarg and allocation ownership must survive submission and
uncertain completion. Reuse `SubmissionQueue` retention and poison rules; unload
or unregister must never release backing storage held by a failed or timed-out
submission. Upstream `hip_platform.cpp:177-185` synchronizes before unregister,
which is a useful lifetime obligation, not permission to pretend synchronization
succeeded. Constructor order, partial failures, duplicate registration, stale
handles, repeated initialization and thread/error isolation need focused tests.

## Architecture decision remains bounded

The reader and proposed registration control fit a small compatibility runtime
over the existing native substrate. This is a feasible next experiment, not a
measured claim that it is the best production architecture. Before broad G4 or
ggml expansion, retain the section 3.2 comparison with a PS5 backend beneath
upstream HIP/CLR. Pinned Prism `e8fd209c975c80aa7fb36a8423571ff3b1b7e955`
already requires streams/events, hipBLAS/rocBLAS and much wider device semantics;
the reader does not remove those costs. Its API inventory does not itself require
a ROCr/HSA queue adaptation. Do not build three production runtimes or declare
G4 complete after this unit. Native normal-launch execution, the reduction and
tiled-matmul examples, HIPIFY and the plan's lifetime/thread tests remain required.
