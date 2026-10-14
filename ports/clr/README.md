# Experimental ROCclr platform port

This patch contains the actual platform changes used by the qualified native
27-unit ROCclr compile and link experiment. It does not replace ps5-hip's current
HIP runtime. Public HIP initialization and the PS5 GPU backend's native behavior
remain unqualified. The later startup overlays and their bounded
CPU-only native validation are described in `ADAPTER.md`.

Apply `ps5-platform.patch` to a clean ROCm/clr checkout at the revision in
`manifest.json`, using `git apply --check` before `git apply`. Build with the two
platform defines listed in the manifest, add this directory to the include path,
and compile `ps5-clr-tls.cpp` with the upstream units. The manifest records both
the original and adapted hash of every changed file and both helper hashes.
The patch was applied to copies of the pinned upstream files and every resulting
file matched the tested native source byte for byte.

The changes provide pthread-key thread/correlation storage, BSD affinity and
stack queries, checked native OS initialization, SDK-compatible environment/PID
queries, and explicit rejection of unsupported graphics sharing. Apply the
startup overlays in `adapter-manifest.json` as documented in `ADAPTER.md` before
native application use: the base patch alone retains two failed startup paths.
Linux remains
available for the actual upstream queue/command regression workload. Native
SIGFPE instruction repair is unsupported; physical-memory queries report zero
when the OS query fails. The TLS helper aborts on infrastructure/allocation
failure because the upstream pointer/reference accessors cannot return errors.

Executable CPU memory is explicitly unsupported in this profile. Reserve,
commit and protection requests for `MEM_PROT_RWX` return null/false before any
mapping or protection call, including in Linux qualification. The v7 test checks
zero calls and preservation of existing bytes, then exercises normal
NONE/READ/RW, uncommit/recommit and release operations. Both upstream CPU workload
modes pass. The changed native OS object was recompiled and linked with the 27
unchanged, hash-verified native objects; no native execution is claimed. A future
executable-memory route needs an owned, qualified native provider.

The port currently covers an internal ROCclr compute profile. Public HIP
initialization and complete linkage need additional work; the memory prerequisite
and profiling TLS writer overlay are documented in `ADAPTER.md`. Inherited process,
`/proc` and dynamic-loader services are not qualified. The native shared-ELF
link uses the SDK's C++ runtime and actual unwind-section bounds; optional weak
SDK loader hooks need explicit treatment before application conversion/startup.

Evidence is summarized in `../../evidence/offline-upstream-clr-native-port.json`.
The current guard qualification is in `../../evidence/offline-clr-rwx-guard.json`.
The exact source hashes and qualification receipts are in `manifest.json`.
The earlier evidence remains a historical v6 checkpoint.
No asynchronous HIP API, graphics interoperability, native runtime behavior,
framework support or performance result is implied by compiling these files.
