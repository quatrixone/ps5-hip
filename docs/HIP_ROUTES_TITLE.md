# Four-kernel HIP title

The `hip_routes` profile packages one genuine HIP translation unit with integer,
static LDS, compiler-spill scratch and 2D geometry kernels. It uses the shared
serialized runtime, three persistent 64 KiB device allocations, and two
changed-input sequences of four blocking launches. The client compares three
complete 16,384-word pages after every launch: 393,216 word comparisons including
inputs, unused tails and guards. This is not a count of active shader outputs.

The compiler recipe uses `--compiler-only --title-entry` to rename `main` to
`ps5hip_routes_client_main` and enable the title-specific success record. These
objects have their own hashes; the bare-main qualification is not reused as the
title object. The fixed profile in `tools/native_vector_title.py` binds the
compiler receipt, combined object, embedded bundle and device ELF.

Build after committing the candidate sources:

```sh
python3 -B tools/build_native_vector_title.py --kernel hip_routes \
  --reference-root /path/to/reference/ps5-hip \
  --routes-probe results/hip-routes-title-compiler-v1
python3 -B tools/stage_native_title.py --target gfx1013 --kernel hip_routes \
  --results-dir /data/ps5-hip/results
```

The existing package tool then consumes the staged folder. Staging requires a
clean frozen build, validates the embedded code and signed executable, and saves
ten provenance files beside the application. No external `canary.hsaco` is used.
The title remains `PPSA99014`; its result is `ps5-hip-gfx1013-hip_routes.log` in the
configured directory. Eden's bundled helper supplies filesystem access.

Startup installs the shell observer and title finalizer before compiler
constructors. After log redirection, the title checks one module and four
registered functions. Successful computation is only an intermediate result:
the genuine compiler unregister and complete runtime State destruction must
precede the final persisted record and shell exit request. Uncertain submission,
retained backing, partial registration, failed unregister or log persistence
failure prevents a successful finalization claim.

`tests/hip_routes_title_test.py` exercises the actual compiler lifecycle with host
service doubles. It checks failure handling and ownership, not GPU arithmetic.
The original vector title remains a separate compatibility control. A console
qualification additionally requires the frozen package, verified deployment,
fresh result, exact process lifecycle, post-run service health and owned-lock
release. Offline receipts alone do not establish these hardware conditions.
