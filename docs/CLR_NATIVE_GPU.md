# Native ROCclr GPU candidate

This candidate preserves the actual upstream two-queue write/copy/kernel/read chain and uses AgcBackend. It composes the portable platform, lifetime, AOT, affinity and executable-path patches from the integrated root. All 46 native units are freshly compiled; historical build receipts provide pinned compiler/startup inputs, never reused executable objects.

`prepare_clr_gpu_source.py` applies the exact patch stack. `qualify_clr_gpu_adapter.py` rebuilds 39 actual upstream/adapter/core units for 14 ASan/UBSan and TSan cases. `probe_clr_gpu_client.py` reuses only those bound objects to test the actual client in 11 modes, including a real failed pthread_join at teardown. Synthetic arithmetic exists only in model transport and cannot enter the native title. Failed client operations quarantine before manual-owner destruction. The watchdog is the same checked CPU-title watchdog with namespace/error-name substitution.

`build_clr_gpu_title.py`, `convert_clr_gpu_title.py` and `audit_clr_gpu_title.py` require strict native linking, actual import providers, genuine startup/finalizer arrays, absent optional hooks, and exact read-only genuine HSACO bytes. Package creation requires a clean source freeze and independent candidate review, followed by UFS fsck and complete extracted-file equality.

Acceptance expects one dispatch, 73 exact scalar/index outputs, 81 readback words and eight input guards, one backend initialization, two buffer frees and two positively joined queues before device/context destruction. The copied input is observed through the guards; active arithmetic does not consume it. CPU copies and one serialized physical GPU queue are not asynchronous DMA or concurrent GPU execution.

This is not public HIP admission or G5 acceptance. Console execution is a separate one-use milestone. The public HIP checkpoint and reset rejection are not linked into this candidate.
