# Console workflow preparation

Status: the PPSA99014 CPU shell-managed lifecycle passed one run on .30, firmware
6.02, including runtime checks, application finalization, self-stop and service
health. See `evidence/native-shell-exit.json`. Ordinary C exit still faults after
main returns; do not repeat those diagnostics. The first GPU candidate failed with
CPU SIGSEGV/RIP zero after the pre-submit checkpoint; GPU execution is unproven.
See `evidence/native-indexed-store-first.json`; do not rerun the failed case. The .40
loader-failure process had unresolved teardown; establish recovery before reuse.
The shared-console runbook in AGENTS.md
remains authoritative for locking, declared services, bounded runs and teardown.

## Frozen controller builds

```sh
python3 tools/build_controllers.py --protocol-source /path/to/ps5-homebrew-dev-protocol
```

`controller-inputs.json` pins the shared protocol at
`ed775bdf67ec3861f2e6c78360a6ec58508f2c22`. The build exports just the committed
launch/close sources and license into ignored dependencies, then compiles them with
the same pinned SDK and LLVM 18 used for the title. It records source and output
hashes under `build/controllers/<title-id>/receipt.json`. There is no network or
send operation in this tool. These ELF payloads are controllers, not title eboots.

The launch controller initializes the user service, obtains the foreground user,
requests exactly the configured title, reports the native result and terminates the service.
The close controller compares the running application's title before closing it;
its fallback obtains an application ID specifically for that title. It never
closes a guessed process ID. Host mocks compile the actual exported sources and
exercise failure paths and an unrelated running title. This validates controller
logic, not the console service ABI or successful native launch/close behavior.

A controller transfer succeeding is not proof that its request succeeded. The
run must capture its result and title-specific lifecycle evidence independently.

## Captured application results

The harness now uses the Eden client and a bundled, exact-title Lapy owned-root
helper. It requires successful filesystem elevation; it does not fall back to
`/download0`. Stage with `--results-dir /chosen/results` to write that absolute
directory into `results-dir.txt`. No console path is hardcoded in the executable.
The chosen directory must exist and be writable before launch. The harness uses
exclusive creation of its exact result filename and never deletes an earlier log.
FTP retrieves this ordinary file even after title exit; no download-data image
extraction is required. GPU objects are opened before elevation changes the root.

Build the helper offline before building a native candidate:

```sh
python3 tools/prepare_elevation.py --eden-source /path/to/eden --lapy-source /path/to/lapy --logging-header /path/to/pinned/ps5log.h
```

`elevation-inputs.json` pins the Eden client and Lapy source separately. The older
Eden helper is not reused: Lapy's owned-root helper supplies the same wire protocol
with runtime layout and title checks. The exported protocol definitions are tested
together. The CPU lifecycle experiment exercised the combined integration on .30;
other environments remain unqualified. The Lapy
source/license and client copyright/SPDX notices remain in the ignored export;
the client is GPL-3.0-or-later. The exact logging header is identified as
GPL-3.0-or-later by its upstream NOTICE; see `docs/DEPENDENCY_PROVENANCE.md` and
`evidence/offline-logging-provenance.json`. Binary-distribution accounting remains
separate work before any release.

Before console traffic, verify a frozen case against committed evidence:

```sh
python3 tools/verify_native_case.py --stage dist/PPSA99014/<candidate>
```

This rejects changed images, controllers, receipts, title identity and staged
files. It does not prove console readiness or execution.

```sh
python3 tools/check_canary_log.py --log results/<run>/application.log --target gfx1013 --kernel runtime
```

This checker reads at most 64 KiB and accepts only the exact complete record
sequence emitted by the selected harness. Duplicate starts, wrong target/kernel,
truncation, unexpected cleanup output and wrong value counts cannot pass. An
explicit failure or quarantine overrides any PASS record in the same input.

`application-pass` means only that the captured bytes satisfy this application
contract. The checker always reports `hardware_gate_passed: false`: freshness,
remote artifact identity, process status, teardown and service health remain
unproven by a log file alone. Its zero exit status must not be used as the full
hardware gate decision.

## Single-case host control contract

The existing generic protocol runner is not called automatically. It includes
graphical-stream management and other assumptions that this compute title does
not require. Result collection uses the exact configured log path from the
candidate manifest.

The CPU lifecycle experiment used a single-case host workflow. Preserve this
contract for each subsequent frozen candidate:

1. Verifies the frozen package/controller receipts locally before opening sockets.
2. Uses the declared environment's lock, endpoints and title lifecycle controls.
3. Uploads the image under a temporary name, verifies bytes, promotes it and waits
   for title-specific registration. Existing installation state must be explicit.
4. Preserves any earlier result, establishes log freshness, captures klog before
   launch, and sends only the frozen launch controller.
5. Observes for the declared deadline, handles failure/quarantine without retry,
   closes only the configured title when needed and verifies teardown/service health.
6. Retrieves the exact configured title log,
   runs the application checker and records the independent lifecycle evidence.
7. Releases only its own lock, including on errors.

The authorized hosts and normal service endpoints are recorded in the ignored
environment file. Recheck current idle state and environment at each console
milestone; the earlier observation of FW 6.02 on .30 is historical evidence.
Eden's rules
also name a shared lock in the adjacent PS5-Claude workspace; coordinate it before
further console work rather than relying only on the protocol helper's default.
Do not infer a stale lock. Use the explicit `shell-request-probe` lifecycle for the
next finite GPU experiment. Its CPU pass does not qualify GPU cleanup, worker
threads or general C exit. A crash after PASS remains a failed teardown.
