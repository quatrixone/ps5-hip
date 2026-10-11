# Console workflow preparation

Status: offline only. `PPSA99012` is the user-selected title. No console has been
contacted, and no controller has been sent. The shared-console runbook in AGENTS.md
remains authoritative for locking, declared services, bounded runs and teardown.

## Frozen controller builds

```sh
python3 tools/build_controllers.py --protocol-source /path/to/ps5-homebrew-dev-protocol
```

`controller-inputs.json` pins the shared protocol at
`ed775bdf67ec3861f2e6c78360a6ec58508f2c22`. The build exports just the committed
launch/close sources and license into ignored dependencies, then compiles them with
the same pinned SDK and LLVM 18 used for the title. It records source and output
hashes under `build/controllers/PPSA99012/receipt.json`. There is no network or
send operation in this tool. These ELF payloads are controllers, not title eboots.

The launch controller initializes the user service, obtains the foreground user,
requests exactly `PPSA99012`, reports the native result and terminates the service.
The close controller compares the running application's title before closing it;
its fallback obtains an application ID specifically for `PPSA99012`. It never
closes a guessed process ID. Host mocks compile the actual exported sources and
exercise failure paths and an unrelated running title. This validates controller
logic, not the console service ABI or successful native launch/close behavior.

A controller transfer succeeding is not proof that its request succeeded. The
run must capture its result and title-specific lifecycle evidence independently.

## Captured application results

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

## Remaining host control work

The existing generic protocol runner is not called automatically. It includes
graphical-stream management and other assumptions that this compute title does
not require. The Vulkan runner fetches loose files from its own title directory;
that does not establish how this title's `/download0` log is exposed by the
selected console environment.

Prepare and test a single-case host workflow that:

1. Verifies the frozen package/controller receipts locally before opening sockets.
2. Uses the declared environment's lock, endpoints and title lifecycle controls.
3. Uploads the image under a temporary name, verifies bytes, promotes it and waits
   for title-specific registration. Existing installation state must be explicit.
4. Preserves any earlier result, establishes log freshness, captures klog before
   launch, and sends only the frozen launch controller.
5. Observes for the declared deadline, handles failure/quarantine without retry,
   closes only `PPSA99012` when needed and verifies teardown/service health.
6. Retrieves the exact title log or its download-data image, verifies extraction,
   runs the application checker and records the independent lifecycle evidence.
7. Releases only its own lock, including on errors.

Firmware, host, approved service endpoints, lock path and log retrieval mapping
have not yet been configured for this project. Do not inherit another title's
settings or infer a stale lock. Native return-from-main behavior also remains a
CPU-control experiment: another runner's comment about exit signals is not proof
of normal completion here. A crash after PASS must remain a failed teardown.
