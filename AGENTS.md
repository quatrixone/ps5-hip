# Working on ps5-hip

Read `STATE.md`, then the active goal in `docs/IMPLEMENTATION_PLAN.md`. Read
`docs/RESEARCH_BASELINE.md` for a relevant hardware or upstream claim. Do not load
the entire workspace, dependency trees or historical logs by default.

## Project boundaries

- This is a separate compute SDK. Keep ProsperoAI UI/model work and graphics-driver
  changes in their own repositories; consume pinned revisions here.
- The plan is ambitious. Do not declare the project complete after an early
  canary, rename a failed gate as success, or reduce acceptance silently.
- Implement the smallest complete step that advances the active gate. Planned
  directories and APIs are not instructions to create empty scaffolding.
- Preserve user changes. Keep the supported build and its source on `main`;
  temporary branches must be integrated before handing off a build. Do not
  publish the repository, create releases or change repository access unasked.
- The user's instructions and current authorization take precedence over this file.

## Correctness and evidence

- Distinguish source compatibility, GPU code-object compatibility, host binary
  compatibility and framework compatibility. Never imply one proves the others.
- Keep hardware observations separate from compiler assumptions and BC-250 reports.
- Reject unsupported metadata, operations and capabilities explicitly. Never return
  success from an unimplemented HIP function or silently advertise unproven features.
- Before each hardware experiment record evidence, hypothesis, expected outcome,
  rejection criterion, timeout and next decision. Inspect the generated ISA first.
- Commit and freeze the candidate; record source/toolchain IDs and artifact hashes.
  Compare exact operands for integer tests and declared tolerances for floats.
- After two inconclusive experiments on the same hypothesis, reassess it offline.
- Update current state and one concise milestone entry after a completed gate.
  Preserve raw logs locally and commit compact, sanitized evidence with provenance.

## Console work

The planning request did not request a console run. When later work is authorized,
follow the current `ps5-homebrew-dev-protocol` runbook and ignored
`.local/ENVIRONMENT.md`. In this workspace the runbook is
`../../docs/ps5-homebrew-dev-protocol/docs/RUNBOOK.md`.

Use WSL for native builds and console traffic. Coordinate the current shared-console
lock; do not import an old project's lock waiver or assume its title ID is available.
Use a configured development title. Freeze, deploy, verify, run for a bounded time,
close that exact title, check service health and release only the owned lock.
Never automatically repeat a suspected panic, alter system settings or firmware,
overwrite the production ProsperoAI installation, or delete model data and saves.
