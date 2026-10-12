# ShadowMountPlus 1.6beta16: exact source policy

The public tag resolves directly to commit `8566c0294cbf37b55375602e950a0e6b6bb928d7`. GitHub records its commit at 2026-06-28T13:53:42Z and release at 2026-06-28T13:59:58Z. The commit archive is 162633 bytes, SHA-256 `29d332edb88f7bfaa7001e66f70746e1285a535baa5a6987b5c3cc57f2384329`. Raw API responses and the complete extracted source are retained beside this report; no existing checkout was modified.

## Concrete lifecycle

**A normal game exit does not request release of the registered image or title nullfs mapping in this tag.** This is affirmative source behavior, not merely a missing log-string search:

- `src/sm_game_lifecycle.c:423-439` handles exit by consuming active title/PID tracking, clearing pending launch state, notifying fakelib and Kstuff, and normalizing snd0info. `540-556` invokes this path for NOTE_EXIT. There is no image-unmount or title-nullfs release on this path.
- `src/sm_kstuff.c:791-802` emits the exact `[KSTUFF] game stopped: <title> pid=<pid>` marker, clears its tracking and restores Kstuff as needed. That marker proves the component processed a tracked exit; it is not a mount-release marker.
- `src/sm_fakelib.c:294-304` cleans matching tracked sandbox library overlays when present. Its helper at `53-63` logs success for successful unmount **or ENOENT/EINVAL**, and a failure otherwise. An absent fakelib log is not a missing cleanup guarantee when no overlay was created. This mechanism does not itself enumerate or destroy the entire OS sandbox.
- `src/sm_scan.c:241-273` prefers valid installed/tracked sources, returns early for already mounted content, and recreates missing title nullfs mappings. `src/sm_filesystem.c:858-965` creates/reuses `/system_ex/app/<title>` nullfs from the source. Consequently this mapping belongs to persistent title registration here; its persistence is not by itself evidence of a per-process resource leak.
- `src/sm_image.c:575-599` keeps an active, accessible cached image unchanged. `1155-1198` only unmounts a missing source or repairs a lost/unreadable mount during stale cleanup. `src/sm_scan.c:715-730` runs that lost-source cleanup, not per-exit release.
- `src/sm_filesystem.c:976-1093` removes title links/controlled stacks only when its source-removal or stale-source predicates require it; valid sources return at `1026-1027`. `src/sm_image.c:1068-1149` does have explicit image-unmount logs, but this operation is reached for cleanup/recovery, USB suspend (`1250-1275`) or payload shutdown (`1353-1380`), not ordinary game exit. `src/main.c:516-527` performs title/image shutdown when the loader itself stops.

Direct pinned source links:

- [Game exit path](https://github.com/drakmor/ShadowMountPlus/blob/8566c0294cbf37b55375602e950a0e6b6bb928d7/src/sm_game_lifecycle.c#L423)
- [Persistent registered title scan](https://github.com/drakmor/ShadowMountPlus/blob/8566c0294cbf37b55375602e950a0e6b6bb928d7/src/sm_scan.c#L241)
- [Title nullfs mapping](https://github.com/drakmor/ShadowMountPlus/blob/8566c0294cbf37b55375602e950a0e6b6bb928d7/src/sm_filesystem.c#L858)
- [Cached image retention/repair](https://github.com/drakmor/ShadowMountPlus/blob/8566c0294cbf37b55375602e950a0e6b6bb928d7/src/sm_image.c#L575)
- [Exact release](https://github.com/drakmor/ShadowMountPlus/releases/tag/1.6beta16)

The later [1.7beta1 release](https://github.com/drakmor/ShadowMountPlus/releases/tag/1.7beta1), published 2026-09-20T20:41:12Z, explicitly introduces default on-demand image preparation/release after exit, and a disabled-by-default persistent-image option. This corroborates a policy change; those later lifecycle guarantees and log expectations must not be projected backward onto 1.6beta16. Only release notes were inspected for this comparison, not the complete 1.7 implementation.

## Identity and date limits

`Makefile:4,8` derives the version label using `git describe --abbrev=6 --dirty --always --tags` and a compiler define. `src/main.c:459-466` logs that string plus compiler `__DATE__`/`__TIME__` and queried firmware. A matching v1.6beta16 banner and June build date are consistent with this public tag; they do not establish that the deployed ELF is this release, unmodified, or built from these exact sources.

The release API reports asset `shadowmountplus.elf`, 1457528 bytes, SHA-256 `a35246fb3bb6042b25653b51cdcbc33254b40339342bf1d2dd0d2eceee2ca526`; this is upstream-published asset metadata, not a locally downloaded/rehash-verified binary and not a deployed-binary comparison. No source-to-binary reproducibility was attempted.

`src/sm_image.c:300-319` forms image mount names using stripped filename plus 32-bit FNV-1a of the **source path**, not image content. The observed `/mnt/shadowmnt/PPSA99014_b6d9b2e4` is therefore a path identity, not a cryptographic assertion of which UFS bytes are mounted. Retain the existing deployment image hashes and source/link checks independently.

## Implication for ps5-hip acceptance and the snapshot

The declared requirement that every registered image and `/system_ex/app` mapping disappear after each title exit conflicts with this tag's normal persistent-registration behavior. A retained registration image alone is neither a leak demonstration nor proof that teardown is healthy. This source finding does not retroactively turn the .40 arithmetic partial-pass into full acceptance, close G2, prove the deployed binary, or authorize changing loader settings/forcing unmounts.

Separate (1) per-run process and sandbox/overlay cleanup, (2) deliberately persistent loader-owned registration mounts and (3) unchanged service health. An explicit environment-specific acceptance clarification may require baseline-equivalent persistent mounts with no new duplicates or changed identity, plus real process/sandbox teardown. Preserve the original criterion/result and explain any revised prospective criterion; do not silently waive it merely to close a gate. Fresh before/after baseline observations would be stronger than a single post-run snapshot.

The prepared read-only snapshot can help identify which visible rows remain, including exact target/source, fsid, filesystem type and flags, and whether two consecutive observations are stable. It can distinguish the known registration image, the title mapping, and visible sandbox rows. Its current `runtime_layer` classification means the `/system_ex/app` path category only: this source shows that path can be persistent registration, not necessarily runtime-only ownership. Two same snapshots are not proof of no transition, and old/new snapshots can only compare fields actually observed.

The snapshot still cannot certify global namespace completeness, the deployed statfs ABI, detached descendants, references invisible to its namespace, or absence of all possible alternate image slots. Missing rows are weaker evidence than positive rows until visibility is qualified. Its `release=unknown accepted=false` and exit2 are correct. It should stay an evidence-collection diagnostic, not become a release-pass oracle based on an empty or stable table. No console access, mount changes, updates, source-code changes or builds were performed in this investigation.
