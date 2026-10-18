# Unfinished dense native package work

These patches preserve local work for a source-only handoff. They are drafts,
not validated implementation or build inputs. Do not apply both patches.

- `title-reviewed-draft.patch`: earlier title and four focused host test groups;
  previously reviewed/tested, but no native compilation or image audit. Its
  inline main branch needs a dedicated audit boundary before integration.
- `title-audit-incomplete.patch`: later in-progress title/auditor/test snapshot,
  captured at interruption. No passing audit or package claimed. Review and
  complete it rather than assuming its checks are sufficient. Relative to main
  at helper commit 5741e1d plus the app-builder changes in this handoff commit.

An image must prove the dense entry reaches the actual helper/backend/HIP calls,
excludes model loading on that path, and accepts transport completion only after
checked fsync/close. Symbol presence or textual call ordering is insufficient.
Preserve existing phase audits. The builder requires
`actual_dense_call_chain_audited=True` and must continue rejecting until justified.

Freezer/controller dense support has not been integrated; no corresponding draft
was present at snapshot. See the top of HANDOFF.md for next steps, reusable local
inputs, hardware limitations and existing evidence. No console was contacted.
