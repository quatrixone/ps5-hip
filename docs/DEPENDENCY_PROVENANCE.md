# Dependency provenance review

Status: source-notice inventory for the current private experiments. This is not
a completed binary-distribution audit or a project-wide license selection.
Exact revisions are recorded in `dependencies.json` and `elevation-inputs.json`.

| Input | Observed notice/source | Use here |
| --- | --- | --- |
| Public PS5 SDK fork | Root LICENSE; native-tool NOTICE identifies headers/platform layer as GPL-3.0-or-later | Native headers, static libraries and link stubs |
| Native title converter/startup | Exported LICENSE and NOTICE.md at pinned PS5_Vulkan revision: GPL-3.0-or-later | Host conversion/signing tools and title startup |
| SDK C++ runtime | Native-tool NOTICE identifies libc++, libc++abi, libunwind and builtins as Apache-2.0 WITH LLVM-exception | Statically linked runtime; archive-member accounting still needed for distribution |
| Eden elevation client | Source headers explicitly carry GPL-3.0-or-later and BlackBearReloaded copyright | Native application elevation client |
| Lapy helper repository | Exported root LICENSE: MIT, copyright 2026 Arksama (Team PHU) | Exact-title helper sources and locally built helper |
| ps5log header | Pinned SHA-256, but no copyright/license marker found in the copied header | Compiled into helper; originating repository/revision/license still unresolved |
| zlib | Pinned source archive, retained upstream license | Host converter dependency |
| UFS2Tool | Pinned source/tool receipt, BSD-2-Clause recorded in dependency lock | Host filesystem packaging and inspection |
| Mesa | Pinned public source references with original per-file notices | Inspected native/register behavior; not linked into this candidate |
| TheRock compiler | Exact submodule heads and upstream patch workflow recorded in build-setup evidence | Local compiler comparison in progress; no redistributed compiler bundle |

The logging header identity is
`394af67d0f8b60b3335deb53396e52855ea2daa50ca914a456ea7663f48900c6`.
Its hash identifies bytes; it does not establish licensing. The helper's MIT
license must not be assumed to grant rights over an independently supplied header.

Before a binary SDK/release milestone, resolve that header's original license,
inventory the actual linked archive members and their notices, select the project
distribution license, and assemble the corresponding source and notice material.
Private local execution and a successful compiler build do not settle these items.
No firmware implementation, proprietary SDK, model weights or credentials are
included in the committed evidence or package-source repository.
