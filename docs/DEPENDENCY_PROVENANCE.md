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
| ps5log header | Exact hash match at ps5-agc-gears `1ae1f918`; its NOTICE.md explicitly licenses native/ps5log as GPL-3.0-or-later | Compiled into helper; source and notice hashes recorded |
| zlib | Pinned source archive, retained upstream license | Host converter dependency |
| UFS2Tool | Pinned source/tool receipt, BSD-2-Clause recorded in dependency lock | Host filesystem packaging and inspection |
| Mesa | Pinned public source references with original per-file notices | Inspected native/register behavior; not linked into this candidate |
| TheRock compiler | Exact submodule heads and upstream patch workflow recorded in build-setup evidence | Local compiler comparison in progress; no redistributed compiler bundle |

The logging header identity is
`394af67d0f8b60b3335deb53396e52855ea2daa50ca914a456ea7663f48900c6`.
The matching source is `native/ps5log/ps5log.h` in
[ps5-agc-gears at 1ae1f918](https://github.com/mpereiraesaa/ps5-agc-gears/tree/1ae1f9182abd2770c131b97419034fb85173c2dc).
That commit's `NOTICE.md`, section "ps5log client", explicitly identifies the
directory as GPL-3.0-or-later. Header, LICENSE and NOTICE Git blob IDs and SHA-256
hashes are in `evidence/offline-logging-provenance.json`; the exact source/notice
copies are retained in ignored `.deps/logging-provenance`. No helper bytes changed.
The helper's MIT license does not replace the logging input's separate notice.

Before a binary SDK/release milestone, inventory the actual linked archive members
and their notices, select the project
distribution license, and assemble the corresponding source and notice material.
Private local execution and a successful compiler build do not settle these items.
No firmware implementation, proprietary SDK, model weights or credentials are
included in the committed evidence or package-source repository.
