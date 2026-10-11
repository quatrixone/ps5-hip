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
| Selected Ubuntu LLVM 18 tools | Source package llvm-toolchain-18, version 1:18.1.8-20ubuntu8; installed copyright records Apache-2.0 with LLVM exceptions by default and separate file-specific notices | Host device compiler, linker and object inspection; no compiler bundle redistributed |
| TheRock build orchestration | MIT at pinned TheRock revision; component licenses remain separate | Local compiler comparison in progress |
| TheRock LLVM component | Apache-2.0 WITH LLVM-exception default; root retains legacy NCSA and third-party boundaries; amd/device-libs has a separate NCSA notice | Pinned compiler/runtime sources; build success and generated-code compatibility are separate evidence |
| TheRock compiler support inputs | SPIR-V translator root notice: NCSA; HIPIFY: Apache-2.0 WITH LLVM-exception; half and rocm-cmake: MIT | Source-notice inventory, not a claim that every fetched project was built |

The logging header identity is
`394af67d0f8b60b3335deb53396e52855ea2daa50ca914a456ea7663f48900c6`.
The matching source is `native/ps5log/ps5log.h` in
[ps5-agc-gears at 1ae1f918](https://github.com/mpereiraesaa/ps5-agc-gears/tree/1ae1f9182abd2770c131b97419034fb85173c2dc).
That commit's `NOTICE.md`, section "ps5log client", explicitly identifies the
directory as GPL-3.0-or-later. Header, LICENSE and NOTICE Git blob IDs and SHA-256
hashes are in `evidence/offline-logging-provenance.json`; the exact source/notice
copies are retained in ignored `.deps/logging-provenance`. No helper bytes changed.
The helper's MIT license does not replace the logging input's separate notice.

## Compiler source notices

`evidence/offline-compiler-provenance.json` records the selected Ubuntu package
identities and copyright hashes for clang-18, lld-18, llvm-18, libclang-cpp18 and
libllvm18. Their installed notices match the independently extracted package root
used for the isolated reproduction. The package source identity is
`llvm-toolchain-18` version `1:18.1.8-20ubuntu8`; this does not assert an upstream
Git revision or an independently rebuilt Ubuntu compiler package. Its copyright
file records additional MIT, BSD, Python and public-domain notices for specific
paths. The receipt also hashes the referenced Apache 2.0 text.

The same evidence records exact revisions, Git blob IDs and SHA-256 hashes for
the pinned TheRock root and compiler/component notices. All 20 recorded files
were compared with their pinned Git blobs, including LLVM's project/runtime
notices and `amd/device-libs/LICENSE.TXT`. TheRock's MIT notice covers its own
orchestration; it does not replace component or embedded third-party notices.
The LLVM root explicitly preserves separate subtree/file terms and a legacy NCSA
notice. Device libraries and the SPIR-V translator have their own NCSA notices.
These distinctions matter if a future compiler or device-library bundle is shipped.

This closes the narrow compiler source-notice inventory for the selected route
and comparison inputs. It does not inventory the complete compiler dependency
graph or determine the obligations of a future binary distribution. It does not
establish that the ongoing TheRock build or compiler comparison has passed.

Before a binary SDK/release milestone, inventory the actual linked archive members
and their notices, select the project
distribution license, and assemble the corresponding source and notice material.
Private local execution and a successful compiler build do not settle these items.
No firmware implementation, proprietary SDK, model weights or credentials are
included in the committed evidence or package-source repository.
