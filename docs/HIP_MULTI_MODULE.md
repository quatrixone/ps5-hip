# Bounded compiler module registration

The reusable contract admits at most 32 lifetime module handles, 64 MiB of
aggregate admitted bundle bytes and 256 active function bindings. Retired
handles remain allocated until registry destruction. Existing fixed diagnostic
contracts and links without a module table retain single-module admission.

Multi-module links supply `__ps5hip_module_spans_begin/end`, an immutable table
of 32-byte entries. Each entry holds four little-endian 64-bit offsets: wrapper
begin/end relative to `__ps5hip_wrappers_begin`, then bundle begin/end relative
to `__ps5hip_bundle_begin`. Their aggregate end symbols bound both regions.
Offsets avoid relying on dynamic relocations for linker-generated `QUAD` data
in PIE executables. Each wrapper must be exactly 24 bytes, each bundle has an
independent exact extent, and every pair of module extents must be disjoint.
The real wrapper's relocated pointer must name its own exact bundle start.
These extents are trusted linker inputs, not a mapping-discovery API.

Host tokens are unique across active modules. Device symbol names are scoped
to their owner. Launch lookup returns the kernel and owning module together
under the existing runtime gate; code materialization uses that owner. Each
module must bind at least one function before launch readiness. Registration
closes when ordinary runtime work begins. Dynamic loading and device linking
are outside this contract.

A failed registration blocks dispatch and unwinds safe metadata from every
admitted module. The fixed handle journal remains available to recognize each
genuine compiler destructor once. Normal retirement removes only that module's
bindings. Safe State destruction still requires normal finish, all callbacks,
no live storage or worker contexts, no pending configurations, and no terminal,
partial or synchronization fault. Missing or failed genuine retirement retains
State. Extra invalid or duplicate callbacks cannot retire another module and do
not prevent complete closure once every genuine callback succeeds.
Unscoped diagnostic handle/token access returns null when multiple modules
make it ambiguous; name lookup rejects duplicate names across modules.

`tools/probe_hip_multi.py` compiles two genuine HIP translation units for Linux
and PS5 hosts. Its ASan/UBSan host model links both object orders and exercises
A/B/A launches. The synthetic transport verifies the complete materialized
code image and entry address against the expected module before computing
from actual input backing. This proves host routing and ownership, not GPU
execution. No native package or console validation is part of this change.
