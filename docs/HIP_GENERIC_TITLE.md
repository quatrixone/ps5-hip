# Bounded reusable HIP title

This title uses the reusable structural launch policy with one genuine compiler
module and two functions. In one process it runs consecutive sequence lengths
0, 1, 3 and 7, keeping registration and the lazily created backend alive until
the genuine compiler destructor. The zero sequence observes prebackend state;
it is not a separate zero-work finalization test. Eleven completed submissions
are observed from runtime state, rather than inferred from a success message.

Each nonzero sequence changes input seeds and 64-bit scalar salt, varies logical
allocation sizes and interior offsets, and checks exact affine/reduction results,
all logical prefix/tail guards and immutable inputs. The existing synthetic
transport independently decodes actual packets and computes reference values.
It is not GPU ISA emulation. Read-only observations do not add an expected count
to reusable runtime admission. Application checks and safe ownership finalization
are separate requirements.

The compiler recipe must explicitly rename the real HIP `main` to
`ps5hip_generic_client_main`. The title links the structural policy, never a fixed
vector/routes adapter. The stage binds the exact combined object, bundle, code,
compiler receipt and `[0,1,3,7]` workload. Pre-init finalizer installation, Eden
elevation, exclusive result log, actual unregister, post-State-destruction POD
proof, persisted final log and shell-managed exit reuse the existing title path.
Any application failure, partial registration, unresolved ownership, skipped
unregister or failed final persistence prevents a successful shell exit path.

Host models include clean failures, returned/exceptional submissions, timeout,
retained backing, late registration, destructor and log failures. They do not
qualify hardware. Custom ASan mappings use explicit guards without ordinary heap
redzones. Native linking/package inspection and a later separately authorized
hardware cycle remain distinct. No full HIP, streams, threading, arbitrary shader
indexing safety or framework support is claimed.
