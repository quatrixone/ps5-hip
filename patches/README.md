# Pinned native converter patch

`native-import-libraries.patch` applies to the native tooling exported from
PS5_Vulkan revision `3f3ee69607013b345d2baa6d6a37c86745649a08`.
It modifies GPL-3.0-or-later source and retains that license and its source notices.
The builder exports fresh pinned files before applying it with zero fuzz; it does
not modify the reference repository.

Purpose: express separate import-library names within a needed module. The
default conversion path retains its original output. Overrides are explicit
`MODULE:SYMBOL:LIBRARY` declarations whose platform provenance must be reviewed
separately. No firmware code or export database is included in this patch.

Validation and usage are in `docs/NATIVE_BUILD.md`.

`native-shell-exit-probe.patch` adds a conditional, pre-initializer registration
point to the same pinned startup source. It is applied only for the CPU shell-exit
diagnostic and inherits the startup source's GPL-3.0-or-later license. This does
not change the default startup source or general platform-exit behavior.
