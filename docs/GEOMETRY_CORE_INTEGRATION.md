# Geometry planner integration

This isolated step starts at checked planner source `ce24c017`, on a development
branch. It combines the reviewed geometry route from `7bdcea0` with the existing
initial, buffer-scratch, LDS and structural private-call planners. No allocator,
submission queue, registration, argument capture, native HIP helper or title
control changes are included.

The existing shared preparation/packet helpers gain an optional geometry shape;
only the explicit geometry wrappers select it. Admission requires the measured
`0x138c` six-user-SGPR profile, wave32, zero scratch/LDS/spills and the complete
supported hidden argument layout. Shapes remain 8x8x1 for two dimensions or
8x4x2 for three dimensions. Axis extents and the complete grid product are
checked. Encoding revalidates all 256 implicit bytes, including reserved zeros,
against the chosen groups, threads and dimensions. Geometry cannot silently
combine with scratch, LDS or the eight-SGPR private entry.

The imported geometry test checks every hidden byte, descriptor mutations and
independently decoded packets against four existing O1/O2 objects. The integration
matrix exercises all five route boundaries and rejects resource/shape substitution;
the prior literal initial packet baseline and scratch/LDS/private test suites remain.
All compiler objects are read-only inputs; none is rebuilt for this integration.

Hardware evidence for the isolated 2D fixture does not qualify these merged
sources. No new HIP API admission, native harness, package, arbitrary geometry or
combined-resource execution is claimed. The structural private-call API still
cannot certify an arbitrary call graph or private aperture. Broader runtime
selection and ownership of optional scratch remain subsequent work.
