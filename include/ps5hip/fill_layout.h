#pragma once
#include <cstddef>
#include <limits>
namespace ps5hip {
struct FillLayout { size_t span; size_t slice; };
// Bound every multiplication/addition before constructing the last touched byte.
// Padding is addressable but never part of the fill itself.
inline bool checked_fill_layout(size_t width, size_t height, size_t depth,
                                size_t pitch, size_t slice_rows, size_t available,
                                FillLayout& result) noexcept {
    if (!width || !height || !depth || pitch < width || slice_rows < height ||
        pitch > std::numeric_limits<size_t>::max() / slice_rows || width > available)
        return false;
    const size_t slice = pitch * slice_rows;
    size_t remaining = available - width;
    if (height - 1 > remaining / pitch) return false;
    const size_t rows = (height - 1) * pitch;
    remaining -= rows;
    if (depth - 1 > remaining / slice) return false;
    result = {width + rows + (depth - 1) * slice, slice};
    return true;
}
} // namespace ps5hip
