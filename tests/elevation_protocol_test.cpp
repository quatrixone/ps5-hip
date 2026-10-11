// Verify the actual exported Eden and Lapy protocol definitions agree.
#include "protocol.hpp"
#define _Static_assert static_assert
#include "lapy_elevation_protocol.h"
#undef _Static_assert
#include <cstring>
#include <utility>
#include <initializer_list>

int main() {
    using namespace elevation;
    wire::Message request;
    request.pid = 123;
    lapy_elevation_message helper{LAPY_ELEVATION_MAGIC, LAPY_ELEVATION_VERSION, 24,
        LAPY_ELEVATION_REQUEST, LAPY_ELEVATION_FILESYSTEM, 123, LAPY_ELEVATION_OK};
    static_assert(sizeof(request) == sizeof(helper));
    if (std::memcmp(&request, &helper, sizeof(request)) != 0) return 1;
    for (auto pair : {std::pair{wire::Kind::prepare, LAPY_ELEVATION_PREPARE},
                      std::pair{wire::Kind::prepared, LAPY_ELEVATION_PREPARED},
                      std::pair{wire::Kind::response, LAPY_ELEVATION_RESPONSE}})
        if (static_cast<unsigned>(pair.first) != static_cast<unsigned>(pair.second)) return 2;
    if (static_cast<unsigned>(Status::protocol_error) != LAPY_ELEVATION_PROTOCOL_ERROR) return 3;
    auto response = request;
    response.kind = wire::Kind::response;
    if (!wire::matches(response, request, wire::Kind::response)) return 4;
    ++response.pid;
    if (wire::matches(response, request, wire::Kind::response)) return 5;
    response = request;
    ++response.version;
    if (wire::validate(response) != Status::unsupported_version) return 6;
    unsigned char bytes[24]{};
    unsigned calls = 0;
    if (!wire::transfer(bytes, sizeof(bytes), [&](auto *, size_t size) {
            ++calls; return size > 3 ? 3 : static_cast<int>(size);
        }) || calls != 8) return 7;
    if (wire::transfer(bytes, sizeof(bytes), [](auto *, size_t) { return 0; })) return 8;
    if (wire::transfer(bytes, sizeof(bytes), [](auto *, size_t size) { return size + 1; })) return 9;
    return 0;
}
