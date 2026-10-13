#pragma once
#include <cstdint>
namespace ps5hip_contention {
struct Report {
    bool passed=false,cleanup_complete=false,unsafe_teardown=false;
    unsigned created=0,joined=0,dispatches=0,values=0,words=0,templates=0,hip_error=0;
    int error=0;
};
unsigned actor() noexcept;
void fail(int) noexcept;
int error() noexcept;
bool await_hold() noexcept;
bool evidence(bool print) noexcept;
const Report &report() noexcept;
}
int ps5hip_contention_client_main(int,char**);
