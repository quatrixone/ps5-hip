# Refreshed wave32 candidate

This candidate rebuilds `wave_full` and `wave_partial` from the current
`a0498530cb1a2d9d6d0db6a62d6f0d4e8df92e3e` launch substrate. It preserves the
wave source and exact oracle from the older `5581dbd` candidate. Old packages,
compiler probes and receipts remain immutable. New native images require their
own qualification; no old hardware result is inherited.

The newer queue explicitly closes command and marker allocations when setup,
resource validation or pre-submit flushing fails. It reports failed cleanup as
`NativeMemoryCleanupError`, and allocation setup reports retained backing through
`NativeMemorySetupError`. The old implementation could lose that distinction in
destructor cleanup. Post-submit uncertain GPU ownership still poisons the queue
and retains submitted resources. The title never prints application pass after
either failure class; these changes do not claim that all failed OS allocations
can be reclaimed. Driver calls remain outside the software polling deadline.

Both wave profiles use one 64-thread group, two wave32 waves and 256 exact output
words. Full waves check broadcast, pair shuffle, odd ballot and sums 528/1552.
Partial waves activate lanes 4 through 19, with sums 200/712 and ballot 0x55550;
all inactive output words retain 0xdeadbeef. Prefix and the complete rounded suffix
remain guarded. Kernarg size is eight bytes, with six user SGPRs and the pointer
in s[4:5]. No hidden arguments, scratch or LDS are enabled.

The offline preparation replays the meaningful oracle and queue failure tests,
checks all four target/optimization compiler objects, and separately builds,
stages and verifies each native image. It does not qualify wave semantics on
hardware, asynchronous execution, full G3 or broader HIP behavior.
