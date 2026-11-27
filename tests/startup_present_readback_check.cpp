#include "startup_stubs/present_readback_state.h"

#include <cstdio>
#include <cstdlib>

namespace {
using x2w_startup::FenceCompletion;
using x2w_startup::PresentReadbackState;
using x2w_startup::fence_completion;

void require(bool ok, const char* message)
{
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}
}

int main()
{
    PresentReadbackState state;
    require(!state.pending() && !state.unusable(), "initial readback not reusable");
    state.submitted(7);
    require(state.pending(), "submission not retained");
    require(state.observe(6) == FenceCompletion::pending && state.pending(),
            "premature completion would release the submitted surface");
    require(state.observe(7) == FenceCompletion::completed && !state.pending() && !state.unusable(),
            "completed submission did not allow reuse");
    state.submitted(8);
    require(state.observe(7) == FenceCompletion::pending && state.pending(),
            "stale event would publish a later submission");
    require(state.observe(9) == FenceCompletion::completed && !state.pending(),
            "past-target completion did not settle submission");
    state.submitted(10);
    require(state.observe(UINT64_MAX) == FenceCompletion::removed && state.unusable(),
            "device removal would publish or allow reuse");

    PresentReadbackState failed;
    failed.signal_failed();
    require(failed.unusable(), "failed signal would allow reuse");
    require(fence_completion(UINT64_MAX - 1, UINT64_MAX - 1) == FenceCompletion::completed,
            "highest usable fence value rejected");
    require(fence_completion(UINT64_MAX, UINT64_MAX) == FenceCompletion::removed,
            "reserved fence value accepted as completion");
    std::puts("present readback: pending, stale event, completion, failed signal and removal controls passed");
}
