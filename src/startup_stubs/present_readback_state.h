#pragma once

#include "startup_stubs/fence_completion.h"

namespace x2w_startup {

class PresentReadbackState {
    std::uint64_t target_ = 0;
    bool unusable_ = false;

public:
    void submitted(std::uint64_t target) noexcept { target_ = target; }
    void signal_failed() noexcept { unusable_ = true; }
    bool pending() const noexcept { return target_ != 0; }
    bool unusable() const noexcept { return unusable_; }

    // Observe only a signalled submission; failure leaves it unavailable for reuse.
    FenceCompletion observe(std::uint64_t observed) noexcept
    {
        const auto state = fence_completion(observed, target_);
        if (state == FenceCompletion::removed) unusable_ = true;
        if (state == FenceCompletion::completed) target_ = 0;
        return state;
    }
};

} // namespace x2w_startup
