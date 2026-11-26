#include "startup_stubs/gpu_build_queue.h"
#include <cstdio>
#include <future>
#include <stdexcept>

int main()
{
    using namespace std::chrono_literals;
    using Queue = x2w_gpu::VpBuildQueue<int>;
    Queue queue(1, 1, 40ms, 1500ms);
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };
    std::promise<void> release, started;
    auto gate = release.get_future().share();
    auto state = std::make_shared<int>(0);
    std::weak_ptr<int> lifetime = state;
    std::atomic<int> builds{0};
    auto slow = queue.request("shader", state, false, [&](int& value) {
        ++builds;
        started.set_value();
        gate.wait();
        value = 42;
    });
    started.get_future().wait();
    auto duplicate = queue.request("shader", std::make_shared<int>(7), false, [&](int&) { ++builds; });
    state.reset();
    check(slow == duplicate && !lifetime.expired(), "equal requests share a job that owns its input");
    // The next caller, on another title thread, must not reserve an already spent frame budget.
    check(!queue.ready(slow, 10, false), "blocked compiler yields after its frame budget");
    auto start = Queue::Clock::now();
    check(!queue.ready(duplicate, 10, false) && Queue::Clock::now() - start < 30ms, "same frame cannot spend its budget twice");
    auto screen = queue.request("screen", std::make_shared<int>(0), true, [](int& value) { value = 99; });
    check(queue.ready(screen, 10, true) && *screen->state == 99, "screen completes with ordinary worker blocked");
    release.set_value();
    check(queue.ready(slow, 11, false) && *duplicate->state == 42 && builds == 1, "later call receives the actual shared result");
    auto bad = queue.request("throw", std::make_shared<int>(0), false, [](int&) { throw std::runtime_error("test compile"); });
    check(queue.ready(bad, 12, false) && bad->failure, "throwing compiler reports terminal failure");
    auto next = queue.request("next", std::make_shared<int>(0), false, [](int& value) { value = 123; });
    check(queue.ready(next, 13, false) && !next->failure && *next->state == 123, "worker remains available after failure");
    Queue guard(1, 0, 1500ms, 0ms);
    std::thread::id retired;
    auto lost = guard.request("driver", std::make_shared<int>(0), false, [&](int&) {
        retired = std::this_thread::get_id();
        Queue::active->lost(std::make_exception_ptr(std::runtime_error("simulated worker loss")));
    });
    check(guard.ready(lost, 1, false) && lost->failure, "lost compile publishes terminal failure");
    auto repeated = guard.request("driver", std::make_shared<int>(0), false, [](int&) {});
    check(repeated == lost && repeated->failure, "same request receives the recorded failure");
    auto replacement = guard.request("other", std::make_shared<int>(0), false, [&](int& value) {
        value = std::this_thread::get_id() != retired;
    });
    check(guard.ready(replacement, 2, false) && *replacement->state == 1, "replacement worker completes the next compile");
    // The exposure reset draw was dropped when unrelated work reserved the shared screen budget.
    // Reproduce that state and ensure the exact initializer gets the first available worker.
    Queue init(1, 0, 0ms, 100ms);
    std::promise<void> open, occupied;
    auto opened = open.get_future().share();
    auto blocker = init.request("busy", std::make_shared<int>(), true, [&](int&) {
        occupied.set_value();
        opened.wait();
    });
    occupied.get_future().wait();
    check(!init.ready(blocker, 20, true), "unrelated compile exhausted screen budget");
    std::promise<void> allow_background;
    auto allowed = allow_background.get_future().share();
    auto background = init.request("ordinary-screen", std::make_shared<int>(), true, [&](int&) { allowed.wait(); });
    auto first_use = init.request("initialization-variant", std::make_shared<int>(), true, [](int& result) { result = 37; }, true);
    check(!init.ready(first_use, 20, true), "ordinary caller cannot wait on the same frame");
    auto release_worker = std::async(std::launch::async, [&] {
        std::this_thread::sleep_for(5ms);
        open.set_value();
    });
    check(init.ready_initialization(first_use) && *first_use->state == 37, "first initialization draw receives prioritized native variant");
    check(!init.ready_initialization(background), "initializer wait on stalled driver is bounded");
    start = Queue::Clock::now();
    check(!init.ready_initialization(background) && Queue::Clock::now() - start < 50ms, "concurrent or later calls cannot extend the deadline");
    allow_background.set_value();
    check(init.ready(background, 21, true), "late completion remains usable");
    check(guard.ready_initialization(lost) && lost->failure, "a terminal compiler fault ends the wait");
    return failures ? 1 : 0;
}
