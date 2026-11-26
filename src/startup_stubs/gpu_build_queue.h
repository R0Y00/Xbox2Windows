#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace x2w_gpu {

// One job owns all input it needs independently of the title pipeline. Equal keys requested
// concurrently share one build; screen passes have reserved workers and a separate frame budget.
template<class T> class VpBuildQueue {
public:
    using Ms = std::chrono::milliseconds;
    using Clock = std::chrono::steady_clock;
    struct Job {
        std::shared_ptr<T> state;
        std::atomic<bool> done{false};
        std::exception_ptr failure;
        Clock::time_point first_use_deadline{};  // protected by the queue mutex
        std::function<void(T&)> build;
    };
    using Handle = std::shared_ptr<Job>;
    // A worker publishes this stack record only while it owns a build. An external
    // monitor may take it over only once the worker is permanently parked.
    struct ActiveBuild {
        VpBuildQueue* queue;
        Handle job;
        bool screen;
        std::thread::id thread;
        void lost(std::exception_ptr reason) const { queue->replace(*this, reason); }
    };
    static inline thread_local ActiveBuild* active = nullptr;

    VpBuildQueue(unsigned general, unsigned screen_workers, Ms ordinary_budget, Ms screen_budget,
                 void (*thread_start)(bool) = nullptr) : limits_{ordinary_budget, screen_budget}, thread_start_(thread_start) {
        try {
            for (unsigned i = 0; i < general + screen_workers; ++i)
                workers_.emplace_back([this, only_screen = i >= general] { work(only_screen); });
        } catch (...) {
            shutdown();
            throw;
        }
    }
    ~VpBuildQueue() { shutdown(); }

    Handle request(std::string key, const std::shared_ptr<T>& state, bool screen, std::function<void(T&)> build, bool initialization = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& weak = jobs_[std::move(key)];
        if (auto job = weak.lock()) return job;
        auto job = std::make_shared<Job>();
        job->state = state;
        job->build = std::move(build);
        auto& lane = screen && limits_[1].count() != 0 ? screen_ : normal_;
        if (initialization) lane.push_front(job);
        else lane.push_back(job);
        weak = job;
        available_.notify_all();
        return job;
    }

    // A one-shot initializer cannot rely on the title recording the draw again. All
    // callers share one absolute deadline for its exact native pipeline, including variants.
    bool ready_initialization(const Handle& job) {
        if (job->done.load(std::memory_order_acquire)) return true;
        std::unique_lock<std::mutex> lock(mutex_);
        if (job->first_use_deadline == Clock::time_point{})
            job->first_use_deadline = Clock::now() + (limits_[1].count() ? limits_[1] : limits_[0]);
        return finished_.wait_until(lock, job->first_use_deadline, [&] {
            return job->done.load(std::memory_order_acquire);
        });
    }

    bool ready(const Handle& job, std::uint64_t frame, bool screen) {
        if (job->done.load(std::memory_order_acquire)) return true;
        const unsigned lane = screen && limits_[1].count() != 0 ? 1 : 0;
        std::unique_lock<std::mutex> lock(mutex_);
        auto& budget = budgets_[lane];
        if (!budget.started || budget.frame < frame)
            budget = {frame, limits_[lane], true};
        if (budget.frame != frame || budget.left <= Clock::duration::zero())
            return job->done.load(std::memory_order_acquire);
        // Reserve the rest now, so concurrent recording threads cannot each spend a full budget.
        const auto started = Clock::now();
        const auto deadline = started + budget.left;
        budget.left = {};
        finished_.wait_until(lock, deadline, [&] { return job->done.load(std::memory_order_acquire); });
        const auto now = Clock::now();
        if (budget.frame == frame && now < deadline) budget.left += deadline - now;
        return job->done.load(std::memory_order_acquire);
    }

private:
    struct Budget { std::uint64_t frame{}; Clock::duration left{}; bool started{}; };
    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
            available_.notify_all();
            finished_.wait(lock, [&] { return closed_workers_ == workers_.size(); });
        }
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    void work(bool only_screen) {
        if (thread_start_) thread_start_(only_screen);
        for (;;) {
            Handle job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                available_.wait(lock, [&] { return stop_ || !screen_.empty() || (!only_screen && !normal_.empty()); });
                if (stop_) {
                    ++closed_workers_;
                    finished_.notify_all();
                    return;
                }
                auto& q = screen_.empty() ? normal_ : screen_;
                job = std::move(q.front());
                q.pop_front();
            }
            ActiveBuild owner{this, job, only_screen, std::this_thread::get_id()};
            active = &owner;
            try { job->build(*job->state); }
            catch (...) { job->failure = std::current_exception(); }
            active = nullptr;
            // A test worker can report its own loss and return normally.
            if (job->done.load(std::memory_order_acquire)) return;
            job->build = {};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                job->done.store(true, std::memory_order_release);
            }
            finished_.notify_all();
        }
    }

    void replace(const ActiveBuild& owner, std::exception_ptr reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        owner.job->failure = reason;
        owner.job->done.store(true, std::memory_order_release);
        finished_.notify_all();
        for (auto& worker : workers_) {
            if (worker.get_id() != owner.thread) continue;
            worker.detach();
            ++closed_workers_;
            if (!stop_) {
                worker = std::thread([this, screen = owner.screen] { work(screen); });
                --closed_workers_;
            }
            break;
        }
    }

    Ms limits_[2];
    void (*thread_start_)(bool);
    Budget budgets_[2]{};
    std::mutex mutex_;
    std::condition_variable available_, finished_;
    std::deque<Handle> normal_, screen_;
    std::unordered_map<std::string, std::weak_ptr<Job>> jobs_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
    std::size_t closed_workers_ = 0;
};

} // namespace x2w_gpu
