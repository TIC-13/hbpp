#ifndef HALIDE_BENCHMARK_PP
#define HALIDE_BENCHMARK_PP

#include <chrono>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>
#include <math.h>
#include <numeric>

namespace hbpp {

// Prefer high_resolution_clock, but only if it's steady...
template<bool HighResIsSteady = std::chrono::high_resolution_clock::is_steady>
struct SteadyClock {
    using type = std::chrono::high_resolution_clock;
};

// ...otherwise use steady_clock.
template<>
struct SteadyClock<false> {
    using type = std::chrono::steady_clock;
};

inline SteadyClock<>::type::time_point benchmark_now() {
    return SteadyClock<>::type::now();
}

inline double benchmark_duration_ms(
    SteadyClock<>::type::time_point start,
    SteadyClock<>::type::time_point end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() * 1e3;
}

struct BenchmarkStats {
    // Best elapsed time (milliseconds).
    double best_time_ms = 1e9;

    // Worst elapsed time (milliseconds).
    double worst_time_ms = -1.0;

    // Mean of elapsed times (milliseconds).
    double mean_time_ms = 0;

    // Standard deviation of elapsed times (milliseconds).
    double std_time_ms = 0;

    // Number of samples used for measurement.
    uint64_t samples;
};

// Benchmark the operation 'op'. The number of iterations refers to
// how many times the operation is run for each time measurement, the
// result is the minimum over a number of samples runs. The result is the
// amount of time in seconds for one iteration.
//
// IMPORTANT NOTE: Using this tool for timing GPU code may be misleading,
// as it does not account for time needed to synchronize to/from the GPU;
// if the callback doesn't include calls to device_sync(), the reported
// time may only be that to queue the requests; if the callback *does*
// include calls to device_sync(), it might exaggerate the sync overhead
// for real-world use. For now, callers using this to benchmark GPU
// code should measure with extreme caution.
inline BenchmarkStats benchmark(uint64_t samples, uint64_t iterations, const std::function<void()> &op) {
    BenchmarkStats stats;
    stats.samples = samples;
    std::vector<double> times;
    double total = 0;
    for (uint64_t i = 0; i < samples; i++) {
        auto start = benchmark_now();
        for (uint64_t j = 0; j < iterations; j++) {
            op();
        }
        auto end = benchmark_now();
        double t = benchmark_duration_ms(start, end);
        times.push_back(t);
        total += t;
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    stats.mean_time_ms = sum / times.size();
    for (double t : times) {
        stats.best_time_ms = std::min(stats.best_time_ms, t);
        stats.worst_time_ms = std::max(stats.worst_time_ms, t);
        stats.std_time_ms += (t - stats.mean_time_ms) * (t - stats.mean_time_ms);
    }
    stats.std_time_ms = stats.std_time_ms / (samples - 1);
    return stats;
}

}  // namespace hbpp

#endif