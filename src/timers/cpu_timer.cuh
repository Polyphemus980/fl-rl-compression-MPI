#ifndef CPU_TIMER_H
#define CPU_TIMER_H

#include <chrono>
#include <cstdint>
#include <string>

#include "timer.cuh"

namespace Timers
{
    class CpuTimer : public Timer
    {
    private:
        std::chrono::high_resolution_clock::time_point _start{};
        std::chrono::high_resolution_clock::time_point _end{};
        
    protected:
        int64_t _timeInMS{};
        int rank = -1;

    public:
        CpuTimer() = default;
        CpuTimer(int rank) : rank(rank) {}
        CpuTimer(const CpuTimer &other) = default;
        void start();
        void end();
        void printResult(const char *s);
    };
} // Timers

#endif // CPU_TIMER_H