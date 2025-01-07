#include <cstdio>

#include "cpu_timer.cuh"

namespace Timers
{
    void CpuTimer::start()
    {
        this->_start = std::chrono::high_resolution_clock::now();
    }

    void CpuTimer::end()
    {
        this->_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->_end - this->_start);
        this->_timeInMS = duration.count();
    }

    void CpuTimer::printResult(const char *s)
    {
        printf("[TIMER] Step: \"%s\", Time: %ld ms\n", s, this->_timeInMS);
    }
} // Timers