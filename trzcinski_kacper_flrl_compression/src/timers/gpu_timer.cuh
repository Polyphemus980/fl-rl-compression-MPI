#ifndef GPU_TIMER_H
#define GPU_TIMER_H

#include "timer.cuh"

namespace Timers
{

    class GpuTimer : public Timer
    {
    private:
        cudaEvent_t _start{};
        cudaEvent_t _end{};
        float _timeInMS{};

    public:
        void start();
        void end();
        void printResult(const char *s);
    };

} // Timers

#endif // GPU_TIMER_H