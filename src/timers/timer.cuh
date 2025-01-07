#ifndef TIMER_H
#define TIMER_H

namespace Timers
{
    class Timer
    {
    public:
        void virtual start() = 0;
        void virtual end() = 0;
        void virtual printResult(const char *s) = 0;
    };
} // Timers

#endif // TIMER_H