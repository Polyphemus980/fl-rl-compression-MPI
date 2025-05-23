#ifndef CPU_TIMER_WITH_TRANSFER_H
#define CPU_TIMER_WITH_TRANSFER_H

#include "cpu_timer.cuh"

namespace Timers
{
    class CpuTimerWithTransfer : public CpuTimer
    {
    private:
        uint64_t _transferSize{};

    public:
        CpuTimerWithTransfer() = default;
        CpuTimerWithTransfer(int rank) : CpuTimer(rank) {}
        void startTransfer();
        void endTransfer();
        void printTransferResult(const char *s);
        void addTransferSize(uint64_t size)
        {
            this->_transferSize += size;
        }
    };
} // Timers

#endif // CPU_TIMER_WITH_TRANSFER_H