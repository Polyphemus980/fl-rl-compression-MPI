#include "cpu_timer_with_transfer.cuh"

namespace Timers
{
    void CpuTimerWithTransfer::startTransfer()
    {
        this->_transferSize = 0;
        this->start();
    }

    void CpuTimerWithTransfer::endTransfer()
    {
        this->end();
    }

    void CpuTimerWithTransfer::printResult(const char *s)
    {
        auto transferSpeed = this->_transferSize / (this->_timeInMS / 1000.0);
        auto unit = "B/s";
        if (transferSpeed > 1024 * 1024 * 1024)
        {
            transferSpeed /= (1024 * 1024 * 1024);
            unit = "GB/s";
        }
        else if (transferSpeed > 1024 * 1024)
        {
            transferSpeed /= (1024 * 1024);
            unit = "MB/s";
        }
        else if (transferSpeed > 1024)
        {
            transferSpeed /= 1024;
            unit = "KB/s";
        }
        if (this->rank != -1)
        {
            printf("[Rank: %d] ", this->rank);
        }
        printf("[TIMER] Step: \"%s\", Transfer Speed: %ld [%s] \n", s, transferSpeed, unit);
    }
} // Timers