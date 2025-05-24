#cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(FlRlCompression LANGUAGES CXX CUDA)

# Find MPI
find_package(MPI REQUIRED)

# Find NCCL
find_path(NCCL_INCLUDE_DIR "nccl.h" /usr/include)
find_library(NCCL_LIB nccl
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib
    NO_DEFAULT_PATH
)

if(NCCL_INCLUDE_DIR AND NCCL_LIB)
    message(STATUS "Found NCCL: ${NCCL_LIB}")
else()
    message(FATAL_ERROR "NCCL not found")
endif()

# Include directories
include_directories(${NCCL_INCLUDE_DIR})

# Add executable
add_executable(compress src/main.cu src/timers/cpu_timer.cu src/timers/gpu_timer.cu src/args_parser.cu src/fl/fl_cpu.cu src/fl/fl_gpu.cu src/file_io.cu src/timers/cpu_timer_with_transfer.cu)

# Link MPI and NCCL
target_link_libraries(compress PRIVATE MPI::MPI_CXX ${NCCL_LIB})
