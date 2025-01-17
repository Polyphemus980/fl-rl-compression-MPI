# On linux: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(FlRlCompression LANGUAGES CXX CUDA)

add_executable(compress src/main.cu src/rl/rl_cpu.cu src/rl/rl_gpu.cu src/timers/cpu_timer.cu src/timers/gpu_timer.cu src/args_parser.cu src/fl/fl_cpu.cu src/fl/fl_gpu.cu src/file_io.cu)
