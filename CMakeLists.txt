# On linux: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(FlRlCompression LANGUAGES CXX CUDA)

add_executable(FlRlCompression src/main.cu src/rl/rl_cpu.cu src/rl/rl_gpu.cu src/timers/cpu_timer.cu src/timers/gpu_timer.cu, src/args_parser.cu)

enable_testing()

add_executable(rl_cpu_tests tests/rl/test_rl_cpu.cu src/rl/rl_cpu.cu src/timers/cpu_timer.cu)

# Disable specific warning for NVIDIA compiler - it's something about acutest not related to the project itself
target_compile_options(rl_cpu_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=2949>)

add_test(NAME rl_cpu_tests COMMAND rl_cpu_tests)


add_executable(rl_gpu_tests tests/rl/test_rl_gpu.cu src/rl/rl_gpu.cu src/timers/cpu_timer.cu src/timers/gpu_timer.cu)

# Disable specific warning for NVIDIA compiler - it's something about acutest not related to the project itself
target_compile_options(rl_gpu_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=2949>)

add_test(NAME rl_gpu_tests COMMAND rl_gpu_tests)