# On linux: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(FlRlCompression LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

add_executable(compress src/main.cu src/rl/rl_cpu.cu src/rl/rl_gpu.cu src/timers/cpu_timer.cu src/timers/gpu_timer.cu src/args_parser.cu src/fl/fl_cpu.cu src/fl/fl_gpu.cu src/file_io.cu)

target_link_libraries(compress PRIVATE MPI::MPI_CXX)
# Tests
enable_testing()

# FL_GPU_TESTS
add_executable(fl_gpu_tests tests/fl/test_fl_gpu.cu src/fl/fl_gpu.cu src/timers/cpu_timer.cu src/timers/gpu_timer.cu)
# Disable specific warning for NVIDIA compiler - it's something about acutest not related to the project itself
target_compile_options(fl_gpu_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=2949>)
target_link_libraries(fl_gpu_tests PRIVATE MPI::MPI_CXX)
add_test(NAME fl_gpu_tests COMMAND fl_gpu_tests)
