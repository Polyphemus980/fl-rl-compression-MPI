# On linux: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(FlRlCompression LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
include(ExternalProject)

# Set NCCL build options
set(NCCL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/nccl")
set(NCCL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/nccl")
set(NCCL_INSTALL_DIR "${NCCL_BINARY_DIR}/install")

# Build NCCL using its native build system
ExternalProject_Add(nccl_external
  SOURCE_DIR "${NCCL_SOURCE_DIR}"
  BINARY_DIR "${NCCL_BINARY_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make -C "${NCCL_SOURCE_DIR}" src.build BUILDDIR=${NCCL_BINARY_DIR}
  INSTALL_COMMAND make -C "${NCCL_SOURCE_DIR}" install DESTDIR=${NCCL_INSTALL_DIR}
  BUILD_BYPRODUCTS "${NCCL_INSTALL_DIR}/lib/libnccl.so"
)

# Create an interface library for NCCL
add_library(nccl INTERFACE)
add_dependencies(nccl nccl_external)

# Set include directories
target_include_directories(nccl INTERFACE 
  "${NCCL_INSTALL_DIR}/include"
)

# Set link libraries
target_link_libraries(nccl INTERFACE
  "${NCCL_INSTALL_DIR}/lib/libnccl.so"
)

add_executable(compress src/main.cu src/timers/cpu_timer.cu src/timers/gpu_timer.cu src/args_parser.cu src/fl/fl_cpu.cu src/fl/fl_gpu.cu src/file_io.cu)

target_link_libraries(compress PRIVATE MPI::MPI_CXX)
target_link_libraries(compress PRIVATE nccl)
# Tests
enable_testing()

# FL_GPU_TESTS
add_executable(fl_gpu_tests tests/fl/test_fl_gpu.cu src/fl/fl_gpu.cu src/timers/cpu_timer.cu src/timers/gpu_timer.cu)
# Disable specific warning for NVIDIA compiler - it's something about acutest not related to the project itself
target_compile_options(fl_gpu_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=2949>)
target_link_libraries(fl_gpu_tests PRIVATE MPI::MPI_CXX)
add_test(NAME fl_gpu_tests COMMAND fl_gpu_tests)
