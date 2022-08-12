cmake_minimum_required(VERSION 3.10)
project(GemmCompute)

set(exe_file gemm_1x1_0)
# set(exe_file gemm_4x1_1)
# set(exe_file gemm_4x4_2)
# set(exe_file gemm_4x4_3)
# set(exe_file gemm_4x4_4)
# set(exe_file gemm_4x4_5)
# set(exe_file gemm_8x4_6)
# set(exe_file gemm_8x4_7)

set(srcs ./src/common.cpp
         ./src/main.cpp
         ./src/optimize/gemm_ref.cpp
         ./src/optimize/${exe_file}.cpp)

add_executable(${PROJECT_NAME} ${srcs})
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_compile_definitions(${PROJECT_NAME} PUBLIC EXE_FILE=${exe_file})