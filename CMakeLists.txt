# Hush Open NR — CMake build for Windows (MSVC). macOS/Linux use plugin/Makefile.
#
#   cmake -B build -A x64 -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#   cmake --build build --config Release
#
# Produces build/Release/OpenNR.ofx (OpenCL render path; CUDA is opt-in via
# -DHUSH_ENABLE_CUDA=ON once the CUDA port has been verified on hardware).

cmake_minimum_required(VERSION 3.20)
project(HushOpenNR LANGUAGES CXX)

option(HUSH_ENABLE_CUDA "Build the (hardware-unverified) CUDA render path" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

set(SUPPORT ofx/Support/Library)
add_library(OpenNR MODULE
    plugin/OpenNRPlugin.cpp
    plugin/OpenCLKernel.cpp
    plugin/SpeakPlugin.cpp
    plugin/SpeakOpenCLKernel.cpp
    ${SUPPORT}/ofxsCore.cpp
    ${SUPPORT}/ofxsImageEffect.cpp
    ${SUPPORT}/ofxsInteract.cpp
    ${SUPPORT}/ofxsLog.cpp
    ${SUPPORT}/ofxsMultiThread.cpp
    ${SUPPORT}/ofxsParams.cpp
    ${SUPPORT}/ofxsProperty.cpp
    ${SUPPORT}/ofxsPropertyValidation.cpp
)

target_include_directories(OpenNR PRIVATE plugin ofx/include ofx/Support/include)
target_link_libraries(OpenNR PRIVATE OpenCL::OpenCL)
set_target_properties(OpenNR PROPERTIES PREFIX "" SUFFIX ".ofx")

if (MSVC)
    target_compile_definitions(OpenNR PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_options(OpenNR PRIVATE /O2 /EHsc)
endif()

if (HUSH_ENABLE_CUDA)
    enable_language(CUDA)
    target_sources(OpenNR PRIVATE plugin/CudaKernel.cu plugin/SpeakCudaKernel.cu)
    target_compile_definitions(OpenNR PRIVATE HUSH_ENABLE_CUDA)
endif()
