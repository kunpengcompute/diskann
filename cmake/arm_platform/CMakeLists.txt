
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override -Wno-return-type -include linux/time_types.h")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

include_directories(${PROJECT_SOURCE_DIR}/include)

if(NOT PYBIND)
    set(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS ON)
endif()

if (DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
    include_directories(${PROJECT_SOURCE_DIR}/gperftools/src)

endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

add_compile_options(-include linux/types.h)

find_package(Boost 1.74 REQUIRED COMPONENTS program_options)

if (NOT Boost_FOUND)
    message(FATAL_ERROR "Couldn't find Boost dependency")
endif()

include_directories(${Boost_INCLUDE_DIR})

# OpenBLAS Config (ARM)
set(OpenBLAS_INCLUDE_DIR "/opt/OpenBLAS/include/")
set(OpenBLAS_LIB_DIR "/opt/OpenBLAS/lib/")

find_library(OPENBLAS_LIB openblas PATHS ${OpenBLAS_LIB_DIR} NO_DEFAULT_PATH)
find_path(OPENBLAS_INCLUDE_DIR_2 cblas.h PATHS ${OpenBLAS_INCLUDE_DIR} NO_DEFAULT_PATH)

if(OPENBLAS_LIB AND OPENBLAS_INCLUDE_DIR_2)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
    message(STATUS "Found OpenBLAS_INCLUDE_DIR: ${OPENBLAS_INCLUDE_DIR_2}")
    include_directories(${OPENBLAS_INCLUDE_DIR_2})
    link_directories(${OpenBLAS_LIB_DIR})
    link_libraries(${OPENBLAS_LIB})
    # -lopenblas moved to target_link_libraries
else()
    message(FATAL_ERROR "OpenBLAS not found")
endif()

if(NOT PYBIND)
    set(DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS "-ltcmalloc")
endif()

if (DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
    add_definitions(-DRELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
endif()


set(DISKANN_ASYNC_LIB aio)


set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
if (NOT PYBIND)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -Ofast")
    if (NOT PORTABLE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
    endif()
else()
    # -Ofast is not supported in a python extension module
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -fPIC")
endif()


add_subdirectory(src)
if (NOT PYBIND)
    add_subdirectory(apps)
    add_subdirectory(apps/utils)
endif()

if (UNIT_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()


if (RESTAPI)
    add_subdirectory(apps/restapi)
endif()

include(clang-format.cmake)

if(PYBIND)
    add_subdirectory(python)
endif()
