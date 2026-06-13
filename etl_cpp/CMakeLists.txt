cmake_minimum_required(VERSION 3.16)
project(sapde_etl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release por defecto (performance) ──────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Buscar OpenMP ──────────────────────────────────────────
find_package(OpenMP REQUIRED)

if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR
        "OpenMP no encontrado.\n"
        "MSVC: instalar Visual Studio Build Tools con 'C++ CMake tools for Windows'\n"
        "MinGW: usar MinGW-w64 con soporte OpenMP (gcc con -fopenmp)\n"
        "Si no puedes compilar, usa el fallback Python: src/sapde/etl/fallback.py"
    )
endif()

# ── Ejecutable ETL ─────────────────────────────────────────
add_executable(etl src/etl_main.cpp)

target_link_libraries(etl PRIVATE OpenMP::OpenMP_CXX)

# Flags por compilador
if(MSVC)
    target_compile_options(etl PRIVATE /O2 /W3)
    # /openmp ya lo agrega OpenMP::OpenMP_CXX en MSVC
else()
    # MinGW / GCC / Clang
    target_compile_options(etl PRIVATE -O3 -Wall)
endif()

# ── Directorio de salida ───────────────────────────────────
set_target_properties(etl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build"
)

message(STATUS "Compilador: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "OpenMP versión: ${OpenMP_CXX_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
