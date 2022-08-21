cmake_minimum_required(VERSION 3.14)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_C_EXTENSIONS ON)

project(Strassen VERSION 1.0.0)

add_compile_options(-pedantic)
add_compile_options(-Wextra)
add_compile_options(-march=native)
# add_compile_options(-Rpass=.*)
# add_compile_options(-Rpass=loop-vectorize)
# add_compile_options(-Rpass-analysis=loop-vectorize)

add_link_options(-flto)

option(DEBUG "Compile in Debug mode" OFF)
option(PROFILE "Compile with profiling enabled" OFF)
option(PARALLEL "Compile with parallelism enabled" OFF)

if (PARALLEL)
    add_compile_options(-DPARALLEL)
    add_link_options(-L/usr/local/opt/libomp/lib/)
    add_link_options(-Xpreprocessor)
    add_link_options(-lomp)
    include_directories("/usr/local/opt/libomp/include")
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

if (PROFILE)
    add_compile_options(-DDEBUG)
    add_compile_options(-O0)
    add_compile_options(-g)
    add_compile_options(-fprofile-arcs)
    add_compile_options(-ftest-coverage)
    add_link_options(-fprofile-arcs)
    add_link_options(-ftest-coverage)
elseif(DEBUG)
    add_compile_options(-DDEBUG)
    add_compile_options(-O0)
    add_compile_options(-g)
else()
    add_compile_options(-DNDEBUG)
    add_compile_options(-O3)
endif()

if (VERBOSE)
    add_compile_options(-DVERBOSE)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(CTest)
enable_testing()
# add_subdirectory(tests)

file(GLOB sources "${PROJECT_SOURCE_DIR}/*.c")

add_executable(strassen strassen.h strassen.c main.c utils.c)

include_directories("${PROJECT_SOURCE_DIR}")

file(GLOB sources "${PROJECT_SOURCE_DIR}/*.c")
list(REMOVE_ITEM sources "${PROJECT_SOURCE_DIR}/main.c")

file(GLOB tests "${PROJECT_SOURCE_DIR}/tests/*.cpp")
list(REMOVE_ITEM tests "${PROJECT_SOURCE_DIR}/tests/main.cpp")

foreach(file ${tests})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}_tests"
    ${sources}
    ${file}
    "${PROJECT_SOURCE_DIR}/tests/main.cpp")
  target_link_libraries("${name}_tests" gtest_main)
  add_test(NAME ${name} COMMAND "${name}_tests")
endforeach()


add_custom_target(coverage
                  COMMAND /usr/local/opt/llvm/bin/llvm-cov gcov "../build/CMakeFiles/strassen.dir/strassen.c.gcno")
