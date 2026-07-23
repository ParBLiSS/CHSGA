cmake_minimum_required(VERSION 3.10)
project(ConSGA VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Options to control which executables (and their deps) are configured
option(BUILD_HANDLEGRAPH_TO_CSR "Build handlegraph_to_csr" OFF)
option(BUILD_CSR_TO_CHAR_GRAPHS "Build csr_to_char_graphs" OFF)
option(BUILD_PROTOBUF_TO_CSR "Build protobuf_to_csr" OFF)

set(BUILD_PYTHON_BINDINGS OFF CACHE INTERNAL "" FORCE)
set(RUN_DOXYGEN OFF CACHE INTERNAL "" FORCE)

# Common include directories used across optional targets
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/ext")

find_package(OpenMP REQUIRED)

# Require ALPHABET_SIZE if any of the graph-building targets need it
if(BUILD_CSR_TO_CHAR_GRAPHS)
  if (NOT DEFINED ALPHABET_SIZE)
    message(FATAL_ERROR "ALPHABET_SIZE is not defined. Run cmake with -DALPHABET_SIZE=4 or -DALPHABET_SIZE=5")
  endif()
  if (NOT ALPHABET_SIZE STREQUAL "4" AND NOT ALPHABET_SIZE STREQUAL "5")
    message(FATAL_ERROR "ALPHABET_SIZE must be 4 or 5 (you passed: ${ALPHABET_SIZE})")
  endif()
endif()

# ------------------------
# BUILD_HANDLEGRAPH_TO_CSR
# ------------------------
if(BUILD_HANDLEGRAPH_TO_CSR)
  # libbdsg is required only for this target
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/ext/libbdsg")
  add_executable(handlegraph_to_csr ${CMAKE_CURRENT_SOURCE_DIR}/src/handlegraph_to_csr.cpp)
  target_link_libraries(handlegraph_to_csr PRIVATE libbdsg)
  install(TARGETS handlegraph_to_csr RUNTIME DESTINATION bin)
endif()

# ---------------------------
# BUILD_CSR_TO_CHAR_GRAPHS
# ---------------------------
if(BUILD_CSR_TO_CHAR_GRAPHS)
  # parlay is required for csr_to_char_graphs (and other targets below that opt-in)
  include(FetchContent)
  FetchContent_Declare(parlaylib
    GIT_REPOSITORY  https://github.com/cmuparlay/parlaylib.git
    GIT_TAG         master
  )
  FetchContent_GetProperties(parlaylib)
  if(NOT parlaylib_POPULATED)
    FetchContent_Populate(parlaylib)
    add_subdirectory(${parlaylib_SOURCE_DIR} EXCLUDE_FROM_ALL)
  endif()

  add_executable(csr_to_char_graphs "${PROJECT_SOURCE_DIR}/src/csr_to_char_graphs.cpp")
  target_include_directories(csr_to_char_graphs
    PRIVATE
      "${PROJECT_SOURCE_DIR}/ext/ParSGA/ext/gapbs/src"
  )
  target_link_libraries(csr_to_char_graphs PUBLIC parlay OpenMP::OpenMP_CXX)

  # propagate both ALPHABET_SIZE and DAG
  target_compile_definitions(csr_to_char_graphs
    PRIVATE
      ALPHABET_SIZE=${ALPHABET_SIZE}
      DAG=${DAG}
  )

  install(TARGETS csr_to_char_graphs RUNTIME DESTINATION bin)
endif()


# --------------------------
# BUILD_PROTOBUF_TO_CSR
# --------------------------
if(BUILD_PROTOBUF_TO_CSR)
  include(FetchContent)
  FetchContent_Declare(parlaylib
    GIT_REPOSITORY  https://github.com/cmuparlay/parlaylib.git
    GIT_TAG         master
  )
  FetchContent_GetProperties(parlaylib)
  if(NOT parlaylib_POPULATED)
    FetchContent_Populate(parlaylib)
    add_subdirectory(${parlaylib_SOURCE_DIR} EXCLUDE_FROM_ALL)
  endif()

  # htslib (only needed for protobuf_to_csr)
  include_directories("${PROJECT_SOURCE_DIR}/ext/PaSGAL/ext/htslib")
  add_custom_target(LIBHTS COMMAND make WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/ext/PaSGAL/ext/htslib" COMMENT "Compiling htslib")
  set(HTS_LIBRARY "${PROJECT_SOURCE_DIR}/ext/PaSGAL/ext/htslib/libhts.so")

  # Allow user to pass PROTOBUF_DIR or fall back to active conda env
  if(NOT DEFINED PROTOBUF_DIR)
    set(PROTOBUF_DIR "" CACHE PATH "absolute path to protobuf installation directory")
  endif()
  if(PROTOBUF_DIR STREQUAL "" AND DEFINED ENV{CONDA_PREFIX})
    set(PROTOBUF_DIR $ENV{CONDA_PREFIX})
  endif()

  set(PROTOBUF_INCLUDE_DIR "${PROTOBUF_DIR}/include")
  set(PROTOBUF_LIBRARY      "${PROTOBUF_DIR}/lib/libprotobuf.so")
  set(PROTOBUF_PROTOC_EXECUTABLE "${PROTOBUF_DIR}/bin/protoc")

  # Add PROTOBUF_DIR to CMake hint paths so find_package can leverage it
  if(PROTOBUF_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${PROTOBUF_DIR})
    list(APPEND CMAKE_LIBRARY_PATH ${PROTOBUF_DIR}/lib)
    include_directories(${PROTOBUF_DIR}/include)
  endif()

  # Early, clear error if expected files are absent
  if(NOT (EXISTS "${PROTOBUF_INCLUDE_DIR}" AND EXISTS "${PROTOBUF_LIBRARY}" AND EXISTS "${PROTOBUF_PROTOC_EXECUTABLE}"))
    message(FATAL_ERROR "Protobuf not found at PROTOBUF_DIR=${PROTOBUF_DIR}. Provide -DPROTOBUF_DIR=/abs/path or activate conda env before running cmake.")
  endif()

  # Prefer CMake FindProtobuf / Protobuf targets
  find_package(Protobuf REQUIRED)
  # Ensure CMake's Protobuf protoc variable is set when user provided PROTOBUF_DIR
  if(NOT Protobuf_PROTOC_EXECUTABLE AND EXISTS "${PROTOBUF_PROTOC_EXECUTABLE}")
    set(Protobuf_PROTOC_EXECUTABLE "${PROTOBUF_PROTOC_EXECUTABLE}" CACHE FILEPATH "protoc from PROTOBUF_DIR" FORCE)
  endif()

  include_directories(${PROTOBUF_INCLUDE_DIR})
  include_directories("${PROJECT_SOURCE_DIR}/ext/PaSGAL/ext/libvgio/include")
  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  # vg.proto path and generation
  set(VG_PROTO "${PROJECT_SOURCE_DIR}/ext/PaSGAL/ext/libvgio/deps/vg.proto")
  if(NOT EXISTS "${VG_PROTO}")
    message(FATAL_ERROR "vg.proto not found at ${VG_PROTO}")
  endif()

  # Use protobuf_generate_cpp if available, otherwise call protoc explicitly
  if (COMMAND protobuf_generate_cpp)
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${VG_PROTO})
  else()
    if(NOT Protobuf_PROTOC_EXECUTABLE)
      message(FATAL_ERROR "protoc executable not found; set -DPROTOBUF_DIR or ensure protoc is on PATH")
    endif()
    execute_process(
      COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I ${PROJECT_SOURCE_DIR} -I ${PROTOBUF_INCLUDE_DIR} ${VG_PROTO}
      RESULT_VARIABLE _protoc_ret
      OUTPUT_VARIABLE _protoc_out
      ERROR_VARIABLE _protoc_err
    )
    if(NOT _protoc_ret EQUAL 0)
      message(FATAL_ERROR "protoc failed generating vg.pb.cc:\n${_protoc_err}")
    endif()
    set(PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/vg.pb.cc")
    set(PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/vg.pb.h")
  endif()

  # SYMLNK target to preserve previous behavior (if needed by other steps)
  add_custom_target(SYMLNK COMMAND ${CMAKE_COMMAND} -E create_symlink . vg WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" COMMENT "Adding symbolic link vg")

  file(GLOB VGIO_SRC "${PROJECT_SOURCE_DIR}/ext/PaSGAL/ext/libvgio/src/*.cpp")
  add_library(LIBVGIO ${PROTO_SRCS} ${VGIO_SRC})
  add_dependencies(LIBVGIO SYMLNK)

  add_executable(protobuf_to_csr "${PROJECT_SOURCE_DIR}/src/protobuf_to_csr.cpp")
  add_dependencies(protobuf_to_csr LIBHTS SYMLNK)
  target_include_directories(protobuf_to_csr PRIVATE "${PROJECT_SOURCE_DIR}/ext/PaSGAL/src/include")
  target_link_libraries(protobuf_to_csr PUBLIC parlay ${PROTOBUF_LIBRARY} LIBVGIO ${HTS_LIBRARY} ${VTUNE_LIBRARY} OpenMP::OpenMP_CXX -lz -lpthread)
  install(TARGETS protobuf_to_csr RUNTIME DESTINATION bin)
endif()
