cmake_minimum_required(VERSION 2.8)

project(selective_search)

execute_process(COMMAND python-config --cflags  OUTPUT_VARIABLE PYTHON3_CFLAGS  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python-config --ldflags OUTPUT_VARIABLE PYTHON3_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
set(BOOST_LIBS "-lboost_system -lboost_python3 -lboost_numpy")

set(SUPPRESS_WARNING "-Wall -Wno-unused-function -Wno-unused-variable -Wno-unused-local-typedefs")

include_directories("segment")
set(CMAKE_CXX_FLAGS "-fPIC -O3 ${PYTHON3_CFLAGS} ${SUPPRESS_WARNING}")
set(CMAKE_SHARED_LINKER_FLAGS "${PYTHON3_LDFLAGS} -shared -fPIC")

add_library(segment SHARED segment_py.cpp)
target_link_libraries(segment ${BOOST_LIBS} segment)
set_target_properties(segment PROPERTIES PREFIX "")
