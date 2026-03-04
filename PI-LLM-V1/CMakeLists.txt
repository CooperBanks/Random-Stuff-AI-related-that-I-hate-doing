cmake_minimum_required(VERSION 3.10)
project(TinyLLM)

set(CMAKE_CXX_STANDARD 17)

# Include paths
include_directories(
    include
    src
    Crow/include      # <--- add this line so compiler finds crow_all.h
    /usr/include/eigen3
	/usr/include/asio
)

add_executable(tiny_llm
    src/main.cpp
    src/lstm.h
    src/utils.h
)
