cmake_minimum_required(VERSION 3.20)

project(SpriteSheetDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(SpriteSheetDetector
  src/main.cpp
  src/SpriteDetector.cpp
  src/SpriteDetector.h
  src/GridSpriteDetector.cpp
  src/GridSpriteDetector.h
  src/ForegroundMask.cpp
  src/ForegroundMask.h
)

target_link_libraries(SpriteSheetDetector PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

target_include_directories(SpriteSheetDetector PRIVATE src)

# Ensure UTF-8 source decoding on MSVC (avoids C4819 with Chinese comments).
if (MSVC)
  target_compile_options(SpriteSheetDetector PRIVATE /utf-8)
endif()

