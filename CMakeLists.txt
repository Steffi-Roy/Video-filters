cmake_minimum_required(VERSION 3.10)
project(OpenCVDemo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED libonnxruntime)

add_executable(app img_display.cpp)
add_executable(vid
    vidDisplay.cpp
    filter.cpp
    faceDetect/faceDetect.cpp
)
add_executable(blurTiming
    blurTiming.cpp
    filter.cpp
)
add_executable(depthImage
    depthImage.cpp
    filter.cpp
)
add_executable(extensions
    extensions.cpp
)

target_link_libraries(app ${OpenCV_LIBS})
target_include_directories(vid PRIVATE da2-code ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_directories(vid PRIVATE ${ONNXRUNTIME_LIBRARY_DIRS})
target_link_libraries(vid ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(blurTiming ${OpenCV_LIBS})
target_include_directories(depthImage PRIVATE da2-code ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_directories(depthImage PRIVATE ${ONNXRUNTIME_LIBRARY_DIRS})
target_link_libraries(depthImage ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(extensions ${OpenCV_LIBS})
