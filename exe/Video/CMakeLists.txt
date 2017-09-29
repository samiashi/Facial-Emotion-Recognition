#TBB library
include_directories(${TBB_ROOT_DIR}/include)

add_executable(Video Video.cpp)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)		

target_link_libraries(Video LandmarkDetector)
target_link_libraries(Video FaceAnalyser)

target_link_libraries(Video dlib)

target_link_libraries(Video ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS Video DESTINATION bin)
