#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)	
			
add_executable(Image Image.cpp)
target_link_libraries(Image LandmarkDetector)
target_link_libraries(Image FaceAnalyser)
target_link_libraries(Image dlib)

target_link_libraries(Image ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS Image DESTINATION bin)
