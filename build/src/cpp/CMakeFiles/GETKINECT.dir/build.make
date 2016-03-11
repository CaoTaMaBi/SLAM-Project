# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lukezhu/slam/try22-final1.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lukezhu/slam/try22-final1.0/build

# Include any dependencies generated for this target.
include src/cpp/CMakeFiles/GETKINECT.dir/depend.make

# Include the progress variables for this target.
include src/cpp/CMakeFiles/GETKINECT.dir/progress.make

# Include the compile flags for this target's objects.
include src/cpp/CMakeFiles/GETKINECT.dir/flags.make

src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o: src/cpp/CMakeFiles/GETKINECT.dir/flags.make
src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o: ../src/cpp/Kinect_Input.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/lukezhu/slam/try22-final1.0/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o"
	cd /home/lukezhu/slam/try22-final1.0/build/src/cpp && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o -c /home/lukezhu/slam/try22-final1.0/src/cpp/Kinect_Input.cpp

src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.i"
	cd /home/lukezhu/slam/try22-final1.0/build/src/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/lukezhu/slam/try22-final1.0/src/cpp/Kinect_Input.cpp > CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.i

src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.s"
	cd /home/lukezhu/slam/try22-final1.0/build/src/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/lukezhu/slam/try22-final1.0/src/cpp/Kinect_Input.cpp -o CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.s

src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o.requires:
.PHONY : src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o.requires

src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o.provides: src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o.requires
	$(MAKE) -f src/cpp/CMakeFiles/GETKINECT.dir/build.make src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o.provides.build
.PHONY : src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o.provides

src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o.provides.build: src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o

# Object files for target GETKINECT
GETKINECT_OBJECTS = \
"CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o"

# External object files for target GETKINECT
GETKINECT_EXTERNAL_OBJECTS =

../lib/libGETKINECT.so: src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o
../lib/libGETKINECT.so: src/cpp/CMakeFiles/GETKINECT.dir/build.make
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../lib/libGETKINECT.so: /usr/local/lib/libopencv_xphoto.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_ximgproc.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_tracking.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_text.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_surface_matching.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_stereo.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_saliency.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_rgbd.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_reg.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_optflow.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_line_descriptor.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_latentsvm.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_face.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_datasets.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_ccalib.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_bioinspired.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_bgsegm.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_adas.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_videostab.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_videoio.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_video.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_superres.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_stitching.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_shape.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_photo.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_objdetect.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_ml.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_imgproc.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_highgui.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_hal.a
../lib/libGETKINECT.so: /usr/local/lib/libopencv_flann.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_features2d.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_core.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_calib3d.so.3.0.0
../lib/libGETKINECT.so: /usr/lib/libOpenNI.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/lib/libOpenNI.so
../lib/libGETKINECT.so: /usr/lib/libOpenNI2.so
../lib/libGETKINECT.so: /usr/lib/libvtkCommon.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkFiltering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkImaging.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGraphics.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGenericFiltering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkIO.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkRendering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkVolumeRendering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkHybrid.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkWidgets.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkParallel.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkInfovis.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGeovis.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkViews.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkCharts.so.5.8.0
../lib/libGETKINECT.so: /usr/local/lib/libpcl_io.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_octree.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/lib/libOpenNI.so
../lib/libGETKINECT.so: /usr/lib/libOpenNI2.so
../lib/libGETKINECT.so: /usr/lib/libvtkCommon.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkFiltering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkImaging.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGraphics.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGenericFiltering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkIO.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkRendering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkVolumeRendering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkHybrid.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkWidgets.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkParallel.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkInfovis.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGeovis.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkViews.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkCharts.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libGETKINECT.so: /usr/local/lib/libpcl_visualization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/lib/libOpenNI.so
../lib/libGETKINECT.so: /usr/lib/libOpenNI2.so
../lib/libGETKINECT.so: /usr/lib/libvtkCommon.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkFiltering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkImaging.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGraphics.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGenericFiltering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkIO.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkRendering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkVolumeRendering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkHybrid.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkWidgets.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkParallel.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkInfovis.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGeovis.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkViews.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkCharts.so.5.8.0
../lib/libGETKINECT.so: /usr/local/lib/libpcl_io.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_octree.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libGETKINECT.so: /usr/local/lib/libpcl_kdtree.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libGETKINECT.so: /usr/local/lib/libpcl_search.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libGETKINECT.so: /usr/local/lib/libpcl_kdtree.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_octree.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libGETKINECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_common.so
../lib/libGETKINECT.so: /usr/lib/libOpenNI.so
../lib/libGETKINECT.so: /usr/lib/libOpenNI2.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_io.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_visualization.so
../lib/libGETKINECT.so: /usr/local/lib/libpcl_search.so
../lib/libGETKINECT.so: /usr/local/lib/libopencv_text.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_face.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_shape.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_video.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_objdetect.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_calib3d.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_features2d.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_ml.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_highgui.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_videoio.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_imgproc.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_flann.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_core.so.3.0.0
../lib/libGETKINECT.so: /usr/local/lib/libopencv_hal.a
../lib/libGETKINECT.so: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
../lib/libGETKINECT.so: /usr/lib/libvtkViews.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkInfovis.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkWidgets.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkVolumeRendering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkHybrid.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkParallel.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkRendering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkImaging.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkGraphics.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkIO.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkFiltering.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtkCommon.so.5.8.0
../lib/libGETKINECT.so: /usr/lib/libvtksys.so.5.8.0
../lib/libGETKINECT.so: src/cpp/CMakeFiles/GETKINECT.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../../../lib/libGETKINECT.so"
	cd /home/lukezhu/slam/try22-final1.0/build/src/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GETKINECT.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/cpp/CMakeFiles/GETKINECT.dir/build: ../lib/libGETKINECT.so
.PHONY : src/cpp/CMakeFiles/GETKINECT.dir/build

src/cpp/CMakeFiles/GETKINECT.dir/requires: src/cpp/CMakeFiles/GETKINECT.dir/Kinect_Input.cpp.o.requires
.PHONY : src/cpp/CMakeFiles/GETKINECT.dir/requires

src/cpp/CMakeFiles/GETKINECT.dir/clean:
	cd /home/lukezhu/slam/try22-final1.0/build/src/cpp && $(CMAKE_COMMAND) -P CMakeFiles/GETKINECT.dir/cmake_clean.cmake
.PHONY : src/cpp/CMakeFiles/GETKINECT.dir/clean

src/cpp/CMakeFiles/GETKINECT.dir/depend:
	cd /home/lukezhu/slam/try22-final1.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lukezhu/slam/try22-final1.0 /home/lukezhu/slam/try22-final1.0/src/cpp /home/lukezhu/slam/try22-final1.0/build /home/lukezhu/slam/try22-final1.0/build/src/cpp /home/lukezhu/slam/try22-final1.0/build/src/cpp/CMakeFiles/GETKINECT.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpp/CMakeFiles/GETKINECT.dir/depend
