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
CMAKE_SOURCE_DIR = "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build"

# Include any dependencies generated for this target.
include src/cpp/CMakeFiles/SURFDETECT.dir/depend.make

# Include the progress variables for this target.
include src/cpp/CMakeFiles/SURFDETECT.dir/progress.make

# Include the compile flags for this target's objects.
include src/cpp/CMakeFiles/SURFDETECT.dir/flags.make

src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o: src/cpp/CMakeFiles/SURFDETECT.dir/flags.make
src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o: ../src/cpp/Detect.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build/CMakeFiles" $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o"
	cd "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build/src/cpp" && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/SURFDETECT.dir/Detect.cpp.o -c "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/src/cpp/Detect.cpp"

src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SURFDETECT.dir/Detect.cpp.i"
	cd "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build/src/cpp" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/src/cpp/Detect.cpp" > CMakeFiles/SURFDETECT.dir/Detect.cpp.i

src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SURFDETECT.dir/Detect.cpp.s"
	cd "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build/src/cpp" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/src/cpp/Detect.cpp" -o CMakeFiles/SURFDETECT.dir/Detect.cpp.s

src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o.requires:
.PHONY : src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o.requires

src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o.provides: src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o.requires
	$(MAKE) -f src/cpp/CMakeFiles/SURFDETECT.dir/build.make src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o.provides.build
.PHONY : src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o.provides

src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o.provides.build: src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o

# Object files for target SURFDETECT
SURFDETECT_OBJECTS = \
"CMakeFiles/SURFDETECT.dir/Detect.cpp.o"

# External object files for target SURFDETECT
SURFDETECT_EXTERNAL_OBJECTS =

../lib/libSURFDETECT.so: src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o
../lib/libSURFDETECT.so: src/cpp/CMakeFiles/SURFDETECT.dir/build.make
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_xphoto.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_ximgproc.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_tracking.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_text.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_surface_matching.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_stereo.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_saliency.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_rgbd.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_reg.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_optflow.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_line_descriptor.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_latentsvm.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_face.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_datasets.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_ccalib.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_bioinspired.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_bgsegm.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_adas.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_viz.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_videostab.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_videoio.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_video.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_superres.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_stitching.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_shape.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_photo.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_objdetect.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_ml.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_imgproc.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_highgui.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_hal.a
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_flann.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_features2d.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_core.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_calib3d.so.3.0.0
../lib/libSURFDETECT.so: ../lib/libGETKINECT.so
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_text.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_face.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_shape.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_video.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_objdetect.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_calib3d.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_features2d.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_ml.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_highgui.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_videoio.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_imgproc.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_flann.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_core.so.3.0.0
../lib/libSURFDETECT.so: /usr/local/lib/libopencv_hal.a
../lib/libSURFDETECT.so: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
../lib/libSURFDETECT.so: src/cpp/CMakeFiles/SURFDETECT.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../../../lib/libSURFDETECT.so"
	cd "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build/src/cpp" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SURFDETECT.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/cpp/CMakeFiles/SURFDETECT.dir/build: ../lib/libSURFDETECT.so
.PHONY : src/cpp/CMakeFiles/SURFDETECT.dir/build

src/cpp/CMakeFiles/SURFDETECT.dir/requires: src/cpp/CMakeFiles/SURFDETECT.dir/Detect.cpp.o.requires
.PHONY : src/cpp/CMakeFiles/SURFDETECT.dir/requires

src/cpp/CMakeFiles/SURFDETECT.dir/clean:
	cd "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build/src/cpp" && $(CMAKE_COMMAND) -P CMakeFiles/SURFDETECT.dir/cmake_clean.cmake
.PHONY : src/cpp/CMakeFiles/SURFDETECT.dir/clean

src/cpp/CMakeFiles/SURFDETECT.dir/depend:
	cd "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7" "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/src/cpp" "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build" "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build/src/cpp" "/home/lukezhu/Documents/Simultaneous localization and mapping/slamdiy/try7/build/src/cpp/CMakeFiles/SURFDETECT.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : src/cpp/CMakeFiles/SURFDETECT.dir/depend

