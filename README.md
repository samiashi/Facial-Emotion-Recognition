# Facial-Emotion-Recognition

Project based on: https://github.com/TadasBaltrusaitis/OpenFace

Mac installation: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Mac-Installation

## Videos:

Parameters for input

-f <filename> the video file being input, can specify multiple files

-fd <depth directory> the directory where depth files are stored (deprecated)

-root <directory> the root of input and output so -f and -ov can be specified relative to it

-inroot <directory> the root of input so -f can be specified relative to it

-outroot <directory> the root of output so -ov can be specified relative to it

Parameters for output

-ov <location of visualized track> where to output video file with tracked landmarks

## Images

Single image analysis

-f <filename> the image file being input, can have multiple -f flags

-of <filename> location of output file for landmark points and action units

-op <filename> location of output file for 3D landmark points

-oi <filename> location of output image with landmarks

-root <dir> the root directory so -f, -of, -op, and -oi can be specified relative to it

-inroot <dir> the input root directory so -f can be specified relative to it

-outroot <dir> the root directory so -of, -op, and -oi can be specified relative to it

## /////////////////////////// Compile ////////////////////////////////////////

CMake:
	sudo cmake -D CMAKE_BUILD_TYPE=RELEASE .

To compile:
	sudo make

To force pull:
	sudo git fetch --all && sudo git reset --hard origin/master
		
## ///////////////////////////// Run //////////////////////////////////////////

cd ~/Desktop/Facial-Emotion-Recognition

To test Images:
	./bin/Image -f "./input/images/isk.jpg" -ofdir "./output/images/" -oidir "./output/images/"

To test Videos:
	./bin/Video -f "./input/videos/DRPHIL-02-01.mp4" -of "./output/videos/DRPHIL-02-01.xlsx" -ov "./output/videos/DRPHIL-02-01.mp4"

## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////