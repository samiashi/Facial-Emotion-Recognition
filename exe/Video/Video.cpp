#include "LandmarkCoreIncludes.h"
// System includes
#include <fstream>
#include <sstream>
// Include OpenCV
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// Include OpenFace
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>
#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>

using namespace std;
using namespace boost::filesystem;

enum Column {
	frame,
	timestamp,
	AU01_i,
	AU02_i,
	AU04_i,
	AU05_i,
	AU06_i,
	AU07_i,
	AU09_i,
	AU10_i,
	AU12_i,
	AU14_i,
	AU15_i,
	AU17_i,
	AU20_i,
	AU23_i,
	AU25_i,
	AU26_i,
	AU28_i,
	AU45_i,
	AU01_o,
	AU02_o,
	AU04_o,
	AU05_o,
	AU06_o,
	AU07_o,
	AU09_o,
	AU10_o,
	AU12_o,
	AU14_o,
	AU15_o,
	AU17_o,
	AU20_o,
	AU23_o,
	AU25_o,
	AU26_o,
	AU28_o,
	AU45_o,
	NUM_COLUMNS
};

std::string columnTitle(int index) {
	switch(index)
	{
		case AU01_i: case AU01_o: return "AU01";
		case AU02_i: case AU02_o: return "AU02";
		case AU04_i: case AU04_o: return "AU04";
		case AU05_i: case AU05_o: return "AU05";
		case AU06_i: case AU06_o: return "AU06";
		case AU07_i: case AU07_o: return "AU07";
		case AU09_i: case AU09_o: return "AU09";
		case AU10_i: case AU10_o: return "AU10";
		case AU12_i: case AU12_o: return "AU12";
		case AU14_i: case AU14_o: return "AU14";
		case AU15_i: case AU15_o: return "AU15";
		case AU17_i: case AU17_o: return "AU17";
		case AU20_i: case AU20_o: return "AU20";
		case AU23_i: case AU23_o: return "AU23";
		case AU25_i: case AU25_o: return "AU25";
		case AU26_i: case AU26_o: return "AU26";
		case AU28_i: case AU28_o: return "AU28";
		case AU45_i: case AU45_o: return "AU45";
		default:
			std::cout << "Error: Unknown column index\n";
	}
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

vector<string> get_arguments(int argc, char **argv);
void visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, int frame_count);
void prepareOutputFile(std::ofstream* output_file, bool output_AUs, vector<string> au_names_class, vector<string> au_names_reg);
void outputAllFeatures(std::ofstream* output_file, bool output_AUs, int frame_count, double time_stamp, const FaceAnalysis::FaceAnalyser& face_analyser);
void analyseAU(string outfeatures);

int main (int argc, char **argv)
{
	vector<string> arguments = get_arguments(argc, argv);

	// Search paths
	boost::filesystem::path config_path = boost::filesystem::path("~");
	boost::filesystem::path parent_path = boost::filesystem::path(arguments[0]).parent_path();

	// Some initial parameters that can be overriden from command line
	vector<string> input_files, depth_directories, output_files, tracked_videos_output;

	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// Indicates that rotation should be with respect to camera or world coordinates
	bool use_world_coordinates;
	string output_codec; //not used but should
	LandmarkDetector::get_video_input_output_params(input_files, depth_directories, output_files, tracked_videos_output, use_world_coordinates, output_codec, arguments);

	bool verbose = true;

	// The modules that are being used for tracking
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	vector<string> output_similarity_align;
	vector<string> output_hog_align_files;

	int f_n = -1; // File Number
	double sim_scale = -1;
	int sim_size = 112;
	bool grayscale = true;
	bool dynamic = true; // Indicates if a dynamic AU model should be used (dynamic is useful if the video is long enough to include neutral expressions)
	int num_hog_rows;
	int num_hog_cols;

	// By default output all parameters, but these can be turned off to get smaller files or slightly faster processing times
	bool output_2D_landmarks = true;
	bool output_model_params = true;
	bool output_AUs = true;
	bool done = false;

	// Used for image masking
	string tri_loc;
	boost::filesystem::path tri_loc_path = boost::filesystem::path("model/tris_68_full.txt");
	if (boost::filesystem::exists(tri_loc_path))
	{
		tri_loc = tri_loc_path.string();
	}
	else if (boost::filesystem::exists(parent_path/tri_loc_path))
	{
		tri_loc = (parent_path/tri_loc_path).string();
	}
	else if (boost::filesystem::exists(config_path/tri_loc_path))
	{
		tri_loc = (config_path/tri_loc_path).string();
	}
	else
	{
		cout << "Can't find triangulation files, exiting" << endl;
		return 1;
	}

	// Will warp to scaled mean shape
	cv::Mat_<double> similarity_normalised_shape = face_model.pdm.mean_shape * sim_scale;
	// Discard the z component
	similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

	string au_loc;
	string au_loc_local;
	if (dynamic)
	{
		au_loc_local = "AU_predictors/AU_all_best.txt";
	}
	else
	{
		au_loc_local = "AU_predictors/AU_all_static.txt";
	}

	boost::filesystem::path au_loc_path = boost::filesystem::path(au_loc_local);
	if (boost::filesystem::exists(au_loc_path))
	{
		au_loc = au_loc_path.string();
	}
	else if (boost::filesystem::exists(parent_path/au_loc_path))
	{
		au_loc = (parent_path/au_loc_path).string();
	}
	else if (boost::filesystem::exists(config_path/au_loc_path))
	{
		au_loc = (config_path/au_loc_path).string();
	}
	else
	{
		cout << "Can't find AU prediction files, exiting" << endl;
		return 1;
	}

	// Creating a  face analyser that will be used for AU extraction
	// Make sure sim_scale is proportional to sim_size if not set
	if (sim_scale == -1) sim_scale = sim_size * (0.7 / 112.0);

	FaceAnalysis::FaceAnalyser face_analyser(vector<cv::Vec3d>(), sim_scale, sim_size, sim_size, au_loc, tri_loc);

	while(!done)
	{
		string current_file;

		cv::VideoCapture video_capture;

		cv::Mat captured_image;
		int total_frames = -1;
		int reported_completion = 0;

		double fps_vid_in = -1.0;

		// We might specify multiple video files as arguments
		if(input_files.size() > 0)
		{
			f_n++;
			current_file = input_files[f_n];
		}

		// Do some grabbing
		if(current_file.size() > 0)
		{
			std::cout << "Attempting to read from file: " << current_file << std::endl;
			video_capture = cv::VideoCapture(current_file);
			total_frames = (int)video_capture.get(CV_CAP_PROP_FRAME_COUNT);
			cout << "Total Frames: " << total_frames << endl;
			fps_vid_in = video_capture.get(CV_CAP_PROP_FPS);

			// Check if fps is nan or less than 0
			if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
			{
			    std::cout << "FPS of the video file cannot be determined, assuming 30" << std::endl;
				fps_vid_in = 30;
			}
		}

		if (!video_capture.isOpened())
		{
			std::cout << "Fatal error: Failed to open video source, exiting" << std::endl;
			return 1;
		}
		else
		{
			std::cout << "File opened" << std::endl;
		}

		video_capture >> captured_image;

		// Creating output files
		std::ofstream output_file;

		if (!output_files.empty())
		{
			output_file.open(output_files[f_n], ios_base::out);
            prepareOutputFile(&output_file, output_AUs, face_analyser.GetAUClassNames(), face_analyser.GetAURegNames());
		}

		// saving the videos
		cv::VideoWriter writerFace;
		if(!tracked_videos_output.empty())
		{
			try
			{
				writerFace = cv::VideoWriter(tracked_videos_output[f_n], CV_FOURCC(output_codec[0],output_codec[1],output_codec[2],output_codec[3]), fps_vid_in, captured_image.size(), true);
			}
			catch(cv::Exception e)
			{
				std::cout << "Warning: Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << std::endl;
			}
		}

		int frame_count = 0;

		// This is useful for a second pass run (if want AU predictions)
		vector<cv::Vec6d> params_global_video;
		vector<bool> successes_video;
		vector<cv::Mat_<double>> params_local_video;
		vector<cv::Mat_<double>> detected_landmarks_video;

		// Use for timestamping if using a webcam
		int64 t_initial = cv::getTickCount();

		bool visualise_hog = verbose;

		// Timestamp in seconds of current processing
		double time_stamp = 0;

		std::cout << "Starting tracking" << std::endl;

		while(!captured_image.empty()) {
			// Grab the timestamp first
			time_stamp = (double)frame_count * (1.0 / fps_vid_in);

			// Reading the images
			cv::Mat_<uchar> grayscale_image;

			if(captured_image.channels() == 3)
			{
				cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
			}
			else
			{
				grayscale_image = captured_image.clone();
			}

			// The actual facial landmark detection / tracking
			bool detection_success;
			detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);

			// Do face alignment
			cv::Mat sim_warped_img;
			cv::Mat_<double> hog_descriptor;

			// But only if needed in output
			if(!output_similarity_align.empty() || output_AUs)
			{
				face_analyser.AddNextFrame(captured_image, face_model, time_stamp, false, !det_parameters.quiet_mode);
				face_analyser.GetLatestAlignedFace(sim_warped_img);

				if(!det_parameters.quiet_mode)
				{
					cv::imshow("sim_warp", sim_warped_img);
				}
			}

			// Write the similarity normalised output
			if (!output_similarity_align.empty())
			{

				if (sim_warped_img.channels() == 3 && grayscale)
				{
					cvtColor(sim_warped_img, sim_warped_img, CV_BGR2GRAY);
				}

				char name[100];

				// Filename is based on frame number
				std::sprintf(name, "frame_det_%06d.bmp", frame_count + 1);

				// Construct the output filename
				boost::filesystem::path slash("/");

				std::string preferredSlash = slash.make_preferred().string();

				string out_file = output_similarity_align[f_n] + preferredSlash + string(name);
				bool write_success = imwrite(out_file, sim_warped_img);

				if (!write_success)
				{
					cout << "Could not output similarity aligned image" << endl;
					return 1;
				}
			}

			// Visualising the tracker
			visualise_tracking(captured_image, face_model, det_parameters, frame_count);

			// Output the AUs
            outputAllFeatures(&output_file, output_AUs, frame_count, time_stamp, face_analyser);

			// output the tracked video
			writerFace << captured_image;
			video_capture >> captured_image;

			// Update the frame count
			frame_count++;

			if(total_frames != -1)
			{
				if((double)frame_count/(double)total_frames >= reported_completion / 10.0)
				{
					cout << reported_completion * 10 << "% ";
					reported_completion = reported_completion + 1;
				}
			}
		}

		output_file.close();

		if (output_files.size() > 0 && output_AUs)
		{
			cout << "\nPostprocessing the Action Unit predictions" << endl;
			face_analyser.PostprocessOutputFile(output_files[f_n], dynamic);
		}
		// Reset the models for the next video
		face_analyser.Reset();
		face_model.Reset();

		frame_count = 0;

		if (total_frames != -1)
		{
			cout << endl;
		}

		// break out of the loop if done with all the files
		if(f_n == input_files.size() -1)
		{
			done = true;
		}
	}

	//cout << "\n\nOutputFile: " << output_files[f_n] << endl;
	analyseAU(output_files[f_n]);
	return 0;
}

vector<string> get_arguments(int argc, char **argv)
{
	vector<string> arguments;

	// First argument is reserved for the name of the executable
	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, int frame_count)
{
	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);
	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);

	if (!det_parameters.quiet_mode)
	{
		cv::namedWindow("tracking_result", 1);
		cv::imshow("tracking_result", captured_image);
	}
}

void prepareOutputFile(std::ofstream* output_file, bool output_AUs, vector<string> au_names_class, vector<string> au_names_reg)
{
	*output_file << "frame, timestamp";

	if (output_AUs)
	{
        // AU intensities
		std::sort(au_names_reg.begin(), au_names_reg.end());
		for (string reg_name : au_names_reg)
		{
			*output_file << ", " << reg_name << "_i";
		}
        // AU occurrences
		std::sort(au_names_class.begin(), au_names_class.end());
		for (string class_name : au_names_class)
		{
			*output_file << ", " << class_name << "_o";
		}
	}
	*output_file << endl;
}

// Output all of the information into one file in one go (quite a few parameters, but simplifies the flow)
void outputAllFeatures(std::ofstream* output_file, bool output_AUs, int frame_count, double time_stamp, const FaceAnalysis::FaceAnalyser& face_analyser)
{
	*output_file << frame_count + 1 << ", " << time_stamp;

	if (output_AUs)
	{
		auto aus_reg = face_analyser.GetCurrentAUsReg();

		vector<string> au_reg_names = face_analyser.GetAURegNames();
		std::sort(au_reg_names.begin(), au_reg_names.end());

		// write out ar the correct index
		for (string au_name : au_reg_names)
		{
			for (auto au_reg : aus_reg)
			{
				if (au_name.compare(au_reg.first) == 0)
				{
					*output_file << ", " << au_reg.second;
					break;
				}
			}
		}

		if (aus_reg.size() == 0)
		{
			for (size_t p = 0; p < face_analyser.GetAURegNames().size(); ++p)
			{
				*output_file << ", 0";
			}
		}

		auto aus_class = face_analyser.GetCurrentAUsClass();

		vector<string> au_class_names = face_analyser.GetAUClassNames();
		std::sort(au_class_names.begin(), au_class_names.end());

		// write out ar the correct index
		for (string au_name : au_class_names)
		{
			for (auto au_class : aus_class)
			{
				if (au_name.compare(au_class.first) == 0)
				{
					*output_file << ", " << au_class.second;
					break;
				}
			}
		}

		if (aus_class.size() == 0)
		{
			for (size_t p = 0; p < face_analyser.GetAUClassNames().size(); ++p)
			{
				*output_file << ", 0";
			}
		}
	}
	*output_file << endl;
}

void analyseAU(string outfeatures)
{
	// std::cout << "AU file: " << outfeatures << std::endl;
	std::ifstream infile(outfeatures);

	if(infile.is_open())
	{
		std::string line;
		std::vector<std::string> writeLines;
		std::getline(infile, line);
		while(std::getline(infile, line))
		{
			std::vector<double> row;
			row.resize(NUM_COLUMNS);
			std::istringstream iss(line);
			int i = 0;
			char discard;
			iss >> row[i++];
			while (iss >> discard >> row[i++]) {}
			std::vector<std::string> action_units;
			for (i = AU01_i; i <= AU45_i; i++)
			{
				if (row[i] >= 0.5 /*&& row[i + 18] == 1*/)
					action_units.push_back(columnTitle(i));
			}
			if (!action_units.size()) continue;

			vector<string> emotions;
			for(int i(0); i < int(action_units.size()); i++)
			{
				// cout << "Action Units: " << action_units[i] << endl;
				if(action_units[i].compare("AU01") == 0)
				{
					emotions.push_back("Fear");
					emotions.push_back("Sad");
					emotions.push_back("Surprise");
				}
				else if(action_units[i].compare("AU02") == 0)
				{
					emotions.push_back("Surprise");
					emotions.push_back("Angry");
				}
				else if(action_units[i].compare("AU04") == 0)
				{
					emotions.push_back("Fear");
					emotions.push_back("Sad");
					emotions.push_back("Disgust");
					emotions.push_back("Angry");
				}
				else if(action_units[i].compare("AU05") == 0)
				{
					emotions.push_back("Fear");
					emotions.push_back("Surprise");
				}
				else if(action_units[i].compare("AU06") == 0)
				{
					emotions.push_back("Sad");
					emotions.push_back("Happy");
					emotions.push_back("Disgust");
				}
				else if(action_units[i].compare("AU07") == 0)
				{
					emotions.push_back("Fear");
					emotions.push_back("Angry");
					emotions.push_back("Disgust");
				}
				else if(action_units[i].compare("AU09") == 0)
				{
					emotions.push_back("Disgust");
				}
				else if(action_units[i].compare("AU10") == 0)
				{
					emotions.push_back("Sad");
					emotions.push_back("Happy");
				}
				else if(action_units[i].compare("AU11") == 0)
				{
					emotions.push_back("Sad");
				}
				else if(action_units[i].compare("AU12") == 0)
				{
					emotions.push_back("Happy");
				}
				else if(action_units[i].compare("AU15") == 0)
				{
					emotions.push_back("Sad");
				}
				else if(action_units[i].compare("AU17") == 0)
				{
					emotions.push_back("Angry");
					emotions.push_back("Sad");
					emotions.push_back("Disgust");
				}
				else if(action_units[i].compare("AU20") == 0)
				{
					emotions.push_back("Fear");
				}
				else if(action_units[i].compare("AU23") == 0)
				{
					emotions.push_back("Happy");
					emotions.push_back("Angry");
				}
				else if(action_units[i].compare("AU25") == 0)
				{
					emotions.push_back("Fear");
				}
				else if(action_units[i].compare("AU26") == 0)
				{
					emotions.push_back("Happy");
					emotions.push_back("Surprise");
				}
				else if(action_units[i].compare("AU27") == 0)
				{
					emotions.push_back("Surprise");
				}
			}

			string emotion;
			map<string, int> m;
			// count occurrences of every string
			for(int i(0); i < emotions.size(); i++)
		    {
				map<string, int>::iterator it = m.find(emotions[i]);

				if(it == m.end())
					m.insert(pair<string, int>(emotions[i], 1));
				else
		    		m[emotions[i]] += 1;
		     }

			// find the max
			map<string, int>::iterator it = m.begin();
			for(map<string, int>::iterator it2 = m.begin(); it2 != m.end(); ++it2)
			{
				if(it2 -> second >= it -> second)
				{
					if(int(it2 -> second) >= 3)
					{
						it = it2;
						emotion = it -> first;
					}
					else
					{
						emotion = "Neutral";
					}
				}
			}
			// cout << "Emotion: " << emotion << endl;
			writeLines.push_back("\n" + std::to_string(row[frame]) + ", " + std::to_string(row[timestamp]) + ", " + emotion);
		}
		infile.close();

		std::ofstream myfile;
		myfile.open(outfeatures);
		myfile << "frame, timestamp, emotion";
		for (int i = 0; i < writeLines.size(); i++)
		{
			myfile << writeLines[i];
		}
		myfile.close();
	}
	else
	{
		perror("File Open");
	}
}
