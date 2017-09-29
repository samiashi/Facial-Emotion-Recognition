#include "LandmarkCoreIncludes.h"
// System includes
#include <fstream>
#include <ostream>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
// Include OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Include OpenFace
#include <filesystem.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <tbb/tbb.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>

using namespace std;

vector<string> get_arguments(int argc, char **argv);
void convert_to_grayscale(const cv::Mat& in, cv::Mat& out);
void write_out_landmarks(const string& outfeatures, const LandmarkDetector::CLNF& clnf_model, std::vector<std::pair<std::string, double>> au_intensities, std::vector<std::pair<std::string, double>> au_occurences);
void create_display_image(const cv::Mat& orig, cv::Mat& display_image, LandmarkDetector::CLNF& clnf_model);
void analyseAU(string outfeatures);

int main (int argc, char **argv)
{
	//Convert arguments to more convenient vector form
	vector<string> arguments = get_arguments(argc, argv);

	// Search paths
	boost::filesystem::path config_path = boost::filesystem::path("~");
	boost::filesystem::path parent_path = boost::filesystem::path(arguments[0]).parent_path();

	// Some initial parameters that can be overriden from command line
	vector<string> files, depth_files, output_images, output_landmark_locations, null;
	vector<cv::Rect_<double>> null1;

	LandmarkDetector::get_image_input_output_params(files, depth_files, output_landmark_locations, null, output_images, null1, arguments);
	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	// No need to validate detections, as we're not doing tracking
	det_parameters.validate_detections = false;

	// The modules that are being used for tracking
	cout << "Loading the model" << endl;
	LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
	cout << "Model loaded" << endl;

	cv::CascadeClassifier classifier(det_parameters.face_detector_location);
	dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();

	// Loading the AU prediction models
	string au_loc = "AU_predictors/AU_all_static.txt";

	boost::filesystem::path au_loc_path = boost::filesystem::path(au_loc);
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

	// Used for image masking for AUs
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

	FaceAnalysis::FaceAnalyser face_analyser(vector<cv::Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);

	bool visualise = !det_parameters.quiet_mode;

	// Do some image loading
	for(size_t i = 0; i < files.size(); i++)
	{
		string file = files.at(i);

		// Loading image
		cv::Mat read_image = cv::imread(file, -1);

		if (read_image.empty())
		{
			cout << "Could not read the input image" << endl;
			return 1;
		}

		// Loading depth file if exists (optional)
		cv::Mat_<float> depth_image;

		if(depth_files.size() > 0)
		{
			string dFile = depth_files.at(i);
			cv::Mat dTemp = cv::imread(dFile, -1);
			dTemp.convertTo(depth_image, CV_32F);
		}

		// Making sure the image is in uchar grayscale
		cv::Mat_<uchar> grayscale_image;
		convert_to_grayscale(read_image, grayscale_image);

		// Use center of image
		float cx = grayscale_image.cols / 2.0f;
		float cy = grayscale_image.rows / 2.0f;
		// Use a rough guess-timate of focal length
		float fx = 500 * (grayscale_image.cols / 640.0);
		float fy = 500 * (grayscale_image.rows / 480.0);

		fx = (fx + fy) / 2.0;
		fy = fx;

		// Detect faces in an image
		vector<cv::Rect_<double> > face_detections;

		if(det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
		{
			vector<double> confidences;
			LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);
		}
		else
		{
			LandmarkDetector::DetectFaces(face_detections, grayscale_image, classifier);
		}

		// Detect landmarks around detected faces
		int face_det = 0;
		// perform landmark detection for every face detected
		for(size_t face=0; face < face_detections.size(); ++face)
		{
			// if there are multiple detections go through them
			bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, face_detections[face], clnf_model, det_parameters);

			auto ActionUnits = face_analyser.PredictStaticAUs(read_image, clnf_model, false);

			// Writing out the detected landmarks (in an OS independent manner)
			char name[100];
			// append detection number (in case multiple faces are detected)
			sprintf(name, "_det_%d", face_det);

			// Construct the output filename
			boost::filesystem::path slash("/");
			std::string preferredSlash = slash.make_preferred().string();

			boost::filesystem::path out_feat_path(output_landmark_locations.at(i));
			boost::filesystem::path dir = out_feat_path.parent_path();
			boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
			boost::filesystem::path ext = out_feat_path.extension();
			string outfeatures = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
			write_out_landmarks(outfeatures, clnf_model, ActionUnits.first, ActionUnits.second);

			// displaying detected landmarks
			cv::Mat display_image;
			create_display_image(read_image, display_image, clnf_model);

			// Saving the display images (in an OS independent manner)
			string outimage = output_images.at(i);
			// append detection number
			boost::filesystem::path outFeat_path(outimage);
			boost::filesystem::path imageDir = outFeat_path.parent_path();
			boost::filesystem::path imageName = outFeat_path.filename().replace_extension("");
			boost::filesystem::path imageExt = outFeat_path.extension();
			outimage = imageDir.string() + preferredSlash + imageName.string() + string(name) + imageExt.string();
			bool write_success = cv::imwrite(outimage, display_image);

			if (!write_success)
			{
				cout << "Could not output a processed image" << endl;
				return 1;
			}

			analyseAU(outfeatures);

			if(success)
			{
				face_det++;
			}
		}
	}
	return 0;
}

vector<string> get_arguments(int argc, char **argv)
{
	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void convert_to_grayscale(const cv::Mat& in, cv::Mat& out)
{
	if(in.channels() == 3)
	{
		// Make sure it's in a correct format
		if(in.depth() != CV_8U)
		{
			if(in.depth() == CV_16U)
			{
				cv::Mat tmp = in / 256;
				tmp.convertTo(tmp, CV_8U);
				cv::cvtColor(tmp, out, CV_BGR2GRAY);
			}
		}
		else
		{
			cv::cvtColor(in, out, CV_BGR2GRAY);
		}
	}
	else if(in.channels() == 4)
	{
		cv::cvtColor(in, out, CV_BGRA2GRAY);
	}
	else
	{
		if(in.depth() == CV_16U)
		{
			cv::Mat tmp = in / 256;
			out = tmp.clone();
		}
		else if(in.depth() != CV_8U)
		{
			in.convertTo(out, CV_8U);
		}
		else
		{
			out = in.clone();
		}
	}
}

void write_out_landmarks(const string& outfeatures, const LandmarkDetector::CLNF& clnf_model, std::vector<std::pair<std::string, double>> au_intensities, std::vector<std::pair<std::string, double>> au_occurences)
{
	std::ofstream featuresFile;
	featuresFile.open(outfeatures);

	if (featuresFile.is_open())
	{
		int n = clnf_model.patch_experts.visibilities[0][0].rows;

		// Do the au intensities
		featuresFile << "au_intensities {"  << endl;
		for (size_t i = 0; i < au_intensities.size(); ++i)
		{
			// Use matlab format, so + 1
			featuresFile << au_intensities[i].first << " " << au_intensities[i].second << endl;
		}
		featuresFile << "}" << endl;

		// Do the au occurrences
		featuresFile << "au_occurrences {" <<  endl;
		for (size_t i = 0; i < au_occurences.size(); ++i)
		{
			// Use matlab format, so + 1
			featuresFile << au_occurences[i].first << " " << au_occurences[i].second << endl;
		}
		featuresFile << "}" << endl;

		featuresFile.close();
	}
}

void create_display_image(const cv::Mat& orig, cv::Mat& display_image, LandmarkDetector::CLNF& clnf_model)
{
	// preparing the visualisation image
	display_image = orig.clone();

	// Creating a display image
	cv::Mat xs = clnf_model.detected_landmarks(cv::Rect(0, 0, 1, clnf_model.detected_landmarks.rows/2));
	cv::Mat ys = clnf_model.detected_landmarks(cv::Rect(0, clnf_model.detected_landmarks.rows/2, 1, clnf_model.detected_landmarks.rows/2));
	double min_x, max_x, min_y, max_y;

	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);

	double width = max_x - min_x;
	double height = max_y - min_y;

	int minCropX = max((int)(min_x-width/3.0),0);
	int minCropY = max((int)(min_y-height),0);

	int widthCrop = min((int)(width*5.0/3.0), display_image.cols - minCropX - 1);
	int heightCrop = min((int)(height*5.0/2.0), display_image.rows - minCropY - 1);

	double scaling = 350.0/widthCrop;

	// first crop the image
	display_image = display_image(cv::Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));

	// now scale it
	cv::resize(display_image.clone(), display_image, cv::Size(), scaling, scaling);

	// Make the adjustments to points
	xs = (xs - minCropX)*scaling;
	ys = (ys - minCropY)*scaling;

	cv::Mat shape = clnf_model.detected_landmarks.clone();

	xs.copyTo(shape(cv::Rect(0, 0, 1, clnf_model.detected_landmarks.rows/2)));
	ys.copyTo(shape(cv::Rect(0, clnf_model.detected_landmarks.rows/2, 1, clnf_model.detected_landmarks.rows/2)));

	// Do the shifting for the hierarchical models as well
	for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
	{
		cv::Mat xs = clnf_model.hierarchical_models[part].detected_landmarks(cv::Rect(0, 0, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2));
		cv::Mat ys = clnf_model.hierarchical_models[part].detected_landmarks(cv::Rect(0, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2));

		xs = (xs - minCropX)*scaling;
		ys = (ys - minCropY)*scaling;

		cv::Mat shape = clnf_model.hierarchical_models[part].detected_landmarks.clone();

		xs.copyTo(shape(cv::Rect(0, 0, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2)));
		ys.copyTo(shape(cv::Rect(0, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2)));
	}

	LandmarkDetector::Draw(display_image, clnf_model);
}

void analyseAU(string outfeatures)
{
	std::cout << "AU file: " << outfeatures << std::endl;

	std::ifstream file(outfeatures);
	std::string str, str1, str2;
	std::vector<string> tokens, str_intensities, str_occurrences;
	std::vector<float> flt_intensities, flt_occurrences;
	std::string::size_type sz;

	while(std::getline(file, str))
	{
		if(str.compare("au_intensities {") == 0)
		{
			while(std::getline(file, str1))
			{
				if(str1.compare("}") == 0)
					break;
				split(tokens, str1, boost::is_any_of(" "));
				str_intensities.push_back(tokens[0]);
				float flt_number = std::stof(tokens[1], &sz);
				flt_intensities.push_back(flt_number);
				tokens.clear();
			}
		}
		if(str.compare("au_occurrences {") == 0)
		{
			while(std::getline(file, str2))
			{
				if(str2.compare("}") == 0)
					break;
				split(tokens, str2, boost::is_any_of(" "));
				str_occurrences.push_back(tokens[0]);
				float flt_number = std::stof(tokens[1], &sz);
				flt_occurrences.push_back(flt_number);
				tokens.clear();
			}
		}
	}
	std::vector<string> action_units;
	for(int i(0); i < int(str_occurrences.size()); i++)
	{
		for(int h(0); h < int(str_intensities.size()); h++)
		{
			if(str_occurrences[i].compare(str_intensities[h]) == 0 /*&& flt_occurrences[i] == 1*/ && flt_intensities[h] >= 0.5)
			{
				action_units.push_back(str_occurrences[i]);
			}
		}
	}
	vector<string> emotions;
	for(int i(0); i < int(action_units.size()); i++)
	{
		cout << "Action Units: " << action_units[i] << endl;
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
	cout << "Emotion: " << emotion << endl;

	ofstream myfile;
	myfile.open(outfeatures);
	myfile << "Emotion: " << emotion;
	myfile.close();
}
