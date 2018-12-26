#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<string.h>
#include<dirent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

void getFileNames(const std::string path, std::vector<std::string>& filenames, const std::string suffix = "bmp")
{
    DIR *pDir;
    struct dirent* ptr;

    if (!(pDir = opendir(path.c_str())))
        return;

    while ((ptr = readdir(pDir))!=0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            std::string file = path  + ptr->d_name;

            if (opendir(file.c_str()))
            {
                getFileNames(file, filenames, suffix);
            }
            else
            {
                if (suffix == file.substr(file.size() - suffix.size()))
                {
                    filenames.push_back(file);
                }
            }
        }
    }
    closedir(pDir);
}

vector< Mat > detectAndSave(Mat& frame, CascadeClassifier& cascade )
{
    Mat original = frame.clone();
    cv::resize(original, original, cv::Size(480, 360), 0, 0, cv::INTER_LINEAR);

    // Convert the current frame to grayscale:
    Mat gray;
    cvtColor(original, gray, COLOR_BGR2GRAY);
    // Find the faces in the frame:
    vector< Rect_<int> > faces;
    vector< Mat > faces_save;

    cascade.detectMultiScale(gray, faces);

    for(size_t i = 0; i < faces.size(); i++) {
        // Process face by face:
        Rect face_i = faces[i];
        // Crop the face from the image. So simple with OpenCV C++:
        Mat face = gray(face_i);

        Mat face_resized;
        cv::resize(face, face_resized, Size(92, 112), 1.0, 1.0, INTER_CUBIC);
        faces_save.push_back(face_resized);
    }
    return faces_save;
}



int main(int argc, char *argv[])
{
    CascadeClassifier cascade;

    string haar_model = string(argv[1]);
    
    cascade.load(haar_model);


	string strPathToPhoto = string(argv[2]);
	std::vector<std::string> filenames;

	getFileNames(strPathToPhoto, filenames);

	cout << filenames.size() << endl;

	unsigned int j = 100;
	for(unsigned int i = 0; i < filenames.size(); i++)
	{
	    cout << filenames[i] << endl;
	    cout << j << endl;
	    Mat frame;
	    vector< Mat > frameSaved;
	    frame = imread(filenames[i]);

	    frameSaved = detectAndSave(frame, cascade);
	    cout << frameSaved.size() << endl;
	    for(unsigned int i = 0; i < frameSaved.size(); i++)
	    {
		Mat frameOut = frameSaved[i];

		stringstream ss;
		ss << j;
		string s_j;
		ss >> s_j;

		string filenameOut;
		filenameOut = strPathToPhoto + s_j + ".bmp";

		imwrite(filenameOut, frameOut);
		j++;
	    }
	}

}
