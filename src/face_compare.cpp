//
// Created by wayne on 24/12/18.
//
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<string.h>
#include<dirent.h>

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"


using namespace cv;
using namespace cv::face;
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

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 4) {
        cout << "usage: " << argv[0] << endl;
        cout << "\t </path/to/haar_cascade>. " << endl;
        cout << "\t </path/to/fisher_model>. " << endl;
        cout << "\t </path/to/test_photo>. " << endl;
        exit(1);
    }

    string haar_model = string(argv[1]);
    string fisher_model = string(argv[2]);


    Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
    model->read(fisher_model);

    CascadeClassifier haar_cascade;
    haar_cascade.load(haar_model);

    // 将待测试的4x2照片放进一个文件夹中, 全部都读出来,再给每一个标号就行了
    string strPath2Test = string(argv[3]);
    std::vector<std::string> filenames;

    getFileNames(strPath2Test, filenames);

    for(unsigned int j = 0; j < filenames.size(); j++)
    {
        // 检测人脸并用矩形框出来,而且要puttext

        Mat frame = imread(filenames[j]);
        Mat original = frame.clone();

        // cv::resize(original, original, cv::Size(480, 360), 0, 0, cv::INTER_LINEAR);

        Mat gray;
        cvtColor(original, gray, COLOR_BGR2GRAY);

        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);

        for(size_t i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];

            Mat face = gray(face_i);

            Mat face_resized;
            cv::resize(face, face_resized, Size(92, 112), 1.0, 1.0, INTER_CUBIC);
            int prediction = model->predict(face_resized);

            string output_info = ": others";
            if (prediction == 3) {
                output_info = ": wangxingbo";
            }
            if (prediction == 7) {
                output_info = ": suxuewei";
            }
	    if (prediction == 5) {
                output_info = ": lilongfei";
            }

            rectangle(original, face_i, Scalar(0, 255, 0), 1);
            // Create the text we will annotate the box with:
            string box_text = "Prediction" + output_info;

            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
  	    if (prediction == 3) {
                putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 6.0, Scalar(255, 255, 255), 3);
            }
	    else if (prediction == 7)
	    {
	       putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 3.0, Scalar(255, 255, 255), 3);
            }
            else if (prediction == 5)
	    {
	       putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 255, 255), 3);
            }
            else 
	    {
	       putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 255, 255), 3);
            }      
	    cv::resize(original, original, Size(480, 360), 1.0, 1.0, INTER_CUBIC);
            imshow(format("face_recognizer_%d", j), original);
        }
    }

    char key = (char) waitKey(0);


    return 0;
}
