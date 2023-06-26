//
// Project 3: Real-time 2-D Object Recognition
//
// created by Kshama Dhaduti 
//
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int getint(FILE* fp, int* v);
int getfloat(FILE* fp, float* v);
int getstring(FILE* fp, char os[]);

int append_image_data_csv(char* filename, char* image_filename, std::vector<float>& image_data, int reset_file);
int read_image_data_csv(char* filename, std::vector<char*>& filenames, std::vector<std::vector<float>>& data, int echo_file);

//----------------------------------------------------------------------------
// 
//Calculating Scaled Euclidean Distance
float scaled_edist(const std::vector<float>& vector_x, const std::vector<float>& vector_y, const std::vector<float>& std_dev) {
    float a = 0.0;
    for (int i = 0; i < vector_x.size(); i++) {
        float scale = (vector_x[i] - vector_y[i]) / std_dev[i];
        a += scale * scale;
    }
    return std::sqrt(a);
}

//2 pass algorithm for segmentation
void Label(Mat& B_img, Mat& L_img)
{
    
    Mat Label = Mat::zeros(B_img.size(), CV_32SC1);
    int Count = 1;
    int offset[] = { -1, 0, 1 };

    for (int i = 0; i < B_img.rows; i++)
    {
        for (int j = 0; j < B_img.cols; j++)
        {
            if (B_img.at<uchar>(i, j) == 255)
            {
                vector<int> neighbor;
                for (int m = 0; m < 3; m++)
                {
                    for (int n = 0; n < 3; n++)
                    {
                        int row = i + offset[m];
                        int col = j + offset[n];
                        if (row >= 0 && col >= 0 && row < B_img.rows && col < B_img.cols)
                        {
                            int Label_neighbor = Label.at<int>(row, col);
                            if (Label_neighbor > 0)
                            {
                                neighbor.push_back(Label_neighbor);
                            }
                        }
                    }
                }
                if (neighbor.empty())
                {
                    Label.at<int>(i, j) = Count++;
                }
                else
                {
                    int Neighbor_min = *min_element(neighbor.begin(), neighbor.end());
                    Label.at<int>(i, j) = Neighbor_min;
                    for (auto neighbor : neighbor)
                    {
                        if (neighbor != Neighbor_min)
                        {
                            for (int m = 0; m < B_img.rows; m++)
                            {
                                for (int n = 0; n < B_img.cols; n++)
                                {
                                    if (Label.at<int>(m, n) == neighbor)
                                    {
                                        Label.at<int>(m, n) = Neighbor_min;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    vector<Vec3b> colour(Count);
    for (int i = 0; i < Count; i++)
    {
        colour[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }
    L_img = cv::Mat::zeros(B_img.size(), CV_8UC3);
    for (int i = 0; i < B_img.rows; i++)
    {
        for (int j = 0; j < B_img.cols; j++)
        {
            int label = Label.at<int>(i, j);
            L_img.at<Vec3b>(i, j) = colour[label];
        }
    }
}
//----------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    //File creation and naming objects
    char filename[] = "data_base.csv";
    char image_filename[] = "knief";
    char input = 1;
    //vector<char*>objnames;
    //vector<std::vector<float>>data;

    //Background subtraction
    Ptr<BackgroundSubtractor> bg_Sub;
    bg_Sub = createBackgroundSubtractorMOG2();

    // Replace the IP address and port number with that of your phone's camera
    std::string ip_address = "192.168.0.130", port_number = "8080";
    Mat S_img;
    Mat C_img;
    Mat Frame, fg_mask;
    double Hu_Moments[7];

    // Video capdev initated 
    VideoCapture capdev("http://" + ip_address + ":" + port_number + "/video");
    if (!capdev.isOpened()) {

        std::cout << "Error opening camera stream" << std::endl;
        return -1;
    }
    cout << "1. Training 2. Classifier :";
    cin >> input;
    while (true)
    {
        while (input == '1')
        {
            cout << "\nLabel Object as:";
            cin >> image_filename;
            while (true) {
                capdev >> Frame;
                if (Frame.empty())
                    break;
//------------------------------------------------------------------------------------
// 
                //Task 1:Threshold the input video
                
                bg_Sub->apply(Frame, fg_mask, 0);
                rectangle(Frame, cv::Point(10, 2), cv::Point(100, 20),
                    cv::Scalar(255, 255, 255), -1);
                stringstream ss;
                ss << capdev.get(CAP_PROP_POS_FRAMES);
                string frameNumberString = ss.str();
                putText(Frame, frameNumberString.c_str(), cv::Point(15, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                imshow("Frame", Frame);
                imshow("FG Mask", fg_mask); //foreground mask

                int keyboard = waitKey(10); //input request
                if (keyboard == 'q' || keyboard == 27)
                    break;
            }
            while (true)
            {
                vector<float> object_features(0);
                capdev >> Frame;
                if (Frame.empty())
                    break;

                //Task 1:Threshold the input video
                
                bg_Sub->apply(Frame, fg_mask, 0);
                rectangle(Frame, cv::Point(10, 2), cv::Point(100, 20),
                    cv::Scalar(255, 255, 255), -1);
                stringstream ss;
                ss << capdev.get(CAP_PROP_POS_FRAMES);
                string frameNumberString = ss.str();
                putText(Frame, frameNumberString.c_str(), cv::Point(15, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                imshow("Frame", Frame);
                imshow("FG Mask", fg_mask);

                //Task2:Clean up the binary image
                //Using morphological operations to clean_up

                Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

                //noise elimination > dialation & erosion
                morphologyEx(fg_mask, C_img, MORPH_OPEN, kernel, Point(-1, -1), 1); 
                morphologyEx(C_img, C_img, MORPH_CLOSE, kernel, Point(-1, -1), 3); 
                morphologyEx(C_img, C_img, MORPH_ERODE, kernel, Point(-1, -1), 1); 

                //Resultant image
                imshow("Cleaned Image", C_img);

                //Task 3:Segmenting the regions using 2-pass algorithm
                //Mat S_img;
                Label(C_img, S_img);
                
                //imshow("Segmented Image", S_img); uncomment to get output for Task 3

                //Task 4 : Compute features for each major region
                // 
                // Countour detection 
                RNG random_no_gen(12345);
                vector<vector<Point>> contours;
                findContours(C_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

                //Largest area
                double area_max = -1;
                int index = -1;
                for (int i = 0; i < contours.size(); i++)
                {
                    double area = contourArea(contours[i]);
                    if (area > area_max)
                    {
                        area_max = area;
                        index = i;
                    }
                }
                //Bounding Box 
                Mat resultant_img = Frame.clone();
                if (index >= 0)
                {
                    Rect Bounding_box = boundingRect(contours[index]);
                    rectangle(resultant_img, Bounding_box, Scalar(0, 255, 0), 2);
                }

                //output - uncomment to oberve
                //imshow("Bounding Boxes", resultant_img);

                // Moments of the largest contour
                Moments moments = cv::moments(contours[index]);

                //Finding center
                Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);

                //Centeral moment computation
                double mu11 = moments.mu11 / moments.m00;
                double mu20 = moments.mu20 / moments.m00;
                double mu02 = moments.mu02 / moments.m00;

                //Angle oriantation 
                double t = 0.5 * atan2(2 * mu11, mu20 - mu02);

                //Axis-parallel Line Computation Method
                double cos_t = cos(t);
                double sin_t = sin(t);
                double x1 = center.x + cos_t * mu20 + sin_t * mu11;
                double y1 = center.y + cos_t * mu11 + sin_t * mu02;
                double x2 = center.x - cos_t * mu20 - sin_t * mu11;
                double y2 = center.y - cos_t * mu11 - sin_t * mu02;

                // ALCM over captured frame 
                line(resultant_img, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

                // Output
                imshow("ALCM", resultant_img);

                // Hu Moments
                cv::Moments mu = cv::moments(contours[index]);
                cv::HuMoments(mu, Hu_Moments);

                // Output
                std::cout << "\nHu Moments: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << Hu_Moments[i] << " ";
                }

                double mu20_normal = moments.m20 / pow(moments.m00, 2.0 / 2.0);
                double mu02_normal = moments.m02 / pow(moments.m00, 2.0 / 2.0);
                double mu11_normal = moments.m11 / pow(moments.m00, 2.0 / 2.0);

                // Normalized Moments
                std::cout << "\nmu20_norm mu02_normal mu11_normal: ";
                std::cout << mu20_normal << " " << mu02_normal << " " << mu11_normal;

                //Input request
                int keyboard = waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                {
                    for (int i = 0; i < 7; i++) {

                        Hu_Moments[i] = -1 * copysign(1.0, Hu_Moments[i]) * log10(abs(Hu_Moments[i]));
                        object_features.push_back(Hu_Moments[i]);
                    }
                    std::cout << "\n\n\nHu Moments: ";
                    for (int i = 0; i < 7; i++) {
                        std::cout << Hu_Moments[i] << " ";
                    }
                    mu20_normal = copysign(1.0, mu20_normal) * log10(abs(mu20_normal));
                    mu02_normal = copysign(1.0, mu02_normal) * log10(abs(mu02_normal));
                    mu11_normal = copysign(1.0, mu11_normal) * log10(abs(mu11_normal));
                    std::cout << "\n\nmu20_norm mu02_normal mu11_normal: ";
                    std::cout << mu20_normal << " " << mu02_normal << " " << mu11_normal;
                    object_features.push_back(mu20_normal);
                    object_features.push_back(mu02_normal);
                    object_features.push_back(mu11_normal);
                    append_image_data_csv(filename, image_filename, object_features, 0);
                    break;
                }
            }
            cout << "\n1. Training  2. Classifier  3.Exit Enter(1/2/3):";
            cin >> input;
        }
        while (input == '2')
        {
            //Classification 
            cv::destroyAllWindows();
            vector<vector<float>> feature_1;
            vector<vector<string>> label_1;
            vector<float> object1_features(0);
            vector<char*> filenames;

            // Reading data_base.csv file 
            read_image_data_csv(filename, filenames, feature_1, 0);
       
            for (int i = 0; i < filenames.size(); ++i)
            {
                cout << "\n" << filenames[i] << " ";
                for (int j = 0; j < feature_1[i].size(); ++j)
                    cout << " " << feature_1[i][j] << " ";
            }
            while (true) {
                capdev >> Frame;
                if (Frame.empty())
                    break;

                //Task 1:Threshold the input video
                
                bg_Sub->apply(Frame, fg_mask, 0);
                rectangle(Frame, cv::Point(10, 2), cv::Point(100, 20),
                    cv::Scalar(255, 255, 255), -1);
                stringstream ss;
                ss << capdev.get(CAP_PROP_POS_FRAMES);
                string frameNumberString = ss.str();
                putText(Frame, frameNumberString.c_str(), cv::Point(15, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                imshow("Frame", Frame);
                imshow("FG Mask", fg_mask);

                //get the input from the keyboard
                int keyboard = waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                    break;
            }
            while (true)
            {
                Rect Bounding_box;
                string topLabel;
                //Task2:Clean up the binary image
                Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

                morphologyEx(fg_mask, C_img, MORPH_OPEN, kernel, Point(-1, -1), 1);
                morphologyEx(C_img, C_img, MORPH_CLOSE, kernel, Point(-1, -1), 3); 
                morphologyEx(C_img, C_img, MORPH_ERODE, kernel, Point(-1, -1), 1);
                imshow("Cleaned Up Image", C_img);

                //Task 3:Segmenting the regions using 2-pass algorithm
                Label(C_img, S_img);
                //imshow("Segmented Image", S_img);

                //Task 4: Compute features for each major region
                // Contours detection
                RNG random_no_gen(12345);
                vector<vector<Point>> contours;
                findContours(C_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
                // Largest area
                double area_max = -1;
                int index = -1;
                for (int i = 0; i < contours.size(); i++)
                {
                    double area = contourArea(contours[i]);
                    if (area > area_max)
                    {
                        area_max = area;
                        index = i;
                    }
                }
                // Bounding box
                Mat resultant_img = Frame.clone();
                if (index >= 0)
                {
                    Bounding_box = boundingRect(contours[index]);
                    rectangle(resultant_img, Bounding_box, Scalar(0, 255, 0), 2);
                }

                // Output - uncomment to observe
                //imshow("Bounding Boxes", resultant_img);

                // Moments of the largest contour
                Moments moments = cv::moments(contours[index]);

                // Finding center
                Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);

                // Centeral moment computation
                double mu11 = moments.mu11 / moments.m00;
                double mu20 = moments.mu20 / moments.m00;
                double mu02 = moments.mu02 / moments.m00;

                // Angle oriantation
                double t = 0.5 * atan2(2 * mu11, mu20 - mu02);

                // Axis-parallel Line Computation Method
                double cos_t = cos(t);
                double sin_t = sin(t);
                double x1 = center.x + cos_t * mu20 + sin_t * mu11;
                double y1 = center.y + cos_t * mu11 + sin_t * mu02;
                double x2 = center.x - cos_t * mu20 - sin_t * mu11;
                double y2 = center.y - cos_t * mu11 - sin_t * mu02;

                // ALCM over captured frame
                line(resultant_img, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

                // 
                imshow("ALCM", resultant_img);

                // Calculate Hu moments
                cv::Moments mu = cv::moments(contours[index]);
                cv::HuMoments(mu, Hu_Moments);

                // Hu Moments
                /*std::cout << "\nHu Moments: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << Hu_Moments[i] << " ";
                }*/

                double mu20_normal = moments.m20 / pow(moments.m00, 2.0 / 2.0);
                double mu02_normal = moments.m02 / pow(moments.m00, 2.0 / 2.0);
                double mu11_normal = moments.m11 / pow(moments.m00, 2.0 / 2.0);

                // Normalized moments
                //std::cout << "\nmu20_norm mu02_normal mu11_normal: ";
                //std::cout << mu20_normal << " " << mu02_normal << " " << mu11_normal;

                //Input requested
                int keyboard = waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                {
                    for (int i = 0; i < 7; i++) {

                        Hu_Moments[i] = -1 * copysign(1.0, Hu_Moments[i]) * log10(abs(Hu_Moments[i]));
                        object1_features.push_back(Hu_Moments[i]);
                    }
                    /*std::cout << "\n\n\nHu Moments: ";
                    for (int i = 0; i < 7; i++) {
                        std::cout << Hu_Moments[i] << " ";
                    }*/
                    mu20_normal = copysign(1.0, mu20_normal) * log10(abs(mu20_normal));
                    mu02_normal = copysign(1.0, mu02_normal) * log10(abs(mu02_normal));
                    mu11_normal = copysign(1.0, mu11_normal) * log10(abs(mu11_normal));
                    object1_features.push_back(mu20_normal);
                    object1_features.push_back(mu02_normal);
                    object1_features.push_back(mu11_normal);
                    //std::cout << "\n\nmu20_norm mu02_normal mu11_normal: ";
                    //std::cout << mu20_normal << " " << mu02_normal << " " << mu11_normal;

                    vector<pair<float, string>> distances;

                    // Standard Deviation for min distance 
                    std::vector<float> std_dev(feature_1[0].size(), 0);
                    for (int i = 0; i < feature_1[0].size(); i++) {
                        float mean = 0.0;
                        for (const auto& fv : feature_1) {
                            mean += fv[i];
                        }
                        mean /= feature_1.size();

                        float var = 0.0;
                        for (const auto& fv : feature_1) {
                            var += (fv[i] - mean) * (fv[i] - mean);
                        }
                        var /= feature_1.size();

                        std_dev[i] = std::sqrt(var);
                    }

                    for (int i = 0; i < filenames.size(); ++i)
                    {
                        float sed = scaled_edist(object1_features, feature_1[i], std_dev);
                        distances.emplace_back(sed, filenames[i]);
                    }
                    sort(distances.begin(), distances.end());
                    cout << "\nTop match:" << distances[0].second;
                    break;

                    topLabel = distances[0].second;

                    //Task 7 :KNN classfier
                    // 
                    // Query and train descriptors
                    std::vector<cv::Mat> Q_desc, T_desc;
                    for (int i = 0; i < feature_1.size(); ++i) {
                        cv::Mat featureMat = cv::Mat(feature_1[i]).reshape(1, 1);
                        T_desc.push_back(featureMat);
                    }

                    //Create a matrix object for the feature vector of the object being detected and add it to the Q_desc variable
                    cv::Mat f_matrix = cv::Mat(object1_features).reshape(1, 1);
                    Q_desc.push_back(f_matrix);

                    int k = 2;
                    std::vector<std::vector<cv::DMatch>> m_h;

                    //Create a DescriptorMatcher object using the FLANNBASED algorithm to perform the matching between the feature vectors
                    cv::Ptr<cv::DescriptorMatcher> m_r = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
                    m_r->knnMatch(Q_desc, T_desc, m_h, k);
  
                    cv::Point t_o(Bounding_box.x, Bounding_box.y - 10);
                    line(resultant_img, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

                    // Bounding box
                    cv::putText(resultant_img, topLabel, t_o, FONT_HERSHEY_COMPLEX, resultant_img.cols / 500, Scalar({ 250,200,0 }), resultant_img.cols / 30);

                    // Output
                    imshow("Top Match", resultant_img);
                }

            }

        }

    }
    if (input == '3')
    {
        // Input requested
        while (cv::waitKey(1) != 'q');
        cv::destroyAllWindows();
        return 0;
    }
}