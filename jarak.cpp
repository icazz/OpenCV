#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;
int main(){

    namedWindow("Trackbars", WINDOW_AUTOSIZE);

    //batas bawah dan atas warna (warna orange)
    int h_min = 0, s_min = 0, v_min = 0;
    int h_max = 179, s_max = 255, v_max = 255;

    //membuat trackbar
    createTrackbar("Hue Min", "Trackbars", &h_min, 179);
    createTrackbar("Hue Max", "Trackbars", &h_max, 179);
    createTrackbar("Sat Min", "Trackbars", &s_min, 255);
    createTrackbar("Sat Max", "Trackbars", &s_max, 255);
    createTrackbar("Val Min", "Trackbars", &v_min, 255);
    createTrackbar("Val Max", "Trackbars", &v_max, 255);

    setTrackbarPos("Hue Min", "Trackbars", 179);
    setTrackbarPos("Hue Max", "Trackbars", 179);
    setTrackbarPos("Sat Min", "Trackbars", 255);
    setTrackbarPos("Sat Max", "Trackbars", 255);
    setTrackbarPos("Val Min", "Trackbars", 255);
    setTrackbarPos("Val Max", "Trackbars", 255);

    vector<double> px = {406, 300, 220, 210, 200, 190, 180, 170, 160, 150, 140, 120, 110, 100, 90, 80, 70, 60, 50};
    vector<double> cm = {5, 13, 18, 19, 20, 21, 22, 24, 25, 27, 29, 32, 34, 36, 39, 46, 52, 61, 80};

    // vector<double> px = {350, 177, 120, 89, 72, 59, 50, 44, 38, 36};
    // vector<double> cm = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    Mat frame;
    Mat frame_hsv;
    Mat thresholded;
    while (true) {
        cap >> frame;

        cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
        inRange(frame_hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), thresholded);

        erode(thresholded, thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(thresholded, thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		dilate(thresholded, thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresholded, thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        vector<vector<Point>> contours;
        findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        //menggambar
        drawContours(frame, vector<vector<Point>>{contours}, -1, Scalar(0, 255, 0), 2);
        

        if (!contours.empty()){
            double max_area = 0;
            vector<Point> largest;

            for (const auto& contour : contours){
                double area = contourArea(contour);
                if (area > max_area){
                    max_area = area;
                    largest = contour;
                }
            }

            Rect boundingBox = boundingRect(largest);
            int size = boundingBox.width;
            //rectangle(frame, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 0), 2);

            if (px.size() > 1 && cm.size() > 1){
                //membangun matriks untuk regresi
                int n = px.size();
                MatrixXd X(n, 2);
                VectorXd Y(n);

                for (int i = 0; i < n; ++i){
                    X(i, 0) = px[i];
                    X(i, 1) = 1;
                    Y(i) = cm[i];
                }
                
                VectorXd coeffs = (X.transpose() * X).inverse() * X.transpose() * Y;
                double distance = coeffs(0) * size + coeffs(1); // y = mx + b

                // Tampilkan jarak pada frame
                string distance_text = "Distance: " + to_string(distance) + " cm";
                putText(frame, distance_text, Point(boundingBox.x, boundingBox.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
            }
        }
        
        imshow("Camera", frame);
        imshow("Thresholded", thresholded);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}

