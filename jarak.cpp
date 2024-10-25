#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
using namespace std;
using namespace cv;
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

        //menghilangkan noise
        erode(thresholded, thresholded, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
		dilate(thresholded, thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//memperhalus
		dilate(thresholded, thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresholded, thresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        vector<vector<Point>> contours;
        findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        //menggambar
        drawContours(frame, vector<vector<Point>>{contours}, -1, Scalar(0, 255, 0), 2);

        if (!contours.empty()) {
            // Menghitung kontur terbesar
            size_t largestContourIndex = 0;
            double largestArea = 0;

            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > largestArea) {
                    largestArea = area;
                    largestContourIndex = i;
                }
            }
            Moments moment = moments(thresholded);
            Moments oMoments = moments(contours[largestContourIndex]);
            double dArea = moment.m00;
            //cout << dArea << endl;

            if (dArea > 100000){
                
                double real = 5.5;
                Rect boundingBox = boundingRect(contours[largestContourIndex]);
                double pixelSize = boundingBox.width;
                int fpx = 770;
                double jarak = (real*fpx) / pixelSize;
                string text = "Distance: " + to_string(jarak) + " cm";
                putText(frame, text, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                //cout << "Distance: " << jarak << " cm" << endl;
                //cout << pixelSize << endl;
                
            }
        }
        
        //menampilkan camera asli
        imshow("Camera", frame);
        //menampilkan threshold
        imshow("Thresholded", thresholded);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}

