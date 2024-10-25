#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
using namespace std;
using namespace cv;

int main() {
    
    VideoCapture cap("/home/zika/magang-video/Video asli.avi");
    if (!cap.isOpened()) {
        return -1;
    }
    Mat hsv;
    Mat frame1, frame2, gray1, gray2;
    vector<Point2f> pointsBefore, pointsAfter;
    vector<uchar> status;
    vector<float> err;


    cap >> frame1;
    cvtColor(frame1, gray1, COLOR_BGR2GRAY);

    goodFeaturesToTrack(gray1, pointsBefore, 100, 0.3, 7);

    static Point2f position(0, 0);

    //resive video
    Size size(680, 400);

    while (true) {
        cap >> frame2;
        if (frame2.empty()) break;
        cvtColor(frame2, gray2, COLOR_BGR2GRAY);

        calcOpticalFlowPyrLK(gray1, gray2, pointsBefore, pointsAfter, status, err);

        Point2f titik(0, 0);
        int realPoints = 0;
        for (size_t i = 0; i < pointsAfter.size(); i++) {
            if (status[i]) {
                titik += (pointsBefore[i] - pointsAfter[i]);
                realPoints++;
                //cout << realPoints << endl;
            }
        }

        if (realPoints > 0) {
            titik.x /= realPoints;
            titik.y /= realPoints;

            //10 px = 1 cm
            titik.x /= 10.0;
            titik.y /= 10.0;

            //Sumbu y
            titik.y *= -1;

            //koordinat robot
            position += titik;
            // position.x += titik.x;
            // position.y += titik.y;
        }

        //output
        string text = "Posisi: (" + to_string(position.x) + " cm, " + to_string(position.y) + " cm)";
        putText(frame2, text, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

        // Mengecilkan ukuran frame
        Mat resized;
        resize(frame2, resized, size);
        
        //deteksi bola
        cvtColor(resized, hsv, COLOR_BGR2HSV);
        
        Scalar min(7, 150, 150);
        Scalar max(10, 255, 255);

        Mat mask;

        inRange(hsv, min, max, mask);
        vector <vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        //menghilangkan noise
        erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(2.5, 2.5)));
		dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//memperhalus
		dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        //gambar kontur
        for (const auto& contour : contours) {
            if (contourArea(contour) > 10) {
                Rect boundingBox = boundingRect(contour);
                rectangle(resized, boundingBox, Scalar(0, 255, 0), 1);
            }
        }

        imshow("Posisi Robot dan Deteksi Bola", resized);
        imshow("cek", mask);


        gray1 = gray2.clone();
        pointsBefore = pointsAfter;

        if (waitKey(30) >= 0) break;
    }

    return 0;
}
