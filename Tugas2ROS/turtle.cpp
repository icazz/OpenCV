#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <turtlesim/Pose.h>
#include <turtlesim/Spawn.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    ros::init(argc, argv, "turtle_test");
    ros::NodeHandle nh;

    ros::Publisher velocity_pub = nh.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel", 10);
    ros::Rate rate(10);

    ros::ServiceClient spawn_client = nh.serviceClient<turtlesim::Spawn>("/turtle1/spawn");


    VideoCapture capture("Video.avi");
    if (!capture.isOpened()) {
        cerr << "Tidak dapat membuka file video!" << endl;
        return -1;
    }

    Mat frame;
    capture >> frame;
    if (frame.empty()) {
        cerr << "Tidak dapat membaca frame pertama!" << endl;
        return -1;
    }

    Point2f center(frame.cols / 2.0f, frame.rows / 2.0f);
    Point2f position(39.288, 35.783); // Didapat dari frame pertama, diperoleh dari codingan terpisah

    turtlesim::Spawn spawn_msg;
    spawn_msg.x = position.x/35/10.0f;
    spawn_msg.y = position.y/35/10.0f;
    spawn_msg.name = "ball";
    if (!spawn_client.call(spawn_msg)) {
        cerr << "Failed to spawn turtle." << endl;
        return -1;
    }

    Mat hsv_frame;

    Scalar lower_orange(5, 150, 150);
    Scalar upper_orange(15, 255, 255);

    while (ros::ok()) {
        capture >> frame;
        if (frame.empty()) break;

        cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

        Mat mask;
        inRange(hsv_frame, lower_orange, upper_orange, mask);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point2f ball_center(0, 0);

        if (!contours.empty()) {
            double max_area = 0;
            for (const auto& contour : contours) {
                double area = contourArea(contour);
                if (area > max_area) {
                    max_area = area;
                    Moments mu = moments(contour);
                    ball_center = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
                }
            }
        }

        circle(frame, center, 5, Scalar(0, 0, 255), -1); // Titik tengah

        if (ball_center.x >= 0 && ball_center.y >= 0) {
            circle(frame, ball_center, 10, Scalar(0, 0, 255), -1); // Gambar bola

            // Hitung koordinat relatif terhadap titik tengah
            float dx = position.x - ((ball_center.x - center.x) / 10.0f);
            float dy = position.y - ((center.y - ball_center.y) / 10.0f); // Y dibalik

            // turtle baru sebagai bola
            geometry_msgs::Twist velocity_msg;
            velocity_msg.linear.x = dx/35/10.0f;
            velocity_msg.linear.y = 10 - (dy/35/10.0f);
            velocity_pub.publish(velocity_msg);

            string positionText = "Posisi Bola: (" + to_string(dx) + " cm, " + to_string(dy) + " cm)";
            putText(frame, positionText, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        } else {
            putText(frame, "Bola tidak ditemukan!", Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        Size size(680, 400);
        Mat resized;
        resize(frame, resized, size);
        imshow("tracking", resized);

        if (waitKey(30) >= 0) break;

        ros::spinOnce();
        rate.sleep();
    }

    capture.release();
    destroyAllWindows();
    return 0;
}
