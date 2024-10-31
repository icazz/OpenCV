#include <ros/ros.h>
#include <math.h>
#include <geometry_msgs/Twist.h>
#include <turtlesim/TeleportAbsolute.h>
#include <turtlesim/SetPen.h>
#include <std_srvs/Empty.h> // Hapus
#include <unistd.h>  // Untuk fungsi sleep

void move(ros::Publisher &pub, double linear_x, double angular_z, double duration) {
    geometry_msgs::Twist move_cmd;
    move_cmd.linear.x = linear_x;
    move_cmd.angular.z = angular_z;

    pub.publish(move_cmd);
    ros::Duration(duration).sleep();

    // Hentikan pergerakan
    move_cmd.linear.x = 0.0;
    move_cmd.angular.z = 0.0;
    pub.publish(move_cmd);
    ros::Duration(0.1).sleep();
}

void teleport(ros::ServiceClient &client, double x, double y, double theta) {
    turtlesim::TeleportAbsolute srv;
    srv.request.x = x;
    srv.request.y = y;
    srv.request.theta = theta;

    if (client.call(srv)) {
        ROS_INFO("Turtle teleported to (%f, %f)", x, y);
    } else {
        ROS_ERROR("Failed to call service teleport_absolute");
    }
}

void rotate(ros::Publisher &pub, double angular_z, double duration) {
    geometry_msgs::Twist rotate_cmd;
    rotate_cmd.linear.x = 0.0;
    rotate_cmd.angular.z = angular_z; // Kecepatan rotasi

    pub.publish(rotate_cmd);
    ros::Duration(duration).sleep(); 

    // Hentikan rotasi
    rotate_cmd.angular.z = 0.0;
    pub.publish(rotate_cmd);
    ros::Duration(1.0).sleep();
}

void setPen(ros::ServiceClient &client, bool off) {
    turtlesim::SetPen srv;
    srv.request.off = off; // true untuk mematikan pena, false untuk menghidupkan
    srv.request.r = 0;
    srv.request.g = 255;
    srv.request.b = 0;
    srv.request.width = 7; // Ketebalan
    if (client.call(srv)) {
        ROS_INFO("Pen %s", off ? "turned off" : "turned on");
    } else {
        ROS_ERROR("Failed to call service set_pen");
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "publisher");
    ros::NodeHandle nh;

    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel", 10);
    ros::ServiceClient client = nh.serviceClient<turtlesim::TeleportAbsolute>("/turtle1/teleport_absolute");
    ros::ServiceClient client_pen = nh.serviceClient<turtlesim::SetPen>("/turtle1/set_pen");
    ros::ServiceClient clear_client = nh.serviceClient<std_srvs::Empty>("/clear");
    ros::Rate rate(10);

    // Tunggu hingga publisher dan service client siap
    while (pub.getNumSubscribers() < 1) {
        ros::Duration(0.1).sleep(); // Tidur sejenak
    }


    while (ros::ok()){
        // Huruf I
        setPen(client_pen, true); // Mati
        teleport(client, 3.05, 9, -1.57);
        setPen(client_pen, false); //  Hidup
        move(pub, 2.05, 0.0, 1.0);
        // Huruf C
        setPen(client_pen, true); // Mati
        teleport(client, 5.55, 8.75, 3 * M_PI / 4);
        setPen(client_pen, false); // Hidup
        move(pub, 3.0, 3.0, 1.0);
        move(pub, 1.0, 1.0, 1.0);
        move(pub, 1.0, 1.0, 0.5);
        // Huruf A
        setPen(client_pen, true); // Mati
        teleport(client, 6.3, 7, M_PI / 2.7);
        setPen(client_pen, false); // Hidup
        move(pub, 2.25, 0.0, 1.0);
        rotate(pub, M_PI * 2.5, 0.5);
        move(pub, 2.25, 0.0, 1.0);
        setPen(client_pen, true); // Mati
        rotate(pub, M_PI * 2, 0.5); // Putar balik
        move(pub, 1.0, 0.0, 1.0); //  Ke tengah
        rotate(pub, M_PI / 1.4, 0.5);
        move(pub, 0.05, 0.0, 0.5);//
        setPen(client_pen, false); // Hidup
        move(pub, 1.05, 0.0, 1.0); // Garis tengah A

        // Huruf Z
        setPen(client_pen, true); // Mati
        teleport(client, 2.05, 5.0, 0.0);
        setPen(client_pen, false); // Hidup
        move(pub, 1.5, 0.0, 1.0);
        rotate(pub, -M_PI * 1.35, 0.5);
        move(pub, 2.45, 0.0, 1.0);
        rotate(pub, M_PI * 1.4, 0.5);
        move(pub, 1.5, 0.0, 1.0);
        // Huruf I
        setPen(client_pen, true); // Mati
        teleport(client, 4.59, 5.0, -1.57);
        setPen(client_pen, false); // Hidup
        move(pub, 2.05, 0.0, 1.0);
        // Huruf K
        setPen(client_pen, true); // Mati
        teleport(client, 5.59, 5.0, -1.57);
        setPen(client_pen, false); // Hidup
        move(pub, 2.05, 0.0, 1.0);
        setPen(client_pen, true); // Mati
        rotate(pub, M_PI * 2, 0.5);
        move(pub, 1.025, 0.0, 1.0);
        rotate(pub, -M_PI / 2, 0.5);
        setPen(client_pen, false); // Hidup
        move(pub, 1.38, 0.0, 1.0); // Garis k
        setPen(client_pen, true); // Mati
        rotate(pub, M_PI * 2, 0.5);
        move(pub, 1.38, 0.0, 1.0);
        rotate(pub, 2 * M_PI / 1.95, 0.5);
        setPen(client_pen, false); // Hidup
        move(pub, 1.45, 0.0, 1.0); // Garis k
        // Huruf A
        setPen(client_pen, true); // Mati
        teleport(client, 7.351, 3, M_PI / 2.7);
        setPen(client_pen, false); // Hidup
        move(pub, 2.25, 0.0, 1.0);
        rotate(pub, M_PI * 2.5, 0.5);
        move(pub, 2.25, 0.0, 1.0);
        setPen(client_pen, true); // Mati
        rotate(pub, M_PI * 2, 0.5); // Putar balik
        move(pub, 1.0, 0.0, 1.0); // Ke tengah
        rotate(pub, M_PI / 1.4, 0.5);
        move(pub, 0.05, 0.0, 1.0);
        setPen(client_pen, false); // Hidup
        move(pub, 1.05, 0.0, 1.0); // Garis tengah A
        setPen(client_pen, true); // Mati
        teleport(client, 0.0, 0.0, 0.0);

        ros::Duration(1.0).sleep();
        std_srvs::Empty srv;
        if (clear_client.call(srv)) {
            ROS_INFO("Canvas cleared");
        } else {
            ROS_ERROR("Failed to call service /clear");
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}

