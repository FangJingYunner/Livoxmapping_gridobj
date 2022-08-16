//
// Created by fjy on 2022/8/12.
//

#include <ros/ros.h>
#include "objseg_grid_obj.h"
using namespace std;

int main(int argc, char **argv) {
    ros::init(argc, argv, "objseg_gridclass");

    ros::NodeHandle nh("~");

    ObjSegGrid_obj SegMethod(nh);

    ros::Rate rate(10);
    bool status = ros::ok();
    while (status) {

        ros::spinOnce();
        SegMethod.ObjSegGridRun();
        rate.sleep();
    }
}