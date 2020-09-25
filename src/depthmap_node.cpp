// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <quadmap/depthmap_node.h>

#include <quadmap/se3.cuh>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <string>
#include <future>

using namespace std;

quadmap::DepthmapNode::DepthmapNode(ros::NodeHandle &nh) : nh_(nh), num_msgs_(0)
{
}

bool quadmap::DepthmapNode::init()
{
    int cam_width = 752;
    int cam_height = 480;
    float cam_fx = 3.6537093297021136e+02;
    float cam_fy = 3.6507591213493185e+02;
    float cam_cx = 3.6964744957606678e+02;
    float cam_cy = 2.4095181606217392e+02;
    double downsample_factor = 1;
    int semi2dense_ratio = 1;


    printf("read : width %d height %d\n", cam_width, cam_height);

    float k1, k2, r1, r2;
    k1 = k2 = r1 = r2 = 0.0;


    // initial the remap mat, it is used for undistort and also resive the image
    cv::Mat input_K = (cv::Mat_<float>(3, 3) << cam_fx , 0.0f, cam_cx, 0.0f, cam_fy, cam_cy, 0.0f, 0.0f, 1.0f);
    cv::Mat input_D = (cv::Mat_<float>(1, 4) << k1, k2, r1, r2);

    float resize_fx, resize_fy, resize_cx, resize_cy;
    resize_fx = cam_fx * downsample_factor;
    resize_fy = cam_fy * downsample_factor;
    resize_cx = cam_cx * downsample_factor;
    resize_cy = cam_cy * downsample_factor;
    cv::Mat resize_K = (cv::Mat_<float>(3, 3)
            << resize_fx, 0.0f, resize_cx, 0.0f, resize_fy, resize_cy, 0.0f, 0.0f, 1.0f);
    resize_K.at<float>(2, 2) = 1.0f;
    int resize_width = cam_width * downsample_factor;
    int resize_height = cam_height * downsample_factor;

    cv::Mat undist_map1, undist_map2;
    cv::initUndistortRectifyMap(input_K, input_D, cv::Mat_<double>::eye(3, 3), resize_K,
                                cv::Size(resize_width, resize_height), CV_32FC1, undist_map1, undist_map2);

    depthmap_ = std::make_shared<quadmap::Depthmap>(resize_width, resize_height, resize_fx, resize_cx, resize_fy,
                                                    resize_cy, undist_map1, undist_map2, semi2dense_ratio);

    bool pub_pointcloud = false;
    publisher_.reset(new quadmap::Publisher(nh_, depthmap_));

    return true;
}


void quadmap::DepthmapNode::Msg_Callback(const sensor_msgs::ImageConstPtr &image_input,
                                         const geometry_msgs::PoseStampedConstPtr &pose_input)
{
    printf("\n\n\n");
    num_msgs_ += 1;
    curret_msg_time = image_input->header.stamp;
    if (!depthmap_)
    {
        ROS_ERROR("depthmap not initialized. Call the DepthmapNode::init() method");
        return;
    }
    cv::Mat img_8uC1;
    try
    {
        cv_bridge::CvImageConstPtr cv_img_ptr = cv_bridge::toCvShare(image_input, sensor_msgs::image_encodings::MONO8);
        img_8uC1 = cv_img_ptr->image;
    } catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    quadmap::SE3<float> T_world_curr(pose_input->pose.orientation.w, pose_input->pose.orientation.x,
                                     pose_input->pose.orientation.y, pose_input->pose.orientation.z,
                                     pose_input->pose.position.x, pose_input->pose.position.y,
                                     pose_input->pose.position.z);

    bool has_result;
    has_result = depthmap_->add_frames(img_8uC1, T_world_curr.inv());
    if (has_result)
        denoiseAndPublishResults();
}


void quadmap::DepthmapNode::denoiseAndPublishResults()
{
    std::async(std::launch::async, &quadmap::Publisher::publishDepthmapAndPointCloud, *publisher_, curret_msg_time);
}