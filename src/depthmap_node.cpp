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

#include <string>
#include <future>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


using namespace std;

quadmap::DepthmapNode::DepthmapNode()
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
    cv::Mat input_K = (cv::Mat_<float>(3, 3) << cam_fx, 0.0f, cam_cx, 0.0f, cam_fy, cam_cy, 0.0f, 0.0f, 1.0f);
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


    ///////////////////// 读取dataset : 初始化实例 //////////////////////
    reader = new DatasetReader(dataset_path, DatasetReader::RGB);
    return true;
}

void quadmap::DepthmapNode::readTumDataSet()
{
    cv::namedWindow("depth");
    cv::namedWindow("test");
    std::fstream ground_pair_file;
    // 指定文件，并设置为只读
    ground_pair_file.open(this->dataset_path + this->tuple_ground_file, std::ios::in);
    while (true)
    {
        // 按行读txt并解析成字符串
        std::vector<std::string> data_list = reader->readAndParseTxt(ground_pair_file);
        if (data_list.size() < 3)
        {
            break;
        }

        // format : timestamp image_path timestamp tx ty tz qx qy qz qw
        curret_msg_time = std::stod(data_list[0]);
        std::string img_path = data_list[1];
        cv::Mat source_img;

        try
        {
            std::cout << "\n\ntarget : " << dataset_path+img_path << std::endl;
            source_img = cv::imread(this->dataset_path+img_path, CV_8UC1);
            cv::imshow("test", source_img);
            if(cv::waitKey(5)==27)
                break;
        }
        catch (cv::Exception &exception)
        {
            std::cout << exception.what() << std::endl;
        }
        quadmap::SE3<float> T_world_curr(std::stod(data_list[9]), std::stod(data_list[6]), std::stod(data_list[7]),
                                          std::stod(data_list[8]), std::stod(data_list[3]), std::stod(data_list[4]),
                                          std::stod(data_list[5]));
        bool has_result = depthmap_->add_frames(source_img, T_world_curr.inv());
        // 有结果则返回true，下一步可以读取深度图
        if (has_result)
        {
            cv::Mat depthmap_mat = depthmap_->getDepthmap();
            cv::imshow("depth", depthmap_mat);
            if(cv::waitKey(5) == 27)
                break;
        }
    }
    delete reader;
}
