//
// Created by lifuguan on 2020/10/11.
//

#include <iostream>

#include <opencv2/opencv.hpp>


int main(int argc, char** argv)
{
    cv::Mat src = cv::imread("/home/robomaster/dataset/quadtree.png");
    cv::resize(src, src, cv::Size(752, 480));
    cv::cvtColor(src, src, CV_BGR2GRAY);
    cv::Mat dst = cv::Mat::zeros(cv::Size(752, 480), CV_8U);

    if (src.size().height != 480 || src.size().width != 752)
    {
        std::cout << "The size of the image does not meet the requirement." << std::endl;
        return 0;
    }
    int grid_x = 16;
    int grid_y = 16;

    for (int step_x = 0; step_x < 480 / grid_x; step_x++)
    {
        for (int step_y = 0; step_y < 752 / grid_y; step_y++)
        {
            float pyramid_intensity[16][16]={};
            int pyramid_num[16][16]={};
            bool approve[16][16]={};

            int pyramid_level_ok[16][16]={};
            int pyramid_level_mark[16][16] ={};

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    int x = step_x * grid_x + i;
                    int y = step_y * grid_y + j;
                    int local_x = i;
                    int local_y = j;

                    float my_intensity = src.at<uchar>(x, y);
                    pyramid_intensity[local_x][local_y] = my_intensity;
                    pyramid_num[local_x][local_y] = 1;
                    approve[local_x][local_y] = false;
                }
            }
            for (int k = 1; k <= 4; k++)
            {

                // TODO 第一次同步 计算平均像素梯度
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        int local_x = i;
                        int local_y = j;

                        int level_x = local_x - local_x % (1 << k);
                        int level_y = local_y - local_y % (1 << k);
                        bool I_AM_LAST_NODE = (local_x % (1 << (k - 1)) == 0) && (local_y % (1 << (k - 1)) == 0);

                        if (I_AM_LAST_NODE && (level_x != local_x || level_y != local_y))
                        {
                            // 原子相加
                            pyramid_intensity[level_x][level_y] += pyramid_intensity[local_x][local_y];
                            pyramid_num[level_x][level_y] += pyramid_num[local_x][local_y];
                        }
                        approve[level_x][level_y] = true;
                    }
                }

                // TODO 第二次同步 计算当前层上的该像素与平均像素梯度的差
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        int x = step_x * grid_x + i;
                        int y = step_y * grid_y + j;
                        int local_x = i;
                        int local_y = j;

                        int level_x = local_x - local_x % (1 << k);
                        int level_y = local_y - local_y % (1 << k);
                        int num_pixels = (1 << k) * (1 << k);

                        if (pyramid_num[level_x][level_y] != num_pixels)
                        {
                            pyramid_level_ok[local_x][local_y ] = true;
                            continue;
                        }

                        float average_color = pyramid_intensity[level_x][level_y] / float(num_pixels);

                        float my_intensity = (float)src.at<uchar>(x, y);

                        if (fabs(my_intensity - average_color) > 10.0)
                        {
                            approve[level_x][level_y] = false;
                        }
                    }
                }

                // TODO 第三次同步
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        int local_x = i;
                        int local_y = j;

                        int level_x = local_x - local_x % (1 << k);
                        int level_y = local_y - local_y % (1 << k);

                        if (approve[level_x][level_y])
                            if (!pyramid_level_ok[local_x][local_y])
                            {
                                pyramid_level_mark[local_x][local_y] = k;
                            }
                        else
                        {
                            pyramid_num[level_x][level_y] = 0;
                            pyramid_level_ok[local_x][local_y] = true;
                            continue;
                        }
                    }
                }
            }
            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    int x = step_x * grid_x + i;
                    int y = step_y * grid_y + j;
                    int pyramid_level = pyramid_level_mark[i][j];
                    pyramid_level = pyramid_level < 2 ? 2 : pyramid_level;
                    dst.at<uchar>(x,y) = (uchar)(pyramid_level * 30);
                }
            }
        }
    }

    cv::imshow("source", src);
    cv::imshow("result", dst);

    cv::waitKey();

    return 0;
}
