//
// Created by robomaster on 2020/10/11.
//

#include <iostream>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    cv::Mat src = cv::imread("/home/robomaster/dataset/quadtree.png");
    cv::resize(src, src, cv::Size(752, 480));
    cv::Mat dst = cv::Mat::zeros(cv::Size(752, 480), CV_8U);

    if (src.size().height != 480 || src.size().width != 752)
    {
        std::cout << "The size of the image does not meet the requirement." << std::endl;
        return 0;
    }
    int grid_x = 30;
    int grid_y = 47;
    int block_x = 16;
    int block_y = 16;



    for (int step_x = 0; step_x < 480 / grid_x; step_x++)
    {
        for (int step_y = 0; step_y < 752 / grid_y; step_y++)
        {
            float pyramid_intensity[16][16];
            int pyramid_num[16][16];
            bool approve[16][16];

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    // int a = dst.at<uchar>(x, y);
                    // std::cout << a << std::endl;
                    int x = step_x * grid_x + i;
                    int y = step_y * grid_y + j;
                    int local_x = i;
                    int local_y = j;


                    float my_intensity = src.at<uchar>(x, y);
                    int pyramid_level = 0;
                    float average_color = my_intensity;

                    pyramid_intensity[local_x][local_y] = my_intensity;
                    pyramid_num[local_x][local_y] = 1;

                    int level_x;
                    int level_y;
                    int num_pixels;
                    for (int i = 1; i <= 4; i++)
                    {
                        // << 左移一位（×2）
                        level_x = local_x - local_x % (1 << i);
                        level_y = local_y - local_y % (1 << i);
                        num_pixels = (1 << i) * (1 << i);
                        bool I_AM_LAST_NODE = (local_x % (1 << (i - 1)) == 0) && (local_y % (1 << (i - 1)) == 0);

                        if (I_AM_LAST_NODE && (level_x != local_x || level_y != local_y))
                        {
                            // 原子相加
                            pyramid_intensity[level_x][level_y] += pyramid_intensity[local_x][local_y];
                            pyramid_num[level_x][level_y] += pyramid_num[local_x][local_y];
                        }
                        approve[level_x][level_y] = true;
                    }

                    for (int i = 1; i <= 4; i++)
                    {
                        if (pyramid_num[level_x][level_y] != num_pixels)
                            break;

                        average_color = pyramid_intensity[level_x][level_y] / float(num_pixels);
                        if (fabs(my_intensity - average_color) > 10.0)
                        {
                            approve[level_x][level_y] = false;
                        }
                    }

                    for (int i = 1; i <= 4; i++)
                    {
                        if (approve[level_x][level_y])
                            pyramid_level = i;
                        else
                        {
                            pyramid_num[level_x][level_y] = 0;
                            break;
                        }
                    }
                    pyramid_level = pyramid_level < 2 ? 2 : pyramid_level;
                    dst.at<uchar>(x,y) = pyramid_level;
                     int a = dst.at<uchar>(x, y);
                     std::cout << a << std::endl;
                }
            }
        }
    }

    cv::imshow("result", dst);
    cv::waitKey(10000);

    return 0;
}
