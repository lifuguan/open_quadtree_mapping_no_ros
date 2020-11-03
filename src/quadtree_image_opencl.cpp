//
// Created by robomaster on 2020/10/26.
//
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * 第三方扩展库
 * git clone https://github.com/HandsOnOpenCL/Exercises-Solutions.git
 */
#include <opencl_common/util.hpp>

int main(int argc, char **argv)
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

    try
    {
        std::vector<cl::Device> device;
        cl::Context context(CL_DEVICE_TYPE_DEFAULT);
        // Get the command queue
        cl::CommandQueue queue(context);

        cl::Image2D image_input = cl::Image2D(context, CL_MEM_READ_ONLY,
                                              cl::ImageFormat(CL_R, CL_FLOAT), src.rows, src.cols, 0, src.data);
        cl::Image2D image_output = cl::Image2D(context, CL_MEM_WRITE_ONLY,
                                               cl::ImageFormat(CL_R, CL_FLOAT), dst.rows, dst.cols, 0, dst.data);



        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("../src/opencl_kernel.cl"), true);

        cl::make_kernel<cl::Image2D, cl::Image2D> opencl_kernel(program, "opencl_kernel");

        const cl::NDRange global(47, 16);  // equal to blocks
        const cl::NDRange local(30, 30);  // equal to threads

        opencl_kernel(cl::EnqueueArgs(queue, global, local), image_input, image_output);

        queue.finish();
        // queue.enqueueReadImage()
        // cl::copy(queue, image_output, dst.begin<uchar>(), dst.end<uchar>());
    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << std::endl;
    }
    return 0;
}
