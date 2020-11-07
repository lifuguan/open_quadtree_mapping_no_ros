//
// Created by robomaster on 2020/10/26.
//
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <opencl_common/err_code.h>
/**
 * 第三方扩展库
 * git clone https://github.com/HandsOnOpenCL/Exercises-Solutions.git
 */
#include <opencl_common/util.hpp>

int main(int argc, char **argv)
{
    std::cout << CL_DEVICE_MAX_WORK_GROUP_SIZE << " " << CL_KERNEL_WORK_GROUP_SIZE << std::endl;
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
        std::vector<cl::Platform> platfromList;

        cl::Platform::get(&platfromList);

        cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) (platfromList[0])(), 0};

        cl::Context context(CL_DEVICE_TYPE_GPU, cprops);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl::CommandQueue queue(context, devices[0], 0);

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("../src/opencl_kernel.cl"));

        program.build(devices);


        cl::Image2D image_input = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), src.rows,
                                              src.cols, 0, src.data);
        cl::Image2D image_output = cl::Image2D(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), dst.rows,
                                               dst.cols, 0, dst.data);


        cl::make_kernel<cl::Image2D, cl::Image2D> image_kernel(program, "quadtree_image_kernel");

        /**
         * 此处表示总共有 752*480 的工作项 （total number of work-items）
         * 同时规定了每个工作组（ per work-group）里有 16*16 的工作项（work-items）
         * 和CUDA的规定有差别
         */
        const cl::NDRange global(752, 480);
        const cl::NDRange local(16, 16);

        image_kernel(cl::EnqueueArgs(global, local), image_input, image_output);

        queue.finish();
        // queue.enqueueReadImage()
        // cl::copy(queue, image_output, dst.begin<uchar>(), dst.end<uchar>());
    } catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" <<  std::endl;
    }
    return 0;
}
