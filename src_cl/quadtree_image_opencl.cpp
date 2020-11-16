//
// Created by robomaster on 2020/10/26.
//
#define __CL_ENABLE_EXCEPTIONS

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

    if (src.size().height != 480 || src.size().width != 752 || src.channels() != 1)
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

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("../src/opencl_kernel.cl"), true);

        cl::CommandQueue queue(context);

        /**
         *  CL_MEM_READ_ONLY : 规定只读
         *  CL_MEM_COPY_HOST_PTR : 必填 我也不知道是啥
         *  CL_R : (R,G,B,A)中只读取R通道(单通道)
         *  CL_UNSIGNED_INT8 : 无符号8位整型
         *
         *  NOTE: 最好使用uint 8数据类型, 因为和uchar匹配
         */
        cl::Image2D image_input = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), src.size().width,
                                              src.size().height, 0, src.data);
        cl::Image2D image_output = cl::Image2D(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                                               src.size().width, src.size().height, 0, dst.data);

        std::vector<std::string> opencl_kernel_name = {"vadd", "quadtree_image_kernel"};
        // 调用quadtree_image_kernel核函数
        cl::make_kernel<cl::Image2D, cl::Image2D> image_kernel(program, opencl_kernel_name[1]);

        /**
         * 此处表示总共有 752*480 的工作项 （total number of work-items）
         * 同时规定了每个工作组（ per work-group）里有 16*16 的工作项（work-items）
         * 和CUDA的规定有差别
         */
        const cl::NDRange global(752, 480);
        const cl::NDRange local(16, 16);

        // 执行quadtree_image_kernel核函数
        image_kernel(cl::EnqueueArgs(queue, global, local), image_input, image_output);

        // 从device中读取图像到host
        cl::size_t<3> origin;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        cl::size_t<3> region;
        region[0] = dst.size().width; // arg1 : 图像的宽
        region[1] = dst.size().height;// arg2 : 图像的高
        region[2] = 1;                // arg3 : 默认0
        queue.enqueueReadImage(image_output, CL_TRUE, origin, region, 0, 0, dst.data);

        queue.finish();

    } catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }
    cv::imshow("test", dst);
    cv::waitKey(10000);
    return 0;
}
