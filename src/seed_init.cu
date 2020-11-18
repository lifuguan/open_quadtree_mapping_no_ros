#include <cuda_toolkit/helper_math.h>
#include <quadmap/device_image.cuh>
#include <quadmap/texture_memory.cuh>
#include <ctime>

namespace quadmap
{
//declear function
void generate_gradient(DeviceImage<float> &image, DeviceImage<float2> &gradient_map);
__global__ void gradient_kernel(DeviceImage<float> *image_dev_ptr, DeviceImage<float2> *gradient_dev_ptr);

//define function
void generate_gradient(DeviceImage<float> &image, DeviceImage<float2> &gradient_map)
{
	int width = gradient_map.width;      // 看来是个常数 752
	int height = gradient_map.height;    // 看来是个常数 480
    // std::cout << "frame width = " << width << ", height = " << height << std::endl;
	dim3 block;
	dim3 grid;
	block.x = 16;
	block.y = 16;
	grid.x = (width + block.x - 1) / block.x;
	grid.y = (height + block.y - 1) / block.y;
	gradient_kernel<<<grid, block>>>(image.dev_ptr, gradient_map.dev_ptr);
}
/**
 *
 * @param image_dev_ptr
 * @param gradient_dev_ptr
 * note: float2数据类型: 包含两个数据x,y
 * solution : 在opencl中利用float4格式,读取R,G两个通道的值,即CL_RG
 */
__global__ void gradient_kernel(DeviceImage<float> *image_dev_ptr, DeviceImage<float2> *gradient_dev_ptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = gradient_dev_ptr->width;
	const int height = gradient_dev_ptr->height;

	if (x >= width - 1 || y >= height - 1 || x <= 0 || y <= 0)
		return;
    // width(横)方向的梯度
	float right_color = image_dev_ptr->atXY(x+1,y);
	float left_color = image_dev_ptr->atXY(x-1,y);
	// height(纵)方向的梯度
	float down_color = image_dev_ptr->atXY(x,y+1);
	float up_color = image_dev_ptr->atXY(x,y-1);

	gradient_dev_ptr->atXY(x, y) = make_float2((right_color - left_color)/2.0, (down_color - up_color)/2.0);
}
}