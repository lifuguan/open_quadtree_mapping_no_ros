// 选择读取模式和过滤模式
const sampler_t my_sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void quadtree_image_kernel(__read_only image2d_t input_image,
                                    __write_only image2d_t output_image) {
  // 获取该工作项的全局ID
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int width = get_global_size(0);  // 752
  const int height = get_global_size(1); // 480

  // 该工作组中的第一个像素的位置
  const int local_x = x - get_local_id(0);
  const int local_y = y - get_local_id(1);

  // 在该block内与threads共享显存
  // 该block中的16*16个像素线程的共同
  __local float pyramid_intensity[16][16];
  __local int pyramid_num[16][16];
  __local bool approve[16][16];

  if (x >= width || y >= height) {
    return;
  }

  // 似乎GPU会读取 (x - 0.5f, y - 0.5f)的插值点？？
  int2 global_coord = (int2)(x, y);

  // 提取该像素的像素值
  const uint4 my_intensity = read_imageui(input_image, my_sampler, global_coord);

  int pyramid_level = 0;
  float average_color = my_intensity.x;

  // 测试用的, 如果图片正确载入GPU, 应该读取出正常数值
  printf("my_intensity(%d, %d) = %f\n", x, y, average_color);

  pyramid_intensity[local_x][local_y] = my_intensity.x;
  pyramid_num[local_x][local_y] = 1;
}

/**
 * test kernel
 */
__kernel void vadd(__read_only image2d_t input_image,
                   __write_only image2d_t output_image) {
  printf("wtf_test\n");
  int i = get_global_id(0);
}