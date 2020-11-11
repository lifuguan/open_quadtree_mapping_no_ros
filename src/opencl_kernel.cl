void atomic_add_float(volatile local float *source, const float operand) {
  union {
    unsigned int intVal;
    float floatVal;
  } newVal;

  union {
    unsigned int intVal;
    float floatVal;
  } prevVal;

  do {
    prevVal.floatVal = *source;
    newVal.floatVal = prevVal.floatVal + operand;
  } while (atomic_cmpxchg((volatile local unsigned int *)source, prevVal.intVal,
                          newVal.intVal) != prevVal.intVal);
}

// 选择读取模式和过滤模式
const sampler_t my_sampler = CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void quadtree_image_kernel(__read_only image2d_t input_image,
                                    __write_only image2d_t output_image) {
  // 获取该工作项的全局ID
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int width = get_image_width(input_image);   // 752
  const int height = get_image_height(input_image); // 480
  // printf("width : %f, height: %f\n", width, height);
  // 该工作组中的第一个像素的local位置
  const int local_x = get_local_id(0);
  const int local_y = get_local_id(1);

  // 在该block内与threads共享显存
  // 该block中的16*16个像素线程的共同
  __local float pyramid_intensity[16][16];
  __local int pyramid_num[16][16];
  __local bool approve[16][16];
  __local float test[16];
  if (x >= width || y >= height) {
    return;
  }

  // 似乎GPU会读取 (x - 0.5f, y - 0.5f)的插值点？？
  int2 global_coord = (int2)(x, y);

  // 提取该像素的像素值
  const uint4 my_intensity =
      read_imageui(input_image, my_sampler, global_coord);

  int pyramid_level = 0;
  float average_color = my_intensity.x;

  pyramid_intensity[local_x][local_y] = my_intensity.x;
  pyramid_num[local_x][local_y] = 1;

  // 测试用的, 如果图片正确载入GPU, 应该读取出正常数值
  // printf("pyramid_intensity(%d, %d) = %f\n", x, y,
  //        pyramid_intensity[local_x][local_y]);

  // 从16×16的最粗糙区块检测是否可以再分
  for (int i = 1; i <= 4; i++) {
    // << 左移一位（×2）
    int level_x = local_x - local_x % (1 << i);
    int level_y = local_y - local_y % (1 << i);
    int num_pixels = (1 << i) * (1 << i);

    bool I_AM_LAST_NODE =
        (local_x % (1 << (i - 1)) == 0) && (local_y % (1 << (i - 1)) == 0);

    if (I_AM_LAST_NODE && (level_x != local_x || level_y != local_y)) {
      // 原子相加
      atomic_add_float(&pyramid_intensity[level_x][level_y],
                       pyramid_intensity[local_x][local_y]);
      atomic_add(&(pyramid_num[level_x][level_y]),
                 pyramid_num[local_x][local_y]);
    }
    approve[level_x][level_y] = true;

    // 屏障同步函数, 完全等于CUDA __syncthreads()
    barrier(CLK_LOCAL_MEM_FENCE);

    if (pyramid_num[level_x][level_y] != num_pixels)
      break;

    average_color = pyramid_intensity[level_x][level_y] / (float)(num_pixels);
    if (fabs(my_intensity.x - average_color) > 10.0) {
      approve[level_x][level_y] = false;
    }
    // 线程同步
    barrier(CLK_LOCAL_MEM_FENCE);

    if (approve[level_x][level_y])
      pyramid_level = i;
    else {
      pyramid_num[level_x][level_y] = 0;
      break;
    }
    // 线程同步
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  pyramid_level = pyramid_level < 2 ? 2 : pyramid_level;

  // *10 强化显示
  uint4 pixel_val = (uint4)(pyramid_level * 10, 0.0f, 0.0f, 0.0f); 

  // printf("(%d, %d) = %f = %f\n", x, y, my_intensity.x, pixel_val.x);

  write_imageui(output_image, (int2)(x, y), pixel_val);
}

/**
 * test kernel
 */
__kernel void vadd(__read_only image2d_t input_image,
                   __write_only image2d_t output_image) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int width = get_image_width(input_image);   // 752
  const int height = get_image_height(input_image); // 480
  // printf("width : %f, height: %f\n", width, height);

  int2 global_coord = (int2)(x, y);
  uint4 my_intensity =
      read_imageui(input_image, my_sampler, global_coord);

  float average_color = my_intensity.x;

  // printf("pixel_val = %f\n", average_color);

  write_imageui(output_image, (int2)(x, y), (uint4)(average_color, 0.0f, 0.0f, 0.0f));
}