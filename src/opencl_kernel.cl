__kernel void quadtree_image_kernel( __read_only image2d_t input_image, __write_only image2d_t output_image)
{
    int local_x = get_global_id(0);
    int local_y = get_global_id(1);
    int a = local_x + local_y;
}

                      