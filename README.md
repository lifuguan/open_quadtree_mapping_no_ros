# 将QuadtreeMapping改成无ROS的版本，并利用TUM数据集进行测试
## A Real-time Monocular Dense Mapping System

This is a monocular dense mapping system following the IROS 2018  **Quadtree-accelerated Real-time Monocular Dense Mapping**, Kaixuan Wang, Wenchao Ding, Shaojie Shen.

## Requirement

- **CUDA** : version >8.0
- **OpenCV** : version > 3.0
- **Eigen**
- **Boost**

## Dataset 

We use **TUM rgbd-slam dataset** as data input. Link: http://vision.in.tum.de/data/datasets/rgbd-dataset/download

## Tools 

### associate.py
1. It reads the time stamps from the **rgb.txt** file and the **depth.txt** file.
2. Joins them by finding the best matches.
3. Write them into the **tuple.txt**.
4. Also compare the time stamps from the **rgb.txt** file and the **groundtruth.txt** file then write them into **tuple_ground.txt**.

```shell script
 python associate.py /home/robomaster/dataset/rgbd/rgb.txt /home/robomaster/dataset/rgbd/depth.txt 
 python associate.py /home/robomaster/dataset/rgbd/rgb.txt /home/robomaster/dataset/rgbd/groundtruth.txt
```


## How to install 

1. Install Eigen & OpenCV (if you don't have it):
```shell script
sudo apt-get install libeigen3-dev libopencv-dev
```
2. install ziplib:
```shell script
sudo apt-get install zlib1g-dev
cd thirdparty
tar -zxvf libzip-1.1.1.tar.gz
cd libzip-1.1.1/
./configure
make
sudo make install
sudo cp lib/zipconf.h /usr/local/include/zipconf.h
```

## tupele_ground.txt file format
The format of each line is `timestamp image_path timestamp tx ty tz qx qy qz qw`

## Configure CMakeLists.txt

```shell script
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/ -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D OPENCV_EXTRA_MODULES_PATH=/home/robomaster/opencv_contrib/modules/ /home/robomaster/opencv-3.4.0/

cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/ -D OPENCV_EXTRA_MODULES_PATH=/home/robomaster/opencv_contrib/modules/ /home/robomaster/opencv-3.4.0/
```

## Using OpenCL to build *quadtree_image_kernel*

The project refers to the following tutorials as well as its code.
1. [HandsOnOpenCL slides](https://github.com/HandsOnOpenCL/Lecture-Slides)
2. [HandsOnOpenCL 中文翻译](https://github.com/Kivy-CN/HandsOnOpenCL_CN)
3. [HandsOnOpenCL 练习代码](https://github.com/HandsOnOpenCL/Exercises-Solutions.git)

### Requirements
Export **cpp_common** files to the include path, especially header file **util.hpp**.

## kernel function help list

| Function                              | Property returned                                                                                                                         |
|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| uint get_work_dim()                   | The number of dimensions 维度统计                                                                                                         |
| size_t get_global_id(uint dimidx)     | The ID of the current work-item [0,WI) in dimension dimidx 读取该工作项在全局工作项范围内的id（输入参数为指定维度）                       |
| size_t get_global_size(uint dimidx)   | The total number of work-items (WI) in dimension dimidx 读取该维度的工作项大小                                                            |
| size_t get_global_offset(uint dimidx) | The offset as specified in the enqueueNDRangeKernel API in dimension dimidx                                                               |
| size_t get_group_id(uint dimidx)      | The ID of the current work-group [0, WG) in dimension dimidx 读取该工作项在该工作组范围内的id（输入参数为指定维度）？？？                 |
| size_t get_local_id(uint dimidx)      | The ID of the work-item within the work-group [0, WI/WG) in dimension dimidx 读取该工作项在该工作组范围内的id（输入参数为指定维度）？？？ |
| size_t get_local_size (uint dimidx)   | The number of work-items per work-group = WI/WG in dimension dimidx                                                                       |
| size_t get_num_groups(uint dimidx)    | The total number of work-groups (WG) in dimension dimidx                                                                                  |

## How CUDA kernel function `generate_gradient()` work in code

### Call hierarchy
```c++
DepthmapNode::readTumDataSet()
    ->Depthmap::add_frames()
        ->SeedMatrix::input_raw()
            ->SeedMatrix::add_frames()
                ->SeedMatrix::add_income_image()
                    ->generate_gradient()
                        ->__kernel__ generate_gradient()
```
### What we need to do 