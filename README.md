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