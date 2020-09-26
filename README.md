# 将QuadtreeMapping改成无ROS的版本，并利用TUM数据集进行测试
## A Real-time Monocular Dense Mapping System

This is a monocular dense mapping system following the IROS 2018  **Quadtree-accelerated Real-time Monocular Dense Mapping**, Kaixuan Wang, Wenchao Ding, Shaojie Shen.

## Requirement

- **CUDA** : version >8.0
- **OpenCV** : version > 3.0
- **Eigen**
- **Boost**
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

## Configure CMakeLists.txt