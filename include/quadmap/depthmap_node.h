// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#pragma once
#include <dataset_reader/BenchmarkDatasetReader.h>

#include <quadmap/depthmap.h>

namespace quadmap
{
    class DepthmapNode
    {
    public:
        DepthmapNode();

        bool init();

        void readTumDataSet();
    private:
        const std::string dataset_path = "/home/robomaster/dataset/rgbd/";
        const std::string tuple_ground_file = "tuple_ground.txt";
        // 共享智能指针
        std::shared_ptr<quadmap::Depthmap> depthmap_;
        int num_msgs_;
        double curret_msg_time;

        DatasetReader* reader;
    };
}