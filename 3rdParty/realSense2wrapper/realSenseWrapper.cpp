#include "realSenseWrapper.h"
#include <iostream>


RealSenseWrapper::RealSenseWrapper() : align_to(RS2_STREAM_DEPTH){
    cfg.disable_all_streams();
    cfg.enable_stream(rs2_stream::RS2_STREAM_DEPTH, 640, 480, rs2_format::RS2_FORMAT_Z16, 30);
    cfg.enable_stream(rs2_stream::RS2_STREAM_COLOR, 640, 480, rs2_format::RS2_FORMAT_BGR8, 30);

    pipe.start(cfg);
}
RealSenseWrapper::~RealSenseWrapper(){
  
}

int RealSenseWrapper::acquireFramesDepthColorized(cv::Mat& _color, cv::Mat& _depth){
    rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
    rs2::frameset aligned_set = align_to.process(data);
    //color frame
    auto color_mat = frame_to_mat(aligned_set.get_color_frame());
    //depth frame
    rs2::frame depth = color_map(aligned_set.get_depth_frame());
    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();
    cv::Mat depth_mat(cv::Size(w, h), CV_8UC3, (void *) depth.get_data(), cv::Mat::AUTO_STEP);
    _depth = depth_mat.clone();
    _color = color_mat.clone();
return 0;
}

int RealSenseWrapper::acquireFrames(cv::Mat& _color,cv::Mat& _depth){
    rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
//    rs2::frameset aligned_set = align_to.process(data);
    //color frame
    auto color_mat = frame_to_mat(data.get_color_frame());
    //depth frame
    rs2::frame depth = data.get_depth_frame();
    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();
    cv::Mat depth_mat(cv::Size(w, h), CV_16UC1, (void *) depth.get_data(), cv::Mat::AUTO_STEP);
    _depth = depth_mat.clone();
    _color = color_mat.clone();
    return 0;
}

cv::Mat RealSenseWrapper::frame_to_mat(const rs2::frame& f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
        cv::cvtColor(r, r, CV_BGR2RGB);
        return r;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);;
    }

    throw std::runtime_error("Frame format is not supported yet!");
}
