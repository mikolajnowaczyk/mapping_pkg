#ifndef REALSENSE2WRAPPER_H
#define REALSENSE2WRAPPER_H
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <exception>

class RealSenseWrapper{
  
public:
  RealSenseWrapper();
  ~RealSenseWrapper();
  int acquireFramesDepthColorized(cv::Mat& _color, cv::Mat& _depth);
  int acquireFrames(cv::Mat& _color, cv::Mat& _depth);
private:

  int			_width;
  int			_height;
  float         _scale;
  rs2::colorizer color_map;
  rs2::align align_to;
  rs2::pipeline pipe;
  rs2::config cfg;
  cv::Mat frame_to_mat(const rs2::frame& f);
};

#endif /* REALSENSE2WRAPPER_H */

