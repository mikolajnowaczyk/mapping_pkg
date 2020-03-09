#include "Defs/defs.h"
#include "Mapping/gaussmap.h"
#include "../3rdParty/tinyXML/tinyxml2.h"
#include "Defs/opencv.h"
#include "Defs/grabber_defs.h"
#include "Grabber/grabber.h"
#include "ImageProcessing/procRGBD.h"
#include "Utilities/weightedGraph.h"
//#include <GL/glut.h>
//#include <qapplication.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <unordered_map>
#include "Utilities/recorder.h"
#include <algorithm>
#include <vector>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

//#include <octomap_msgs/conversions.h>
//#include <octomap_msgs/Octomap.h>
//#include <octomap_msgs/GetOctomap.h>
//#include <octomap_ros/conversions.h>
//#include <ellipsoid_msgs/Ellipsoid.h>

//#include <single_ellipsoid_msgs/Ellipsoid.h>
//#include <single_ellipsoid_msgs/EllipsoidArray.h>

#include <opencv2/highgui/highgui_c.h>

#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>

#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>


using namespace std;
namespace enc = sensor_msgs::image_encodings;

bool run_thread = true;

struct Frame
{
    cv::Mat rgbImage;
    cv::Mat depthImage;
    mapping::Mat34 cameraPoseDiff;
    mapping::Mat34 framePose;
    int id;
    bool processed = false;
};

bool operator== (Frame &a, Frame &b)
{
    if(a.cameraPoseDiff.matrix() == b.cameraPoseDiff.matrix() && a.framePose.matrix() == b.framePose.matrix())
    {
        return true;
    }
    else
        return false;
}

//Check what type of Mat
string type2str(int type)
{
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    switch ( depth )
    {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    return r;
}

//Check how many channels is contained in Mat
int no_of_channels(int type)
{
uchar chans = 1 + (type >> CV_CN_SHIFT);
return chans;
}

//get RGB & depth image
class BagImage
{
    cv::Mat RGBImage;
    cv::Mat DepthImage;
    public:
    void callbackRGB(const sensor_msgs::ImageConstPtr& msg)
    {
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        RGBImage = cv_ptr->image;
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    }
    void callbackDepth(const sensor_msgs::ImageConstPtr& msg)
    {
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, enc::TYPE_16UC1); // "mono8" "mono16"  enc::TYPE_16UC1   TUM - enc::TYPE_32FC1
        DepthImage = cv_ptr->image;
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    }
    cv::Mat getImageRGB()
    {
      return RGBImage;
    }
    cv::Mat getImageDepth()
    {
      return DepthImage;
    }
};

//get Pose and KF number
class PoseInfo
{
    public:
    int currentKFNumber = -1;
    mapping::Mat34 currentPosition;
    vector<mapping::Mat34> graph;
    unordered_map<int, mapping::Mat34> graphPS;
    vector<int> ids;
    int currentKFId;
    mutex mutex_poseinfo;

    void callbackCurrentKFNumber(const std_msgs::Int32::ConstPtr& msg)
    {
     const std::lock_guard<std::mutex> guard(mutex_poseinfo);
     currentKFNumber = msg->data;
    }

    void callbackPosition(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
      mapping::Mat34 pose(mapping::Mat34::Identity());
      Eigen::Vector3d tempV(msg->pose.position.x,msg->pose.position.y,msg->pose.position.z);
      pose.matrix().block<3,1>(0,3) = tempV;
      mapping::Quaternion rot;
      rot.w() = msg->pose.orientation.w;
      rot.x() = msg->pose.orientation.x;
      rot.y() = msg->pose.orientation.y;
      rot.z() = msg->pose.orientation.z;
      pose.matrix().block<3,3>(0,0) = rot.matrix();
      const std::lock_guard<std::mutex> guard(mutex_poseinfo);
      currentPosition = pose;
    }

    void callbackKFGraphPoseStamped(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
    //aktualizacja istniejacego frame w graph PS
    mapping::Mat34 pose(mapping::Mat34::Identity());
    Eigen::Vector3d tempV(msg->pose.position.x,msg->pose.position.y,msg->pose.position.z);
    pose.matrix().block<3,1>(0,3) = tempV;
    mapping::Quaternion rot;
    rot.w() = msg->pose.orientation.w;
    rot.x() = msg->pose.orientation.x;
    rot.y() = msg->pose.orientation.y;
    rot.z() = msg->pose.orientation.z;
    pose.matrix().block<3,3>(0,0) = rot.matrix();
    const std::lock_guard<std::mutex> guard(mutex_poseinfo);
    currentKFId = stoi(msg->header.frame_id);
    graphPS[stoi(msg->header.frame_id)] = pose;
    }

    void callbackKFGraphIndexes(const std_msgs::Int32MultiArray::ConstPtr& msg)
    {
      const std::lock_guard<std::mutex> guard(mutex_poseinfo);
      ids.clear();
      for(int i = 0; i < msg->data.size(); i++)
      {
          ids.push_back(msg->data[i]);
      }
    }
    mapping::Mat34 getCurrentPosition()
    {
      return currentPosition;
    }

    int getCurrentKFNumber()
    {
      return currentKFNumber;
    }

    mapping::Mat34 getCurrentKF()
    {
      if(currentKFId>=0)
      {
         if(graphPS.find(currentKFId) != graphPS.end())
             return graphPS[currentKFId];
      }
//      else
//          return mapping::Mat34::Identity();
    }

    int getCurrentKFId()
    {
      return currentKFId;
    }
};

class Manager
{
public:
    BagImage bagimage; //current RGB and Depth images
    PoseInfo poseinfo; //current Position, current Graph as PoseArray and current Graph as PoseStamped to get index of pose
    unordered_map<int, Frame> framesMap;
    unordered_map<int, std::unique_ptr<mapping::Map>> localMaps;
    mutex mutex_framesMap;
    mutex mutex_framesMap2;
    mutex mutex_localMaps;
};

visualization_msgs::MarkerArray createEllipoidsMessage(mapping::Ellipsoid::Seq& _elis, int step = 600)
{
    //too slow
    visualization_msgs::MarkerArray marker_array;
    for(unsigned long i = 0; i < _elis.size(); i+=step)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time();
        marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 1;
        marker.scale.y = 1;
        marker.scale.z = 1;
        marker.color.a = _elis[i].color.getAlpha()/255; // Don't forget to set the alpha!
        marker.color.r = _elis[i].color.getRed()/255;
        marker.color.g = _elis[i].color.getGreen()/255;
        marker.color.b = _elis[i].color.getBlue()/255;

        //points on an ellips
        mapping::Vec3 pos = _elis[i].position;
        mapping::Mat33 covariance = _elis[i].cov;
        mapping::Vec3 normal = _elis[i].normal;
        vector<mapping::Vec3> ellis_points;
        GLfloat		mat[16];
        Eigen::SelfAdjointEigenSolver<mapping::Mat33> es;
        es.compute(covariance);
        mapping::Mat33 m_eigVec(es.eigenvectors());
        m_eigVec = m_eigVec;

        //  A homogeneous transformation matrix, in this order:
        //
        //     0  4  8  12
        //     1  5  9  13
        //     2  6  10 14
        //     3  7  11 15

        mat[3] = mat[7] = mat[11] = 0;
        mat[15] = 1;
        mat[12] = mat[13] = mat[14] = 0;

        mat[0] = (float)m_eigVec(0,0); mat[1] = (float)m_eigVec(1,0); mat[2] = (float)m_eigVec(2,0); mat[12] = (float)pos.x();// New X-axis
        mat[4] = (float)m_eigVec(0,1); mat[5] = (float)m_eigVec(1,1); mat[6] = (float)m_eigVec(2,1);	mat[13] = (float)pos.y();// New X-axis
        mat[8] = (float)m_eigVec(0,2); mat[9] = (float)m_eigVec(1,2); mat[10] = (float)m_eigVec(2,2); mat[14] = (float)pos.z();// New X-axis

        int ellipseDiv = 12;
        mapping::Vec3 normRot;

        // move the normal vector of the plane to the ellipsoid frame
        normRot.vector() = m_eigVec.inverse() * normal.vector();

        if (es.eigenvalues()(0)>0&&es.eigenvalues()(1)>0&&es.eigenvalues()(2)>0)
        {
            // glBegin(GL_TRIANGLE_FAN);
    //        glNormal3d(normal.x(), normal.y(), normal.z());
            // glVertex3d(0, 0, 0); // center of circle
            double meanx = 0.0;
            double meany = 0.0;
            double meanz = 0.0;
            for (int triangleNo = 0; triangleNo <= ellipseDiv; triangleNo++)
            {
                // hint from https://math.stackexchange.com/questions/2505548/intersection-of-an-ellipsoid-and-plane-in-parametric-form
                double t = triangleNo*(2.0*M_PI/double(ellipseDiv));
                double x = sqrt(es.eigenvalues()(0))*sqrt(es.eigenvalues()(2))*normRot.z()*cos(t)/sqrt(es.eigenvalues()(2)*normRot.z()*normRot.z()+es.eigenvalues()(0)*normRot.x()*normRot.x());
                double y = sqrt(es.eigenvalues()(1))*sqrt(es.eigenvalues()(2))*normRot.z()*sin(t)/sqrt(es.eigenvalues()(2)*normRot.z()*normRot.z()+es.eigenvalues()(1)*normRot.y()*normRot.y());
                double z = (-normRot.x()*x-normRot.y()*y)/normRot.z();
    //            double z = -sqrt(es.eigenvalues()(0))*sqrt(es.eigenvalues()(2))*normRot.x()*cos(t)/sqrt(es.eigenvalues()(2)*normRot.z()*normRot.z()+es.eigenvalues()(0)*normRot.x()*normRot.x()) - sqrt(es.eigenvalues()(1))*sqrt(es.eigenvalues()(2))*normRot.y()*sin(t)/sqrt(es.eigenvalues()(2)*normRot.z()*normRot.z()+es.eigenvalues()(1)*normRot.y()*normRot.y());
                if (!std::isnan(x)&&!std::isnan(y)&&!std::isnan(z))
                {
                    mapping::Vec3 temp(x,y,z);
                    ellis_points.push_back(temp);
                    meanx += x / ellipseDiv;
                    meany += y / ellipseDiv;
                    meanz += z / ellipseDiv;
                    cout<<"ADDING ELLIPSE WITH CENTER IN: "<<_elis[i].position.x()<<" "<<_elis[i].position.y()<<" "<<_elis[i].position.z()<<endl;
                }
            }
            for(int i = 0; i < ellipseDiv; i++)
            {
                geometry_msgs::Point tempp;
                tempp.x = meanx;
                tempp.y = meany;
                tempp.z = meanz;
                marker.points.push_back(tempp);
                tempp.x = ellis_points[i].x();
                tempp.y = ellis_points[i].y();
                tempp.z = ellis_points[i].z();
                marker.points.push_back(tempp);
                tempp.x = ellis_points[(i+1)%ellipseDiv].x();
                tempp.y = ellis_points[(i+1)%ellipseDiv].y();
                tempp.z = ellis_points[(i+1)%ellipseDiv].z();
                marker.points.push_back(tempp);
            }
        }
        marker_array.markers.push_back(marker);
    }
    return marker_array;
}

void PublishEllipsoidsVisu(/*vector<unique_ptr<mapping::Map>>& localMaps*/Manager& manager, ros::Publisher& vis_pub, ros::Publisher vis_pub3)
{
    //    cout<<"Trying to send ellipsoids!"<<endl;

    int counter = 0;
    //    std::map<int,mapping::Ellipsoid> ellipsoids_raw;
    //    manager.poseinfo.mutex_poseinfo.lock();
     vector<int> temp_poseinfo_ids;
    {
        const std::lock_guard<std::mutex> guard(manager.poseinfo.mutex_poseinfo);
        temp_poseinfo_ids = manager.poseinfo.ids;
    }
    //    manager.poseinfo.mutex_poseinfo.unlock();
    for(auto index: temp_poseinfo_ids)
    {
    //        localMaps[j]->getRawEllipsoids(ellipsoids_raw);
    //        cout<<"test visu "<<j<<endl;
        visualization_msgs::MarkerArray marker_array;
        mapping::Ellipsoid::Seq _elis;
        bool temp1;
        {
            const std::lock_guard<std::mutex> guard2(manager.mutex_localMaps);
            temp1 = (manager.localMaps.find(index) == manager.localMaps.end());
        }
    //        manager.mutex_manager.unlock();
        if(temp1)
            continue;
    //        manager.mutex_manager.lock();
        {
            const std::lock_guard<std::mutex> guard3(manager.mutex_localMaps);
            manager.localMaps[index]->getEllipsoidsGlobal(_elis);
            cout<<"Thread get "<<_elis.size()<<" elis!"<<endl;
        }
//        vis_pub.publish( createEllipoidsMessage(_elis));
//            manager.mutex_manager.unlock();
             for(unsigned int i = 0; i < _elis.size(); i+=4)
             {
     //            cout<<"test visu2 "<<i<<endl;
                 counter++;
                 visualization_msgs::Marker marker;
                 marker.header.frame_id = "map";
                 marker.header.stamp = ros::Time();
                 //marker.ns = "my_namespace";
                 marker.id = counter;
                 marker.type = visualization_msgs::Marker::CUBE;
                 marker.action = visualization_msgs::Marker::ADD;

                 //vertical display
     //            marker.pose.position.x = _elis[i].position.x();
     //            marker.pose.position.y = _elis[i].position.y();
     //            marker.pose.position.z = _elis[i].position.z();
                 //horizontal display
                 marker.pose.position.x = _elis[i].position.z();
                 marker.pose.position.y = -1.0 * _elis[i].position.x();
                 marker.pose.position.z = -1.0 * _elis[i].position.y();

                 marker.pose.orientation.x = 0.0;
                 marker.pose.orientation.y = 0.0;
                 marker.pose.orientation.z = 0.0;
                 marker.pose.orientation.w = 1.0;

                 float param = 0.018;
                 marker.scale.x = param;
                 marker.scale.y = param;
                 marker.scale.z = param;

                 //ICL
                  marker.color.a = _elis[i].color.getAlpha()/255; // Don't forget to set the alpha!
                  marker.color.r = _elis[i].color.getRed()/255;
                  marker.color.g = _elis[i].color.getGreen()/255;
                  marker.color.b = _elis[i].color.getBlue()/255;

                    //TUM
//                  marker.color.a = _elis[i].color.getAlpha()/255; // Don't forget to set the alpha!
//                  marker.color.r = _elis[i].color.getBlue()/255;
//                  marker.color.g = _elis[i].color.getGreen()/255;
//                  marker.color.b = _elis[i].color.getRed()/255;

                 marker_array.markers.push_back(marker);
     //            vis_pub3.publish(marker);
             }
        vis_pub.publish( marker_array);
        ros::spinOnce();
    }
    //    cout<<"Sent "<<marker_array.markers.size()<< " markers!" << endl;
}

void threadMapUpdate(Manager &manager, grabber::CameraModel &asusModel)
{
    vector<int> temp_poseinfo_ids;
    {
        const std::lock_guard<std::mutex> guard(manager.poseinfo.mutex_poseinfo);
        temp_poseinfo_ids = manager.poseinfo.ids;
    }
    for(auto index: temp_poseinfo_ids)
    {
        bool temp; //check if index is in range and if frame need processing
        {
            const std::lock_guard<std::mutex> guard(manager.mutex_framesMap);
            const std::lock_guard<std::mutex> guard2(manager.mutex_localMaps);
            temp = ((manager.framesMap.find(index) != manager.framesMap.end()) && (manager.localMaps.find(index) != manager.localMaps.end()));
            temp = temp && (!manager.framesMap[index].processed);
        }
        if(temp)
        {
    //            cout<<"Processing new frame with index "<<index<<endl;
            mapping::Mat34 camPoseDiff;
            {
                const std::lock_guard<std::mutex> guard(manager.mutex_framesMap);
                camPoseDiff = manager.framesMap[index].cameraPoseDiff;
    //                cout<<"2 manager.framesMap[index].processed = "<<manager.framesMap[index].processed<<" index ="<<index<<endl;
                manager.framesMap[index].processed = true;
            }
            {
                const std::lock_guard<std::mutex> guard(manager.mutex_framesMap);
    //                cout<<"Doing median depth!"<< " ((manager.framesMap.find(index) != manager.framesMap.end()) && (manager.localMaps.find(index) != manager.localMaps.end())) = "<<((manager.framesMap.find(index) != manager.framesMap.end()) && (manager.localMaps.find(index) != manager.localMaps.end()))<<" !manager.framesMap[index].processed = "<< !manager.framesMap[index].processed<<endl;
    //                cout<<"Final test: "<<(((manager.framesMap.find(index) != manager.framesMap.end()) && (manager.localMaps.find(index) != manager.localMaps.end())) && (!manager.framesMap[index].processed))<<endl;
//                grabber::Grabber::filterDepthImage(manager.framesMap[index].depthImage, manager.framesMap[index].depthImage, "median", 5); //15
    //                cout<<"3 manager.framesMap[index].processed = "<<manager.framesMap[index].processed<<" index ="<<index<<endl;
            }
            grabber::PointCloud secondCloud;
            {
                const std::lock_guard<std::mutex> guard(manager.mutex_framesMap);
                secondCloud = asusModel.depth2cloud(manager.framesMap[index].depthImage, manager.framesMap[index].rgbImage, (1.0/1000.0), 0.0, 1000000.0); //0.0002 TUM - 5000.0 else - 1000.0
    //                cout<<"4 manager.framesMap[index].processed = "<<manager.framesMap[index].processed<<" index ="<<index<<endl;
            }
            grabber::PointCloud ontoFirst;
            pcl::transformPointCloud (secondCloud, ontoFirst, camPoseDiff);
            mapping::Ellipsoid::Seq ellipsoids;
            {
                const std::lock_guard<std::mutex> guard(manager.mutex_localMaps);
                ellipsoids = manager.localMaps[index]->updateMap(ontoFirst, camPoseDiff, asusModel);
    //                cout<<"5 manager.framesMap[index].processed = "<<manager.framesMap[index].processed<<" index ="<<index<<endl;
            }
            int eliNo=0;
            std::map<int,mapping::Ellipsoid> _ellipsoids;
            for (const auto eli : ellipsoids)
            {
                _ellipsoids[eliNo] = eli;
                eliNo++;
            }
            {
                const std::lock_guard<std::mutex> guard(manager.mutex_framesMap);
                const std::lock_guard<std::mutex> guard2(manager.mutex_localMaps);
                manager.localMaps[index]->setEllipsoids(_ellipsoids, manager.framesMap[index].framePose);
//                cout<<"Set "<<_ellipsoids.size()<<" ellipsoids"<<endl;
            }
        }
    }
    cout<<"thread work is done!"<<endl;
    {
        const std::lock_guard<std::mutex> guard(manager.mutex_framesMap);
        const std::lock_guard<std::mutex> guard2(manager.mutex_framesMap2);
    }
    run_thread = true;
}

void PublishGraph(Manager &manager, ros::Publisher graph_publisher, ros::Publisher pose_pub)
{
    geometry_msgs::PoseStamped pose_msg1;
    pose_msg1.header.frame_id = "base_link";

    mapping::Mat34 pose1 = manager.poseinfo.getCurrentPosition();

    pose_msg1.pose.position.x = pose1(0,3);
    pose_msg1.pose.position.y = pose1(1,3);
    pose_msg1.pose.position.z = pose1(2,3);

    mapping::Quaternion quat1(pose1.rotation());

    pose_msg1.pose.orientation.w = quat1.w();
    pose_msg1.pose.orientation.x = quat1.x();
    pose_msg1.pose.orientation.y = quat1.y();
    pose_msg1.pose.orientation.z = quat1.z();

    pose_pub.publish(pose_msg1);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mapping_pkg_node");
    ros::NodeHandle nh("~");
    std::string name_of_node = ros::this_node::getName();
    Manager manager;
    //set subscribers
    image_transport::ImageTransport it(nh);
    //    image_transport::Subscriber rgb_sub = it.subscribe("/camera/rgb/image_color", 1, &BagImage::callbackRGB, &bagimage);
    //    image_transport::Subscriber depth_sub = it.subscribe("/camera/depth/image_raw", 1, &BagImage::callbackDepth, &bagimage);

    //astra
    image_transport::Subscriber rgb_sub = it.subscribe("/camera/rgb/image_raw", 1, &BagImage::callbackRGB, &manager.bagimage);
    image_transport::Subscriber depth_sub = it.subscribe("/camera/depth/image_raw", 1, &BagImage::callbackDepth, &manager.bagimage);

    ros::Subscriber pose_sub = nh.subscribe("/RGBD_Modified/pose", 1, &PoseInfo::callbackPosition, &manager.poseinfo);
    ros::Subscriber pose_number_sub = nh.subscribe("/RGBD_Modified/pose_number", 1, &PoseInfo::callbackCurrentKFNumber, &manager.poseinfo);
    //    ros::Subscriber pose_graph_sub = nh.subscribe("/RGBD_Modified/kf_graph", 1, &PoseInfo::callbackKFGraph, &manager.poseinfo);
    ros::Subscriber graph_indexes_sub = nh.subscribe("/RGBD_Modified/kf_graph_indexes", 1, &PoseInfo::callbackKFGraphIndexes, &manager.poseinfo);
    ros::Subscriber grap_pose_stamped_sub = nh.subscribe("/RGBD_Modified/kf_graph_pose_stamped", 500, &PoseInfo::callbackKFGraphPoseStamped, &manager.poseinfo);

    //    ros::Publisher ellipsoids_raw_pub = nh.advertise<visualization_msgs::MarkerArray>(name_of_node + "/ellipsoidsRaw", 1);
    ros::Publisher vis_pub_3 = nh.advertise<visualization_msgs::Marker>( name_of_node + "/visualization_marker", 250);
    ros::Publisher vis_pub = nh.advertise<visualization_msgs::MarkerArray>( name_of_node + "/visualization_marker_array", 1);
    ros::Publisher pose_publisher = nh.advertise<geometry_msgs::PoseStamped> (name_of_node + "/pose", 1);
    //    ros::Publisher pose_KF_publisher = nh.advertise<geometry_msgs::PoseStamped> (name_of_node + "/poseKF", 1);
    ros::Publisher local_map_poses_publisher = nh.advertise<geometry_msgs::PoseArray> (name_of_node + "/local_map_poses", 1);
    ros::Publisher vis_pub_2 = nh.advertise<visualization_msgs::Marker>( name_of_node + "/graph_visualization", 1);
    //pose_number_publisher = nh.advertise<std_msgs::Int32> (name_of_node + "/pose_number", 1);

    std::srand((unsigned int)time(NULL));

    //cout<<"Subscribers done!"<<endl;

    setlocale(LC_NUMERIC,"C");
    /*tinyxml2::XMLDocument config;
    config.LoadFile("../../resources/configGlobal.xml");
    if (config.ErrorID())
        std::cout << "unable to load config file.\n";

    cout<<"Config done!"<<endl;*/

    int mapSize = 512;
    int localMapSize = 256;
    double resolution = 0.1;
    double raytraceFactor = 0.08;
    int pointThreshold = 10;

    grabber::CameraModel asusModel("/home/mikolaj/Sources/mapping/resources/KinectModel.xml");

    ros::Rate loop_rate(4); //10 hz
    Frame tempFrame;
    //bool startThread = true;
    while(ros::ok())
    {
      int temp_getCurretnKFNumber;
      {
        const std::lock_guard<std::mutex> guard(manager.poseinfo.mutex_poseinfo);
        temp_getCurretnKFNumber = manager.poseinfo.getCurrentKFNumber();
      }
      cout<<"Current kf number = "<<temp_getCurretnKFNumber<<endl;
      if(temp_getCurretnKFNumber != -1)
      {
         cout<<"while loop!"<<endl;
          tempFrame.rgbImage = manager.bagimage.getImageRGB();
          tempFrame.depthImage = manager.bagimage.getImageDepth();
          if(manager.poseinfo.graphPS.find(manager.poseinfo.getCurrentKFId()) != manager.poseinfo.graphPS.end())
          {
            const std::lock_guard<std::mutex> guard(manager.poseinfo.mutex_poseinfo);
            tempFrame.framePose = manager.poseinfo.getCurrentKF();
            tempFrame.id = manager.poseinfo.getCurrentKFId();
            tempFrame.cameraPoseDiff = manager.poseinfo.getCurrentKF().inverse() * manager.poseinfo.getCurrentPosition();
          }
          //new localMap
          vector<int> temp_poseinfo_ids;
          {
              const std::lock_guard<std::mutex> guard(manager.poseinfo.mutex_poseinfo);
              temp_poseinfo_ids = manager.poseinfo.ids;
          }
          cout<<"Updating map poses!"<<endl;
          cout<<"Manager poseinfo ids size = "<<manager.poseinfo.ids.size()<<endl;
          for(auto index : temp_poseinfo_ids)
          {
              const std::lock_guard<std::mutex> guard2(manager.poseinfo.mutex_poseinfo);
              const std::lock_guard<std::mutex> guard(manager.mutex_localMaps);
              if(manager.localMaps.find(index) == manager.localMaps.end())
              {
                  //didn't find a localMap with current id and adding new MapGauss
                  manager.localMaps[index] = mapping::createMapGauss(localMapSize, resolution, raytraceFactor, pointThreshold);
                  cout<<"New local map added"<<endl;
              }
              manager.localMaps[index]->updateMapPose(manager.poseinfo.graphPS[index]);

          }
          if(manager.poseinfo.graphPS.find(manager.poseinfo.getCurrentKFId()) != manager.poseinfo.graphPS.end())
          {
              const std::lock_guard<std::mutex> guard(manager.mutex_framesMap);
              manager.framesMap[tempFrame.id] = tempFrame;
          }
          if(run_thread == true)
          {
              thread th1(&threadMapUpdate, std::ref(manager), ref(asusModel));
              th1.detach();
              run_thread = false;
          }
          PublishEllipsoidsVisu(ref(manager), vis_pub, vis_pub_3);
      }//end if poseinfo.getCurrentKFNumber() != -1
      {
        const std::lock_guard<std::mutex> guard(manager.poseinfo.mutex_poseinfo);
        const std::lock_guard<std::mutex> guard2(manager.mutex_framesMap);
        cout<<"manager.localMaps.size() = "<<manager.localMaps.size()<<endl;
        cout<<"manager.poseinfo.ids.size() = "<<manager.poseinfo.ids.size()<<endl;
        PublishGraph(ref(manager), local_map_poses_publisher, pose_publisher);
      }
      ros::spinOnce();
      loop_rate.sleep();
    }//end while

    //===========================================

//    cout<<"Creating cloud for ICL ..."<<endl;
//    ros::Duration(3.5).sleep();
//    // Calculate total number of points
//    int total_number_points = 0;
//    int counter = 0;
//    vector<int> temp_poseinfo_ids;
//    {
//        const std::lock_guard<std::mutex> guard(manager.poseinfo.mutex_poseinfo);
//        temp_poseinfo_ids = manager.poseinfo.ids;
//    }
//    for(auto index: temp_poseinfo_ids)
//    {
//        mapping::Ellipsoid::Seq _elis;
//        bool temp1;
//        {
//            const std::lock_guard<std::mutex> guard2(manager.mutex_localMaps);
//            temp1 = (manager.localMaps.find(index) == manager.localMaps.end());
//        }
//        if(temp1)
//            continue;
//        {
//            const std::lock_guard<std::mutex> guard3(manager.mutex_localMaps);
//            manager.localMaps[index]->getEllipsoidsGlobal(_elis);
//        }
//        total_number_points += _elis.size();
//    }
//    cout<<"Total number of points is " <<total_number_points <<endl;

//    // Fill in the cloud data
//    pcl::PointCloud<pcl::PointXYZRGB> cloud;
//    cloud.width    = total_number_points + 1;
//    cloud.height   = 1;
//    cloud.is_dense = false;
//    cloud.points.resize (cloud.width * cloud.height);
//    cloud.sensor_orientation_ = Eigen::Quaternionf (0, 0, 0, 1);
//    for(auto index: temp_poseinfo_ids)
//    {
//        mapping::Ellipsoid::Seq _elis;
//        bool temp1;
//        {
//            const std::lock_guard<std::mutex> guard2(manager.mutex_localMaps);
//            temp1 = (manager.localMaps.find(index) == manager.localMaps.end());
//        }
//        if(temp1)
//            continue;
//        {
//            const std::lock_guard<std::mutex> guard3(manager.mutex_localMaps);
//            manager.localMaps[index]->getEllipsoidsGlobal(_elis);
//            cout<<_elis.size()<<" ellipsoids added!"<<endl;
//        }
//        for(int i = 0; i < _elis.size(); i++)
//        {
//            cout<<"Trying to add "<<counter<<" point of total "<<total_number_points<<endl;
////            cloud.points[i].x = _elis[i].position.z();
////            cloud.points[i].y = -1.0 * _elis[i].position.x();
////            cloud.points[i].z = -1.0 * _elis[i].position.y();
//            cloud.points[counter].x = _elis[i].position.x();
//            cloud.points[counter].y = _elis[i].position.y();
//            cloud.points[counter].z = _elis[i].position.z();
//            cloud.points[counter].r = _elis[i].color.getRed()/255;
//            cloud.points[counter].g = _elis[i].color.getGreen()/255;
//            cloud.points[counter].b = _elis[i].color.getBlue()/255;
////            cloud.points[i+counter].a = _elis[i].color.getAlpha()/255;
//            cout<<"Added point: "<<counter<< " of total "<<total_number_points<<endl;
//            ++counter;
//        }
//        cout<<"end of i loop"<<endl;
//        //set point
//    }
//    cout<<"Trying to save .."<<endl;

////    pcl::PCLPointCloud2 point_cloud2;
//    //pcl::PointCloud<pcl::PointXYZRGBA> point_cloud;

//    //pcl::toPCLPointCloud2(cloud, point_cloud2);

//    pcl::io::savePCDFileASCII ("/home/mikolaj/catkin_ws/src/mapping_pkg/test_pcd.pcd", cloud);
//    pcl::io::savePLYFile ("/home/mikolaj/catkin_ws/src/mapping_pkg/icl4_cloud.ply", cloud);
//    //pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);

    cout<<"Done"<<endl;
}
