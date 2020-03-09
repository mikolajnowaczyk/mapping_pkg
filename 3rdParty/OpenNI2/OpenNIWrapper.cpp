/*****************************************************************************
*                                                                            *
*  OpenNI 2.x Alpha                                                          *
*  Copyright (C) 2012 PrimeSense Ltd.                                        *
*                                                                            *
*  This file is part of OpenNI.                                              *
*                                                                            *
*  Licensed under the Apache License, Version 2.0 (the "License");           *
*  you may not use this file except in compliance with the License.          *
*  You may obtain a copy of the License at                                   *
*                                                                            *
*      http://www.apache.org/licenses/LICENSE-2.0                            *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*****************************************************************************/
// Undeprecate CRT functions
#ifndef _CRT_SECURE_NO_DEPRECATE 
	#define _CRT_SECURE_NO_DEPRECATE 1
#endif

#include "OpenNIWrapper.h"


//#include <OniSampleUtilities.h>

#define DEFAULT_DISPLAY_MODE	DISPLAY_MODE_DEPTH

#define MIN_NUM_CHUNKS(data_size, chunk_size)	((((data_size)-1) / (chunk_size) + 1))
#define MIN_CHUNKS_SIZE(data_size, chunk_size)	(MIN_NUM_CHUNKS(data_size, chunk_size) * (chunk_size))


OpenNIWrapper::OpenNIWrapper(){
  rc = openni::STATUS_OK;
}
OpenNIWrapper::~OpenNIWrapper(){
    depth.stop();
    depth.destroy();
    color.stop();
    color.destroy();
    device.close();
    //OpenNI:shutdown();
}
int OpenNIWrapper::InitOpenNI (){
	const char* deviceURI = openni::ANY_DEVICE;  //we only have one device any device
	
	rc = openni::OpenNI::initialize();

	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());

	rc = device.open(deviceURI);
	if (rc != openni::STATUS_OK)
	{
		printf("SimpleViewer: Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();
		return 1;
	}

	rc = depth.create(device, openni::SENSOR_DEPTH);
	if (rc == openni::STATUS_OK)
	{
        const openni::SensorInfo* sinfo = device.getSensorInfo(openni::SENSOR_DEPTH); // select index=4 640x480, 30 fps, 1mm
        const openni::Array<openni::VideoMode>& modes = sinfo->getSupportedVideoModes();
        rc = depth.setVideoMode(modes[4]); // 0,1,2,3,6,7    4: 640x480, 30fps, format 100 (1mm)
        if (rc != openni::STATUS_OK) {
            printf("Failed to set depth resolution\n ");
            return rc;
        }
        color.setMirroringEnabled(false);
		rc = depth.start();
		if (rc != openni::STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
			depth.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
	}
    depth.setMirroringEnabled(false);

	rc = color.create(device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
        const openni::SensorInfo* sinfo = device.getSensorInfo(openni::SENSOR_COLOR); // select index=4 640x480, 30 fps, 1mm
        const openni::Array<openni::VideoMode>& modes = sinfo->getSupportedVideoModes();
        rc = color.setVideoMode(modes[9]); // 0,1,2,3,6,7    4: 640x480, 30fps, format 100 (1mm)
        if (rc != openni::STATUS_OK) {
            printf("Failed to set color resolution\n ");
            return rc;
        }
        color.setMirroringEnabled(false);
		rc = color.start();
		if (rc != openni::STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			color.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
    }
    rc = device.setDepthColorSyncEnabled(true);
    if (rc != openni::STATUS_OK) {
        printf("Failed to set depth color sync\n ");
    }

	if (!depth.isValid() || !color.isValid())
	{
		printf("SimpleViewer: No valid streams. Exiting\n");
		openni::OpenNI::shutdown();
		return 2;
	}
    return 0;
}


bool OpenNIWrapper::AcquireDepthFrame(cv::Mat &m){
    rc = depth.readFrame(&depthFrame);
    if (rc != openni::STATUS_OK){
        printf("Wait failed\n");
    }

    if (depthFrame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_1_MM && depthFrame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_100_UM){
        printf("Unexpected frame format\n");
    }

    openni::DepthPixel* pDepth = (openni::DepthPixel*)depthFrame.getData();
    m.create(depthFrame.getHeight(),depthFrame.getWidth(),CV_16SC1);
    memcpy(m.data,pDepth,depthFrame.getStrideInBytes() * depthFrame.getHeight());
    return true;
}

bool OpenNIWrapper::AcquireColorFrame(cv::Mat &m){
    rc = color.readFrame(&colorFrame);
    if (rc != openni::STATUS_OK) {
        printf("Wait failed\n");
    }

    if (colorFrame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888 && colorFrame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_GRAY16) {
        printf("Unexpected frame format\n");
    }

    const openni::RGB888Pixel* pImageRow = (const openni::RGB888Pixel*)colorFrame.getData();
    m.create(colorFrame.getHeight(),colorFrame.getWidth(),CV_8UC3);
    memcpy(m.data,pImageRow,colorFrame.getStrideInBytes() * colorFrame.getHeight());
    cv::cvtColor(m, m, CV_RGB2BGR);
    return true;
}
