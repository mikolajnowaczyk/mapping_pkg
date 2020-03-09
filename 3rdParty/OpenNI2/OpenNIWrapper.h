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

#ifndef OPENNI_WRAPPER_H_INCLUDED
#define OPENNI_WRAPPER_H_INCLUDED

#include "Defs/openni.h"
#include "Defs/opencv.h"

#define MAX_DEPTH 10000

enum DisplayModes
{
	DISPLAY_MODE_OVERLAY,
	DISPLAY_MODE_DEPTH,
	DISPLAY_MODE_IMAGE
};

class OpenNIWrapper{
  
public:
  OpenNIWrapper();
  ~OpenNIWrapper();
  int InitOpenNI ();
  bool OpenOpenNI();
  bool CloseOpenNI();
  bool AcquireFrame();
  bool AcquireDepthFrame(cv::Mat &m);
  bool AcquireColorFrame(cv::Mat &m);
  int InitDisplayMode();
  int getSizeX();
  int getSizeY();
  int colorMode();
  int bitsPerPixel(); 
  openni::Status rc;
  openni::Device device;
  openni::VideoStream depth;
  openni::VideoStream color;
  /// ONI Frame ref
  openni::VideoFrameRef depthFrame;
  openni::VideoFrameRef colorFrame;
  
private:
  
  float			m_pDepthHist[MAX_DEPTH];
  char			m_strSampleName[ONI_MAX_STR];
  unsigned int		m_nTexMapX;
  unsigned int		m_nTexMapY;
  DisplayModes		m_eViewState;
  openni::RGB888Pixel*	m_pTexMap;
  int			m_width;
  int			m_height;
  
};


#endif // OPENNI_WRAPPER_H_INCLUDED
