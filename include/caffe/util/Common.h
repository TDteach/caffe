#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
using namespace std;

#include <cctype>
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <cmath>
#include <cassert>

#ifdef _WIN32
#include <direct.h> 
#include <io.h>
#include <windows.h> 
#define MKDIR(dir) CreateDirectory(dir, NULL) 
#elif __linux__
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>
#define MKDIR(dir) mkdir((dir),0755) 
#endif

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/contrib/contrib.hpp"
using namespace cv;


typedef vector< Point2f > FacialPose;

const int INF = (1 << 30) - 1;
const float PI = (float)acos(-1.0);
const float EPS = 1e-5;
const float BASE2 = 1.0/log(2);

