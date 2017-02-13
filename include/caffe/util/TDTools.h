#ifndef _H_TD_TOOLS_
#define _H_TD_TOOLS_
#include "caffe/util/Common.h"

namespace TDTOOLS {

//about sort
	struct SortItem
	{
		int index;
		float value;
	};

	bool cmpSortItem(SortItem p, SortItem q);

//about dataset
	struct DataSet
	{
		int n;
		vector<Mat> imgs;
		vector<FacialPose> poses;
		vector<FacialPose> inits;
		FacialPose meanposeTgt;
		FacialPose meanposeIni;
	};

	void myGetDataSet(DataSet &data,  string configPath, int numlimit = 0);
	void divideDataSet(DataSet &origSet, DataSet &trainSet, DataSet &testSet, float ratio);
	void catDataSet(DataSet &origSet, DataSet &addSet);
	void fluctuateDataImg(Mat &img, FacialPose &pose, FacialPose &meanpose, string ctrl);
	void fluctuateDataImg(Mat &img, FacialPose &inipose, FacialPose &meanpose, FacialPose &pose, string ctrl);
	void fluctuateDataSet(DataSet &data);
	void augmentDataSet(DataSet &data, int times);

void shuffle(vector<int> &a);

void resize(Mat &img, FacialPose &pose, int win_size);
void resize(vector<Mat> &imgs, vector<FacialPose> &poses, int win_size);
void resize(uchar src[], int Wsrc, int Hsrc, uchar dst[], int Wdst, int Hdst);

void padding(Mat &img, FacialPose &pose, int size);
void padding(vector<Mat> &imgs, vector<FacialPose> &poses, int size);

int itoa(char *dst, int t, int p);

Mat cp2rigid(Point2f* src, Point2f* dst, int M);
Rect pose2box(FacialPose &pose, FacialPose &meanpose, int win_size = 255);
void cropPose(FacialPose &pose, Mat &transMat);
void cropPoses(vector<FacialPose> &poses, vector<Mat> &transMats);
void cropImg(Mat &img, int win_size, Mat &transMat);
void cropImg(Mat &img, int rows, int cols, Mat &transMat);
void cropImgs(vector<Mat> &imgs, int win_size, vector<Mat> &transMats);
void invPose(FacialPose &pose, Mat &trans);
void invPoses(vector<FacialPose> &poses, vector<Mat> &transMats);
void calcTransMat(FacialPose &src, FacialPose &tgt, Mat &transMat);
void calcTransMat(FacialPose &src, FacialPose &tgt, vector<int> &index, Mat &transMat);
void calcTransMats(vector<FacialPose> &poses, FacialPose &meanpose, vector<Mat> &transMats);
void calcTransMats(vector<FacialPose> &src, vector<FacialPose> &tgt, vector<int> &index, vector<Mat> &transMats);
void calcPose(FacialPose pose, float &yaw, float &pitch);

void print_func(const char *a);
bool isTrueBox(FacialPose &pose, Vec4i &box);


//about math
	float sqr(float w);
	float dist(Point2f &a, Point2f &b);
	int rand(int R);
	float randGX();

//about co-fingding-set
	int findcfs(vector<int> &f, int i);

//about show result
	Mat displaySamples(const vector< Mat > & img, const vector< FacialPose > & poses, string file_name, int m = 4);
	void displaySamples(const vector< Mat > & img, string file_name, int m = 4);

}
#endif
