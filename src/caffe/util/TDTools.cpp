#include "caffe/util/TDTools.h"

void TDTOOLS::resize(Mat &img, FacialPose &pose, int win_size)
{
	float xratio__;
	float yratio__;

	xratio__ = (float)win_size/(float)img.cols;
	yratio__ = (float)win_size/(float)img.rows;

	resize(img,img,Size(win_size,win_size));

	for (int j = 0; j < pose.size(); j++) {
		pose[j].x *= xratio__;
		pose[j].y *= yratio__;
	}

	

}
void TDTOOLS::resize(vector<Mat> &imgs, vector<FacialPose> &poses, int win_size)
{
	#pragma omp parallel for num_threads(20)
	for (int i = 0; i < imgs.size(); i++) {
		resize(imgs[i],poses[i],win_size);
	}
}

void TDTOOLS::padding(Mat &img, FacialPose &pose, int size)
{
	int r = img.rows;
	int c = img.cols;
	Mat rst__;
	rst__ = Mat::zeros(r+size*2,c+size*2,CV_8U);

	img.copyTo(rst__(Rect(size,size,c,r)));
	img.release();
	img = rst__;

	for (int i = 0; i < pose.size(); i++) {
		pose[i].x += size;
		pose[i].y += size;
	}
}
void TDTOOLS::padding(vector<Mat> &imgs, vector<FacialPose> &poses, int size)
{
	assert(poses.size() >= imgs.size());
	#pragma omp parallel for num_threads(20)
	for (int i = 0; i < imgs.size(); i++) {
		padding(imgs[i], poses[i], size);
	}

}



void TDTOOLS::cropPose(FacialPose &pose, Mat &transMat)
{
	float tx__, ty__;
	for (int j = 0; j < pose.size(); j++) {
		tx__ = pose[j].x;
		ty__ = pose[j].y;
		pose[j].x = transMat.at<float>(0,0)*tx__+transMat.at<float>(0,1)*ty__+transMat.at<float>(0,2);
		pose[j].y = transMat.at<float>(1,0)*tx__+transMat.at<float>(1,1)*ty__+transMat.at<float>(1,2);
	}
}
void TDTOOLS::cropPoses(vector<FacialPose> &poses, vector<Mat> &transMats)
{
	for (int i = 0; i < poses.size(); i++) {		
		cropPose(poses[i], transMats[i]);
	}
}
void TDTOOLS::cropImg(Mat &img, int win_size, Mat &transMat)
{
	cropImg(img,win_size, win_size, transMat);
}
void TDTOOLS::cropImg(Mat &img, int rows, int cols, Mat &transMat)
{
	Mat temp__(rows,cols,CV_8U);
	warpAffine(img, temp__, transMat, temp__.size());
	img.release();
	img = temp__;
}
void TDTOOLS::cropImgs(vector<Mat> &imgs, int win_size, vector<Mat> &transMats)
{
	for (int i = 0; i < imgs.size(); i++) {
		cropImg(imgs[i],win_size,transMats[i]);
	}
}

void TDTOOLS::invPose(FacialPose &pose, Mat &trans)
{
	float d;

	d = trans.at<float>(0,0)*trans.at<float>(0,0) + trans.at<float>(0,1)*trans.at<float>(0,1);

	for (int j = 0; j < pose.size(); j++) {
		pose[j].x -= trans.at<float>(0,2);
		pose[j].y -= trans.at<float>(1,2);

		float x = pose[j].x*trans.at<float>(0,0)-pose[j].y*trans.at<float>(0,1);
		float y = -pose[j].x*trans.at<float>(1,0)+pose[j].y*trans.at<float>(1,1);

		pose[j].x = x/d;
		pose[j].y = y/d;
	}

}
void TDTOOLS::invPoses(vector<FacialPose> &poses, vector<Mat> &transMats)
{
	int n_imgs__ = transMats.size();
	int i;
	
	for (i = 0; i < n_imgs__; i++) {
		invPose(poses[i], transMats[i]);
	}
}

void TDTOOLS::calcTransMat(FacialPose &src, FacialPose &tgt, Mat &transMat)
{
	int l = min(src.size(), tgt.size());
	transMat = cp2rigid(&(src[0]), &(tgt[0]), l);
}
void TDTOOLS::calcTransMat(FacialPose &src, FacialPose &tgt, vector<int> &index, Mat &transMat)
{
	int j;
	int l = index.size();
	FacialPose tmp__;
	
	
	tmp__.resize(l);
	for (j = 0; j < l; j++)
		tmp__[j] = src[index[j]];

	transMat = cp2rigid(&(tmp__[0]), &(tgt[0]), l);
}
void TDTOOLS::calcTransMats(vector<FacialPose> &poses, FacialPose &meanpose, vector<Mat> &transMats)
{
	int n_poses__ = poses.size();
	int i;
	int l = meanpose.size();
	FacialPose pose__;
	
	pose__.resize(l);
	transMats.resize(n_poses__);
	for (i = 0; i < n_poses__; i++){
		pose__[0] = poses[i][0];
		pose__[1] = poses[i][1];
		pose__[2] = poses[i][2];
		pose__[3] = poses[i][3];
		pose__[4] = poses[i][4];
		
		transMats[i] = cp2rigid(&(pose__[0]), &(meanpose[0]),l);
	}
}
void TDTOOLS::calcTransMats(vector<FacialPose> &src, vector<FacialPose> &tgt, vector<int> &index, vector<Mat> &transMats)
{
	int n__ = src.size();
	int i;
	
	transMats.resize(n__);
	for (i = 0; i < n__; i++){
		calcTransMat(src[i], tgt[i], index, transMats[i]);
	}
}
void TDTOOLS::calcPose(FacialPose pose, float &yaw, float &pitch)
{

	int l = pose.size();
	assert(l == 21);


	FacialPose mean_pose;
	mean_pose.push_back(cv::Point2f(13.7033f, 16.671f));
	mean_pose.push_back(cv::Point2f(30.4758f, 12.5936f));
	mean_pose.push_back(cv::Point2f(49.4828f, 17.0663f));
	mean_pose.push_back(cv::Point2f(77.3963f, 16.2806f));
	mean_pose.push_back(cv::Point2f(96.0747f, 11.1543f));
	mean_pose.push_back(cv::Point2f(113.412f, 14.3097f));
	mean_pose.push_back(cv::Point2f(22.765f, 29.7737f));
	mean_pose.push_back(cv::Point2f(47.6365f, 30.3461f));
	mean_pose.push_back(cv::Point2f(80.6844f, 29.6988f));
	mean_pose.push_back(cv::Point2f(105.275f, 28.1353f));
	mean_pose.push_back(cv::Point2f(48.1454f, 65.2725f));
	mean_pose.push_back(cv::Point2f(64.8271f, 73.7766f));
	mean_pose.push_back(cv::Point2f(81.1373f, 64.8805f));
	mean_pose.push_back(cv::Point2f(64.7944f, 84.6744f));
	mean_pose.push_back(cv::Point2f(65.2175f, 92.9841f));
	mean_pose.push_back(cv::Point2f(65.4613f, 104.054f));
	mean_pose.push_back(cv::Point2f(34.9618f, 28.0176f));
	mean_pose.push_back(cv::Point2f(92.3712f, 26.8753f));
	mean_pose.push_back(cv::Point2f(63.75f, 63.75f));
	mean_pose.push_back(cv::Point2f(36.5422f, 92.8654f));
	mean_pose.push_back(cv::Point2f(87.956f, 91.9989f));

	float parameter[2][43] = { 145.297, -235.475, 163.508, -153.979, 143.044,
		-64.2536, 147.299, 72.9581, 170.414, 162.093, 156.671, 245.589, 81.9347,
		-192.14, 78.3035, -71.5685, 81.2156, 87.8689, 89.9782, 206.867,
		-90.4048, -69.4082, -131.019, 11.2424, -88.4095, 90.0975, -184.983,
		11.0926, -224.492, 13.1861, -277.709, 14.3299, 89.7954, -133.888,
		95.1383, 145.051, -83.7443, 6.09385, -223.148, -125.762, -219.075,
		123.343, -1648.46, -97.6281, 29.8396, -86.9529, 79.0641, -99.0863,
		131.197, -96.8538, 210.432, -81.571, 263.099, -91.2352, 310.788,
		-134.538, 56.6912, -136.435, 125.511, -134.699, 219.463, -130.342,
		288.851, -234.42, 128.295, -258.654, 173.577, -233.631, 220.719,
		-289.166, 174.881, -312.64, 175.972, -343.741, 176.929, -130.098,
		90.8279, -126.095, 252.344, -230.425, 173.079, -312.174, 94.8727,
		-309.98, 239.619, 75948.8 };



	Mat trans;
	

	calcTransMat(pose, mean_pose, trans);
	cropPose(pose, trans);

	yaw = pitch = 0;
	for (int i = 0; i < l; i++) {
		yaw += pose[i].x*parameter[0][i*2]+pose[i].y*parameter[0][i*2+1];
		pitch += pose[i].x*parameter[1][i*2]+pose[i].y*parameter[1][i*2+1];
	}
	yaw += parameter[0][42];
	pitch += parameter[1][42];
}

bool TDTOOLS::isTrueBox(FacialPose & pose, Vec4i & box)
{
	if (box[0] < 0)
		return false;
	if (max(pose[1].x, pose[4].x) > box[0] + box[2])
		return false;
	if (min(pose[0].x, pose[3].x) < box[0])
		return false;
	if (min(pose[3].y, pose[4].y) > box[1] + box[3])
		return false;
	if (max(pose[0].y, pose[1].y) < box[1])
		return false;	
	return true;
}
void TDTOOLS::myGetDataSet(DataSet &data, string configPath, int numlimit)
{
	FILE *f;
	char buf[256];
	string work_dir;	
	string poseIni_list;
	string poseTgt_list;
	string meanTgt_path;
	string meanIni_path;
	string path_list;
	int LTgt;
	int LIni;

	vector<string> namelist;
	int num;

	f = fopen(configPath.c_str(),"r");
	fscanf(f,"%s",buf); work_dir = string(buf);
	fscanf(f,"%s",buf); path_list = string(buf);
	fscanf(f,"%d",&LTgt);
	if (LTgt > 0) {
		fscanf(f,"%s",buf); poseTgt_list = string(buf);
		fscanf(f,"%s",buf); meanTgt_path = string(buf);
	}
	fscanf(f,"%d",&LIni);
	if (LIni > 0) {
		fscanf(f,"%s",buf); poseIni_list = string(buf);
		fscanf(f,"%s",buf); meanIni_path = string(buf);
	}
	fclose(f);


	namelist.clear();
	f = fopen(path_list.c_str(),"r");
	while (fscanf(f, "%s", buf) != EOF) {
		namelist.push_back(string(buf));
	}
	fclose(f);

	num = namelist.size(); 
	if (numlimit > 0) num = min(num, numlimit);
	cout << "total " << num << " imgs in namelist to use" <<endl;


	data.n = num;


	if (LTgt > 0) {
		f = fopen(poseTgt_list.c_str(), "r");
		data.poses.resize(num);
		for (int i = 0; i < num; i++) {
			data.poses[i].resize(LTgt);
			for (int j = 0; j <data.poses[i].size(); j++) {
				fscanf(f,"%f%f", &data.poses[i][j].x, &data.poses[i][j].y);
			}
		}
		fclose(f);
		f = fopen(meanTgt_path.c_str(), "r");
		data.meanposeTgt.resize(LTgt);
		for (int i = 0; i < LTgt; i++) {
			fscanf(f,"%f%f",&data.meanposeTgt[i].x, &data.meanposeTgt[i].y);
		}
		fclose(f);
	}


	if (LIni > 0) {

		f = fopen(poseIni_list.c_str(), "r");
		data.inits.resize(num);
		for (int i = 0; i < num; i++) {
			data.inits[i].resize(LIni);
			for (int j = 0; j < LIni; j++) {
				fscanf(f, "%f%f", &data.inits[i][j].x, &data.inits[i][j].y);
			}
		}
		fclose(f);

		f = fopen(meanIni_path.c_str(), "r");
		data.meanposeIni.resize(LIni);
		for (int i = 0; i < LIni; i++) {
			fscanf(f,"%f%f",&data.meanposeIni[i].x, &data.meanposeIni[i].y);
		}
		fclose(f);
	}
	else if (LTgt > 0) {
		data.meanposeIni = data.meanposeTgt;
		data.inits.resize(num);
		for (int i = 0; i < num; i++) {
			data.inits[i] = data.poses[i];
		}
	}

	cout << "poses read over " << endl;


	vector<int> okimg;
	okimg.resize(num);

	data.imgs.resize(num);
#pragma omp parallel for num_threads(20)
	for (int i = 0; i < num; i++) {
		Mat tm = imread(work_dir+namelist[i]);
		if (tm.type() == CV_8UC3) {
			cvtColor(tm,tm,CV_BGR2GRAY);
		}

		if (tm.type() != CV_8U || tm.rows == 0) {
			okimg[i] = 0;
		}
		else {
			okimg[i] = 1;
			data.imgs[i] = tm;
		}
	}
	cout << "imgs read over" <<endl;


	int z = 0;
	for (int i = 0; i < num; i++) {
		if (okimg[i] == 0) continue;
		data.imgs[z] = data.imgs[i];
		if (data.inits.size() > 0)
			data.inits[z] = data.inits[i];
		if (data.poses.size() > 0)
			data.poses[z] = data.poses[i];
		z++;
	}

	data.imgs.resize(z);
	if (data.inits.size() > 0)
		data.inits.resize(z);
	if (data.poses.size() > 0)
		data.poses.resize(z);
	data.n = z;
	cout << "clean imgs to " << data.n << endl;
/*
	cout << " haha " <<endl;
	int showN = 1;
	vector<Mat> showI;
	vector<FacialPose> showP;
	showI.resize(showN);
	showP.resize(showN);
	for (int i = 0; i < showN; i++) {
		int k = i+60;
		showI[i] = data.imgs[k];
		showP[i] = data.poses[k];
		cout << showP[i] << endl;
	}
	Mat aa=displaySamples(showI,showP, "../debug/debug.jpg",sqrt(showN));
	cout << "debug.jpg has written! " << endl;
	getchar();
	*/
}
void TDTOOLS::print_func(const char *a)
{
	return;
}
float TDTOOLS::sqr(float w)
{
	return w*w;
}
float TDTOOLS::dist(Point2f &a, Point2f &b)
{
	return sqrt(sqr(a.x-b.x)+sqr(a.y-b.y));
}


void TDTOOLS::divideDataSet(DataSet &origSet, DataSet &trainSet, DataSet &testSet, float ratio)
{
	testSet.n = origSet.n * ratio;
	trainSet.n = origSet.n-testSet.n;

	trainSet.imgs.resize(trainSet.n);
	trainSet.poses.resize(trainSet.n);
	trainSet.inits.resize(trainSet.n);
	for (int i = 0; i < trainSet.n; i++){
		trainSet.imgs[i] = origSet.imgs[i];
		trainSet.poses[i] = origSet.poses[i];
		trainSet.inits[i] = origSet.inits[i];
	}


	
	testSet.imgs.resize(testSet.n);
	testSet.poses.resize(testSet.n);
	testSet.inits.resize(testSet.n);
	for (int i = 0; i < testSet.n; i++){
		testSet.imgs[i] = origSet.imgs[i+trainSet.n];
		testSet.poses[i] = origSet.poses[i+trainSet.n];
		testSet.inits[i] = origSet.inits[i+trainSet.n];
	}

	trainSet.meanposeIni = origSet.meanposeIni;
	trainSet.meanposeTgt = origSet.meanposeTgt;
	testSet.meanposeIni = origSet.meanposeIni;
	testSet.meanposeTgt = origSet.meanposeTgt;
}

void TDTOOLS::catDataSet(DataSet &origSet, DataSet &addSet)
{
	origSet.imgs.resize(origSet.n+addSet.n);
	origSet.poses.resize(origSet.n+addSet.n);
	origSet.inits.resize(origSet.n+addSet.n);

	for (int i = 0; i < addSet.n; i++) {
		origSet.imgs[i+origSet.n] = addSet.imgs[i];
		origSet.poses[i+origSet.n] = addSet.poses[i];
		origSet.inits[i+origSet.n] = addSet.inits[i];
	}

	origSet.n += addSet.n;
}
void TDTOOLS::fluctuateDataImg(Mat &img, FacialPose &pose, FacialPose &meanpose, string ctrl)
{
	FacialPose tmp;
	fluctuateDataImg(img, pose, meanpose, tmp, ctrl);
}
void TDTOOLS::fluctuateDataImg(Mat &img, FacialPose &inipose, FacialPose &meanpose, FacialPose &pose, string ctrl)
{
	float w, d;
	float cx, cy;
	float hl, ag;
	Rect box = pose2box(inipose, meanpose); 

	hl = (box.width+box.height)/4.0;
	cx = box.x+hl;
	cy = box.y+hl;
	hl *= sqrt(2);
	ag = 0;

	for (int i = 0; i < ctrl.size(); i++) {
		w = randGX()/2.0;

		if ( w > 1) w = 1;
		if ( w < -1) w = -1;
		switch (ctrl.at(i)) {
			case 'h': {
						cx += 0.1*hl*w;
	//					cout << "h: " << 0.1*hl*w << endl;
						break;
					  }
			case 'v': {
						cy += 0.1*hl*w;
	//					cout << "v: " << 0.1*hl*w << endl;
						break;
					  }
			case 's': {
						hl += 0.1*hl*w;
	//					cout << "s: " << 0.1*hl*abs(w) << endl;
						break;
					  }
			case 'r': {
						ag += (1.0/6.0)*PI*w;
						break;
					  }
		}
	}


	FacialPose tmp;
	FacialPose tmp1;
	float dx, dy;
    
	dx = cos(ag+PI*0.25)*hl;
	dy = sin(ag+PI*0.25)*hl;

	tmp1.resize(4);
	tmp1[0].x = 0;
	tmp1[0].y = 0;
	tmp1[1].x = 0;
	tmp1[1].y = 255;
	tmp1[2].x = 255;
	tmp1[2].y = 0;
	tmp1[3].x = 255;
	tmp1[3].y = 255;

	tmp.resize(4);
	tmp[0].x = cx-dx;
	tmp[0].y = cy-dy;
	tmp[1].x = cx-dy;
	tmp[1].y = cy+dx;
	tmp[2].x = cx+dy;
	tmp[2].y = cy-dx;
	tmp[3].x = cx+dx;
	tmp[3].y = cy+dy;


	Mat trans;
	calcTransMat(tmp, tmp1, trans);
	cropPose(inipose,trans);
	cropPose(pose, trans);
	cropImg(img,256,trans);
/*
	cout << cx << " " << cy << endl;
	cout << hl  << endl;
	cout << ag/PI*180 << endl;

	imshow("haha", img);
	cvWaitKey();
	*/
}
void TDTOOLS::augmentDataSet(DataSet &data, int times = 0)
{
	if (times < 0) return;

	string ctrl("shv");

	for (int t = 1; t < times; t++) {
		for (int i = 0; i < data.n; i++) {
			Mat img;
			FacialPose inipose;
			FacialPose pose;

			data.imgs[i].copyTo(img);
			inipose = data.inits[i];
			pose = data.poses[i];

			fluctuateDataImg(img, inipose, data.meanposeIni, pose, ctrl);

			data.imgs.push_back(img);
			data.poses.push_back(pose);
			data.inits.push_back(inipose);
		}
	}
/*
	cout << data.inits[0] << endl;
	Mat mshow = data.imgs[0].clone();
	for (int j = 0; j < data.inits[0].size(); j++) {
		circle(mshow, Point(data.inits[0][j].x, data.inits[0][j].y), 2, Scalar(0,255,0));
	}
	imshow("haha",mshow);
	cvWaitKey(0);
*/

	for (int i = 0; i < data.n; i++) {
		if (times > 0)
			fluctuateDataImg(data.imgs[i], data.inits[i], data.meanposeIni, data.poses[i],ctrl);
		else {

			fluctuateDataImg(data.imgs[i], data.inits[i], data.meanposeIni, data.poses[i],"");

		}
	}

	if (times > 0) data.n *= times;

	cout << "aug over " << data.n << endl;
}

bool TDTOOLS::cmpSortItem(SortItem p, SortItem q)
{
	return (p.value < q.value);
}


int TDTOOLS::rand(int n)
{
	assert(n > 0);
	int ret = 0;
	while (n > 1) 
	{
		int m = n >> 1;
		if (::rand() & 1)
			ret += m;
		n -= m;
	}
	return ret;
}

static void displaySamples( Mat & dsp, const vector< Mat > & img, const vector< FacialPose > & poses, int m)
{
	if( img.size() == 0 ) return;

	int rows_ = img[0].rows;
	int cols_ = img[0].cols;
	Scalar color = Scalar(255, 0, 0);
	int m_ = m;
	int pad_ = 4;
	int row_step_ = ( rows_ + pad_ );
	int col_step_ = ( cols_ + pad_ );
	dsp = Mat::zeros( row_step_ * m_,  col_step_ * m_, img[0].type() );
	
	for(int win_id = 0; win_id < min( (int) img.size() , m_ * m_ ) ; win_id++)
	{
		int i = win_id % m_;
		int j = win_id / m_;
		
		int shift_x = col_step_ * i;
		int shift_y = row_step_ * j;
		
		img[ win_id ].copyTo( dsp( Range( shift_y, shift_y + rows_ ) , Range( shift_x, shift_x + cols_ ) ) );	
		
	}

	if(dsp.type() == CV_8U) cv::cvtColor( dsp, dsp, CV_GRAY2BGR );

	if ((int)poses.size() == 0) return;

	for(int win_id = 0; win_id < min( (int) img.size() , m_ * m_ ) ; win_id++)
	{
		int i = win_id % m_;
		int j = win_id / m_;
		
		int shift_x = col_step_ * i;
		int shift_y = row_step_ * j;

		for(int pid = 0; pid < (int) poses[win_id].size() ; pid++)
		{
			circle( dsp, poses[win_id][pid] + Point2f( (float)shift_x, (float)shift_y ), 2, color);
		}		
	}
}

Mat TDTOOLS::displaySamples(const vector< Mat > & img, const vector< FacialPose > & poses, string file_name, int m)
{
	Mat disp_mat;
	::displaySamples( disp_mat, img, poses, m);
	imwrite(file_name, disp_mat);

	return disp_mat;
}

void TDTOOLS::displaySamples(const vector< Mat > & img, string file_name, int m)
{
	Mat disp_mat;
	vector< FacialPose > poses;
	::displaySamples( disp_mat, img, poses, m);
	imwrite(file_name, disp_mat);
}

float TDTOOLS::randGX()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if (phase == 0) {
		do {
			double U1 = (double)::rand()/ RAND_MAX;
			double U2 = (double)::rand()/ RAND_MAX;

			V1 = 2*U1-1;
			V2 = 2*U2-1;
			S = V1*V1+V2*V2;
		}while (S > 1-1e-6 || S < 1e-6);

		X = V1*sqrt(-2*log(S)/S);
	}
	else {
		X = V2*sqrt(-2*log(S)/S);
	}

	return X;
}

Mat TDTOOLS::cp2rigid(Point2f* src, Point2f* dst, int M)
{
	Mat X(2 * M, 4, CV_32FC1), U(2 * M, 1, CV_32FC1);
	for (int i = 0; i < M; ++i) {
		X.at<float>(i, 0) = src[i].x;
		X.at<float>(i + M, 0) = src[i].y;
		X.at<float>(i, 1) = src[i].y;
		X.at<float>(i + M, 1) = -src[i].x;
		X.at<float>(i, 2) = X.at<float>(i + M, 3) = 1.;
		X.at<float>(i, 3) = X.at<float>(i + M, 2) = 0.;
		U.at<float>(i, 0) = dst[i].x;
		U.at<float>(i + M, 0) = dst[i].y;
	}
	Mat r = X.inv(DECOMP_SVD) * U, trans(2, 3, CV_32FC1);
	trans.at<float>(0, 0) = r.at<float>(0, 0);
	trans.at<float>(0, 1) = r.at<float>(1, 0);
	trans.at<float>(0, 2) = r.at<float>(2, 0);
	trans.at<float>(1, 0) = -r.at<float>(1, 0);
	trans.at<float>(1, 1) = r.at<float>(0, 0);
	trans.at<float>(1, 2) = r.at<float>(3, 0);
	return trans;
}

Rect TDTOOLS::pose2box(FacialPose &pose, FacialPose &meanpose, int win_size)
{
	Mat trans;
	FacialPose tmp;

	calcTransMat(meanpose, pose, trans);
	
	tmp.resize(4);
	tmp[0].x = 0;
	tmp[0].y = 0;
	tmp[1].x = 0;
	tmp[1].y = win_size;
	tmp[2].x = win_size;
	tmp[2].y = 0;
	tmp[3].x = win_size;
	tmp[3].y = win_size;

	cropPose(tmp, trans);

	Vec4f rst;
	rst[0] = rst[2] = tmp[0].x;
	rst[1] = rst[3] = tmp[0].y;

	for (int i = 1; i < 4; i++) {
		if (tmp[i].x < rst[0]) rst[0] = tmp[i].x;
		if (tmp[i].x > rst[2]) rst[2] = tmp[i].x;
		if (tmp[i].y < rst[1]) rst[1] = tmp[i].y;
		if (tmp[i].y > rst[3]) rst[3] = tmp[i].y;
	}

	Rect ret(rst[0],rst[1],(rst[2]-rst[0]), (rst[3]-rst[1]));

	return ret;
}

int TDTOOLS::findcfs(vector<int> &f, int i)
{
	if (f[i] != i) f[i] = findcfs(f,f[i]);
	return f[i];
}

void TDTOOLS::shuffle(vector<int> &a)
{

	int j;
	for (int i = 1; i < a.size(); i++) {

		j = (int)(::rand())%i;
		swap(a[i],a[j]);
	}
}
