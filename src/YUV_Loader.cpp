#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#define flag ACCESS_READ

using namespace std;
using namespace cv;

FILE * pFile = fopen ("FTV_LOGFILE.txt","w");

void serial(){
    double t = (double)getTickCount();
    //---------------------------------------------------------------------------------------------------

	//SERIAL:
	Mat img, gray;
	img = imread("image.jpg", 1);

	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray,Size(7, 7), 1.5);
	Canny(gray, gray, 0, 50);

	//---------------------------------------------------------------------------------------------------
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "SERIAL TIME(sec): " << t << endl;
}

void parallel(){
    double t = (double)getTickCount();
    //---------------------------------------------------------------------------------------------------

	//PARALLEL:
	UMat img, gray;
	imread("image.jpg", 1).copyTo(img);

	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray,Size(7, 7), 1.5);
	Canny(gray, gray, 0, 50);

	//---------------------------------------------------------------------------------------------------
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "PARALLEL TIME(sec): " << t << endl;
}

void compM(Mat &A, Mat &B){

    Mat_<double>::iterator it = A.begin<double>();
    Mat_<double>::iterator it2 = B.begin<double>();
    Mat_<double>::iterator itend = A.end<double>();

    for (; it != itend; ++it, it2++) {

    	if( (*it) != (*it2) ) cout << "NOT SAME: " << (*it) << " , " << (*it2) << endl;
    }
}

//________________________________________________________________________________________________________________________
//      LOAD FRAME FROM YUV:
//________________________________________________________________________________________________________________________

IplImage *
cvLoadImageYUV(FILE *fin, int w, int h)
{
    assert(fin);

    IplImage *py      = cvCreateImage(cvSize(w,    h), IPL_DEPTH_8U, 1);
    IplImage *pu      = cvCreateImage(cvSize(w/2,h/2), IPL_DEPTH_8U, 1);
    IplImage *pv      = cvCreateImage(cvSize(w/2,h/2), IPL_DEPTH_8U, 1);
    IplImage *pu_big  = cvCreateImage(cvSize(w,    h), IPL_DEPTH_8U, 1);
    IplImage *pv_big  = cvCreateImage(cvSize(w,    h), IPL_DEPTH_8U, 1);
    IplImage *image   = cvCreateImage(cvSize(w,    h), IPL_DEPTH_8U, 3);
    IplImage *result  = NULL;

    assert(py);
    assert(pu);
    assert(pv);
    assert(pu_big);
    assert(pv_big);
    assert(image);

    for (int i = 0; i < w*h; ++i)
    {
        int j = fgetc(fin);
        if (j < 0)
            goto cleanup;
        py->imageData[i] = (unsigned char) j;
    }

    for (int i = 0; i < w*h/4; ++i)
    {
        int j = fgetc(fin);
        if (j < 0)
            goto cleanup;
        pu->imageData[i] = (unsigned char) j;
    }

    for (int i = 0; i < w*h/4; ++i)
    {
        int j = fgetc(fin);
        if (j < 0)
            goto cleanup;
        pv->imageData[i] = (unsigned char) j;
    }

    cvResize(pu, pu_big, CV_INTER_NN);
    cvResize(pv, pv_big, CV_INTER_NN);
    cvMerge(py, pu_big, pv_big, NULL, image);

    result = image;

cleanup:
    cvReleaseImage(&pu);
    cvReleaseImage(&pv);

    cvReleaseImage(&py);
    cvReleaseImage(&pu_big);
    cvReleaseImage(&pv_big);

    if (result == NULL)
        cvReleaseImage(&image);

    return result;
}

void getFrame(const char* filePath, int RGB, Mat &iframe){

    // ex. FilePath: "/Users/Mark/balloons1.yuv",
    FILE *openFile = fopen (filePath, "rb+");

    //if( openFile != NULL) printf("READ FILE\n");

    IplImage * ipl = cvLoadImageYUV(openFile, 1024, 768);

    //CONVERT FROM IplImage TO Mat:
    iframe = cvarrToMat(ipl);

    // CONVERT FROM YUV (YCrCb) TO RGB:
    cvtColor(iframe, iframe, CV_YCrCb2RGB);

    if(RGB == 0){
        // CONVERT FROM YUV (YCrCb) TO GRAY:
        cvtColor(iframe, iframe, CV_RGB2GRAY);
    }

}

//________________________________________________________________________________________________________________________
//  BAD BOUNDARY DETECTION:
//________________________________________________________________________________________________________________________
void createFGMask(Mat &depth, Mat &binary){
	threshold(depth, binary, 120, 255, 0);
}

void createDilationMask(Mat &fgMask, Mat &dilationMask){
    // Dilate the image
    dilate(fgMask,dilationMask,Mat());
}

void extractOutline(Mat &depthMap, Mat &bdMask){
    Mat fgMask, dilatedMask;
    createFGMask(depthMap, fgMask);
    createDilationMask(fgMask, dilatedMask);

    bitwise_xor(fgMask, dilatedMask, bdMask);
}


//________________________________________________________________________________________________________________________
//  FORWARD WARPING:
//________________________________________________________________________________________________________________________

//Reference Camera Parameters:
Mat K = (Mat_<double>(3,3) << 2241.25607, 0.0, 701.5, 0.0, 2241.25607, 514.5, 0.0, 0.0, 1.0 );
Mat R = (Mat_<double>(3,3) << 1.0, 0 , 0 , 0 , 1.0, 0 , 0 , 0 , 1.0 );

Mat t_l = (Mat_<double>(3,1) << 20, 0, 0);
Mat t_r = (Mat_<double>(3,1) << 25, 0, 0);

Mat backProjection(double x, double y, double z, int cam){
    Mat t;
    if( cam == 0 ) t = t_l; // 0 = Left_Cam;
    else t = t_r; // 1 = Right_Cam;

    Mat refCoord = (Mat_<double>(3,1) << x, y, 1);
    Mat coord3D = (R.inv() )*( (z*K.inv() )*refCoord - t );

    return coord3D;
}

Mat reProjection(Mat &coord3D){
    Mat c =  ((-R).inv()) * ( ((1 - 0.4)*(-R)*(t_l) + (0.4)*(-R)*(t_r) ) );

    Mat virtualCoord = (  K * (( R*coord3D) + c) );

    return virtualCoord;
}

void calcRealDepth(Mat &depth, Mat& realDepth){

	float MAX_DEPTH = 11206.28035f;
	float MIN_DEPTH = 448.251214f;
	float DIFF_DEPTH = -(MAX_DEPTH - MIN_DEPTH)/255;
	float PLUS_DEPTH = MIN_DEPTH - (DIFF_DEPTH* 255);

	realDepth = depth.clone();
	realDepth.convertTo(realDepth, CV_64FC1);

	realDepth *= DIFF_DEPTH;
	realDepth += PLUS_DEPTH;
}


int roundCoord(double newX){
    if( (newX + 0.5) >= (int(newX) + 1) )
        newX = int(newX)+1;
    else
        newX = int(newX);

    return newX;
}

int calcIntDepth(double depth){
    return roundCoord( 265.62 - ( 0.0237* depth));
}

void accessVec3B(Mat &img){
	unsigned char *input = (unsigned char*)(img.data);
	Mat_<Vec3b>::iterator it = img.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = img.end<Vec3b>();

	for(int j = 0; j < img.rows; j++){
		for(int i = 0; i < img.cols; i++){
			unsigned char b = input[ img.cols * j+i];
			unsigned char g = input[ img.cols * j+i + 1];
			unsigned char r = input[ img.cols * j+i + 2];

			fprintf (pFile, "PTR- R: %d, G: %d, B: %d .\n",r, g, b );

			uchar rb = (*it)[0];
			uchar rg = (*it)[1];
			uchar rr = (*it)[2];
			fprintf (pFile, "ITR- R: %d, G: %d, B: %d .\n\n",rr, rg, rb );
			 ++it;

		}
	}
}

void accessIt( Mat &img){
	Mat_<Vec3b>::iterator it = img.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = img.end<Vec3b>();

	fprintf (pFile, "\n\n\n_________________________________________________________________________\n\n" );
	for (; it != itend; ++it) {
		uchar r = (*it)[0];
		uchar g = (*it)[1];
		uchar b = (*it)[2];
		fprintf (pFile, "ITR- R: %d, G: %d, B: %d .\n",r, g, b );
	}
}


//----------------------------------------------------------------------
int main(int argc, char** argv){

    //THREE CHANNEL MAT:
    Mat image1; getFrame("YUV/balloons4.yuv", 1, image1);

    Mat image3; getFrame("YUV/depth_balloons_4.yuv", 0, image3);

    UMat image4 = image1.getUMat( flag );
    UMat image5 = image3.getUMat( flag );

    Mat Y, X;
    for(int i = 0; i <40 ; i++){

    }
    accessVec3B(image1);
    accessIt(image1);


    double t = (double)getTickCount();
    //---------------------------------------------------------------------------------------------------

     for(int i = 0; i <40 ; i++){
     }

    //---------------------------------------------------------------------------------------------------
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t/40 << endl;







    fclose (pFile);
    return 0;
}







