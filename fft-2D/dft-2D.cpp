#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>

using namespace cv;
using namespace std;

#define N_TRAINING_SAMPLES 1000
#define N_ROWS 512//img.rows
#define N_COLS 512//img.cols
#define DIM 512
#define PI 3.1415926

const char* keys =
{
    "{@image|./lena.jpg|input image file}"
};

int dft_efficient(double **spectrum, double **spatial){

	double sum_real = 0;
	double sum_img = 0;
	double sum_real_2 = 0;
	double sum_img_2 = 0;
	double row_transforms_real[DIM];
	double row_transforms_img[DIM];

	// for each point in specturm
	for (int u=0;u<DIM;u++){
		for (int v=0;v<DIM;v++){

			// Inner transform: 
			// calculate the transform of each row and store them
			for (int row=0;row<DIM;row++){
				/*-----------NN 1 will learn this part-----------*/
				for (int n=0;n<DIM;n++){
					sum_real += (spatial[row][n] * 
							cos(2*PI*(1.0*n*v/DIM)) ) / DIM;
					sum_img -= (spatial[row][n] * 
							sin(2*PI*(1.0*n*v/DIM)) ) / DIM;
				}
				/*---------------------------------------------*/
				row_transforms_real[row] = sum_real;
				row_transforms_img[row] = sum_img;
				sum_real = 0;
				sum_img = 0;
			}

			/*-----------NN 2 will learn thi part-------------*/
			// Outer tansform:
			for (int m=0;m<DIM;m++){
				sum_real_2 += (row_transforms_real[m] * 
						cos(2*PI*(1.0*m*u/DIM)) ) / DIM;
				sum_img_2 -= (row_transforms_img[m] * 
					sin(2*PI*(1.0*m*u/DIM)) ) / DIM;
			/*---------------------------------------------*/
			}
			
			
			double amplitude = sqrt(sum_real_2*sum_real_2+sum_img_2*sum_img_2);
			spectrum[u][v] = amplitude;
			sum_real_2 = 0;
			sum_img_2 = 0;
			//cout << "(" << u << "," << v << "): " << amplitude << endl;
			if (!v){
				cout << "(" << u << "," << v << "): " << amplitude << endl;
			}
		}
	}
	spectrum[0][0] = 2;

}
/*
int NN_dft(double **spectrum, double **spatial){
	// for each point in specturm
	for (int u=0;u<DIM;u++){
		for (int v=0;v<DIM;v++){
			
			// Inner transform:
			for (int row;row<512;row++){
				sum = NN_row_transform(row,v);
				row_transforms_real[row] = sum.real;
				row_transforms_img[row] = sum.img;	
			}
			// Outer tansform:
			sum_2 = NN_col_transform(m,u);

			// Write result
			amplitude = sqrt(sum_2.real*sum_2.real + sum_2.img*sum_2.img);
			spectrum[u][v] = amplitude;
		}
	}

}
*/
int main(int argc, const char ** argv)
{
	// dynamically allocate input and output matrices to FFT
	double **spatial;
	spatial = new double*[512]; 
	for (int i = 0; i < 512; ++i) {
	  spatial[i] = new double[512];
	}

	double **spectrum;
	spectrum = new double*[512];
	for (int i = 0; i < 512; ++i) {
	  spectrum[i] = new double[512];
	}

    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);

    Mat img = imread(filename, IMREAD_GRAYSCALE);

	//imshow("spatial magnitude", img);
    //waitKey();

	for (int i = 0; i < N_ROWS; i++){
	   	for (int j = 0; j < N_COLS; j++){
	        spatial[i][j] = img.at<uchar>(i,j);
		}
	}

	/*-----------------PREPROCESSING FOR OPENCV DFT----------------*/

	if( img.empty() ){
	    printf("ERROR: Cannot read image file: %s\n", filename.c_str());
	    return -1;
	}
	int M = getOptimalDFTSize( img.rows );
	int N = getOptimalDFTSize( img.cols );
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);

	//imshow("spatial magnitude 2", complexImg);
    //waitKey();


	/*------------------------DFT--------------------------*/

	// OpenCV Implementation
	//dft(complexImg, complexImg);

	//Implementation For training (new)
	dft_efficient(spectrum,spatial);

	// NN Implementation
	//NN_dft();

	
	Mat img_calc = imread(filename, IMREAD_GRAYSCALE);
	for (int i = 0; i < N_ROWS; i++){
	   	for (int j = 0; j < N_COLS; j++){
	        img_calc.at<uchar>(i,j) = (int)spectrum[i][j];
		}
	}

	ofstream myfile_2;
	std::string myfile_2_name = "raw_Lena_dft.txt"; 
	myfile_2.open (myfile_2_name);
	for(int i=0;i<N_ROWS;i++) {
		for (int j=0;j<N_COLS;j++){ 
			//if(j>0){myfile_2<<",";}
			//else if(i>0){myfile_2<<",";}			
			myfile_2<<img_calc.at<uchar>(i,j)<<endl;
		}
	}

	myfile_2<<endl;
	

	log(img_calc, img_calc);
	normalize(img_calc, img_calc, 0, 255, NORM_MINMAX);
	for(int i=0;i<512;i++) {
	  for (int j=0;j<512;j++){ 
		cout<<(int)img_calc.at<uchar>(i,j)<<",";	
	  }
	}
	imshow("spatial magnitude: img_calc", img_calc);
    waitKey();
	




	/*-----------------POST PROCESSING---------------------*/

	/*
	for(int j=0;j<mag.rows;j++) {
	  for (int i=0;i<mag.cols;i++){ 
	  	complexImg.at<uchar>(i,j) = result[i][j];
		//cout<<pixelValue<<",";	
	  }
	}
	*/	
	
	// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat mag = planes[0];
	mag += Scalar::all(1);
	log(mag, mag);
	
	//imshow("spectrum magnitude", mag);
    //waitKey();

	// crop the spectrum, if it has an odd number of rows or columns
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols/2;
	int cy = mag.rows/2;

	//imshow("spectrum magnitude", mag);
    //waitKey();

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//imshow("spectrum magnitude", mag);
    //waitKey();

	/*
	for(int i=0;i<mag.rows;i++) {
	  for (int j=0;j<mag.cols;j++){ 
		cout<<(int)mag.at<uchar>(i,j)<<",";	
	  }
	}
	*/

	normalize(mag, mag, 0, 1, NORM_MINMAX);
/*
	for(int i=0;i<mag.rows;i++) {
	  for (int j=0;j<mag.cols;j++){ 
		cout<<(int)mag.at<uchar>(i,j)<<",";	
	  }
	}
*/
	//imshow("spectrum magnitude", mag);
    //waitKey();

    return 0;
}
