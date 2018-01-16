#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

#include "./fft-2D.cpp"

using namespace cv;
using namespace std;

#define N_TRAINING_SAMPLES 1000
#define N_ROWS 512//img.rows
#define N_COLS 512//img.cols
#define DIM 512
#define PI 3.141592653589793238460
#define NUM_NEURONS 2048
#define SIG_N 512

int NN_dft_naive(double **spectrum, double **spatial);

int read_weights(double **weights_0_1, double **weights_1_2, 
					//double **layer_1_bias, double **output_bias,
						double **weights_0_1_imag, double **weights_1_2_imag);
							//double **layer_1_bias_imag, double **output_bias_imag);

const char* keys =
{
    //"{@image|./lena.jpg|input image file}"
	"{@image| /home/junjian/Desktop/1.png|input image file}"
};

int dft_for_training(double **spectrum, double **spatial){

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
			// 1-D FFT of rows and store results
			for (int row=0;row<DIM;row++){
				/*-----------NN will learn this part-----------*/
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

			/*-----------NN will learn this part-------------*/
			// Outer tansform (1D-FFT of columns):
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
			if (!v){
				cout << "processing: (print once for every 512)";
				cout << "(" << u << "," << v << "): " << amplitude << endl;
			}
		}
	}

}

int NN_dft(double **spectrum, double **spatial){

	double sum_real_2;
	double sum_imag_2;

	double row_transforms_real[DIM][DIM];
	double row_transforms_imag[DIM][DIM];
//
	// NN matrices (real)
	double signal[SIG_N][1];
    double layer_1[NUM_NEURONS][1];
	double output[SIG_N][1];

	double layer_1_bias[NUM_NEURONS][1];
	double output_bias[SIG_N][1];
	double **weights_0_1;
	weights_0_1 = new double*[NUM_NEURONS]; 
	for (int i = 0; i < NUM_NEURONS; ++i) {
	  weights_0_1[i] = new double[SIG_N];
	}
	double **weights_1_2;
	weights_1_2 = new double*[SIG_N]; 
	for (int i = 0; i < SIG_N; ++i) {
	  weights_1_2[i] = new double[NUM_NEURONS];
	}

	// NN matrices (imag)
	double signal_imag[SIG_N][1];
    double layer_1_imag[NUM_NEURONS][1];
	double output_imag[SIG_N][1];

	double layer_1_bias_imag[NUM_NEURONS][1];
	double output_bias_imag[SIG_N][1];
	double **weights_0_1_imag;
	weights_0_1_imag = new double*[NUM_NEURONS]; 
	for (int i = 0; i < NUM_NEURONS; ++i) {
	  weights_0_1_imag[i] = new double[SIG_N];
	}
	double **weights_1_2_imag;
	weights_1_2_imag = new double*[SIG_N]; 
	for (int i = 0; i < SIG_N; ++i) {
	  weights_1_2_imag[i] = new double[NUM_NEURONS];
	}


	read_weights(weights_0_1, weights_1_2, 
						//layer_1_bias, output_bias,
							weights_0_1_imag, weights_1_2_imag 
								//layer_1_bias_imag, output_bias_imag
	);

	int i=0,j=0,k=0;
	char *ptr;
	FILE *fp;
	char str[20];
	// layer_1_bias
	char* filename_2 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights/input_layer_1_bias.txt";
	fp = fopen(filename_2, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename_2);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		layer_1_bias[i][0] = strtod(str, &ptr);
	}
	fclose(fp);

	// output_bias
	char* filename_3 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights/layer_1_2_bias.txt";
	fp = fopen(filename_3, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename_3);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		output_bias[i][0] = strtod(str, &ptr);
	}
	fclose(fp);

	// layer_1_bias (imag)
	char* filename_7 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights_imag/input_layer_1_bias_imag.txt";
	fp = fopen(filename_7, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename_7);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		layer_1_bias_imag[i][0] = strtod(str, &ptr);
	}
	fclose(fp);

	// output_bias (imag)
	char* filename_8 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights_imag/layer_1_2_bias_imag.txt";
	fp = fopen(filename_8, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename_8);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		output_bias_imag[i][0] = strtod(str, &ptr);
	}
	fclose(fp);

	// Inner transform: 
	// calculate the transform of each row and store them
	for (int row=0;row<DIM;row++){
		//---------------------------------------------------/
		//------------------Feed Forwarding-----------------/
		for (int n=0; n < SIG_N; n++) {
			signal[n][0] = spatial[row][n];
		}							
		// dgemm(layer_1,signal,weights_0_1);
		for(int i=0 ; i<NUM_NEURONS ; i++)
		{
			layer_1[i][0] = 0;
			for(int k=0 ; k<SIG_N ; k++ ){
				layer_1[i][0] += signal[k][0] * weights_0_1[i][k] ;
			}
		}
		// bias
			for(int i=0 ; i<NUM_NEURONS ; i++){
				layer_1[i][0] += layer_1_bias[i][0];
		}
		// dgemm(output,layer_1,weights_1_2);  
		for(int i=0 ; i<SIG_N ; i++)
		{
			output[i][0] = 0;
			for(int k=0 ; k<NUM_NEURONS; k++ ){
				output[i][0] += layer_1[k][0] * weights_1_2[i][k] ;
			}
			//printf("\n");
		}
		// bias
		for(int i=0 ; i<SIG_N ; i++){
			output[i][0] += output_bias[i][0];
		}
		//____________---(imag)---________________						
		// dgemm(layer_1,signal,weights_0_1);
		for(int i=0 ; i<NUM_NEURONS ; i++)
		{
			layer_1_imag[i][0] = 0;
			for(int k=0 ; k<SIG_N ; k++ ){
				layer_1_imag[i][0] += signal[k][0] * weights_0_1_imag[i][k] ;
			}
		}
		// bias
			for(int i=0 ; i<NUM_NEURONS ; i++){
				layer_1_imag[i][0] += layer_1_bias_imag[i][0];
		}
		// dgemm(output,layer_1,weights_1_2);  
		for(int i=0 ; i<SIG_N ; i++)
		{
			output_imag[i][0] = 0;
			for(int k=0 ; k<NUM_NEURONS; k++ ){
				output_imag[i][0] += layer_1_imag[k][0] * weights_1_2_imag[i][k] ;
			}
			//printf("\n");
		}
		// bias
		for(int i=0 ; i<SIG_N ; i++){
			output_imag[i][0] += output_bias_imag[i][0];
		}
		//________________________________________
		
		
		//-----------------------------------------------------/
		//----------------------------------------------------/
		for (int v=0;v<DIM;v++){
			row_transforms_real[row][v] = output[v][0];
			row_transforms_imag[row][v] = output_imag[v][0];
		}
	}

	// for each point in specturm
	for (int u=0;u<DIM;u++){
		for (int v=0;v<DIM;v++){
			//-----------NN 2 will learn this part-------------/
			// Outer tansform:
			for (int m=0;m<DIM;m++){
				sum_real_2 += (row_transforms_real[m][v] * 
						cos(2*PI*(1.0*m*u/DIM)) ) / DIM;
				sum_imag_2 -= (row_transforms_imag[m][v] * 
						sin(2*PI*(1.0*m*u/DIM)) ) / DIM;
			//-------------------------------------------------/
			}
			double amplitude = sqrt(sum_real_2*sum_real_2+sum_imag_2*sum_imag_2);
			spectrum[u][v] = amplitude;
			sum_real_2 = 0;
			sum_imag_2 = 0;
			if (!v){
				cout << "processing: (print once for every 512)";
				cout << "(" << u << "," << v << "): " << amplitude << endl;
			}
		}
	}


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
	spatial = new double*[DIM]; 
	for (int i = 0; i < DIM; ++i) {
	  spatial[i] = new double[DIM];
	}

	double **spectrum;
	spectrum = new double*[DIM];
	for (int i = 0; i < DIM; ++i) {
	  spectrum[i] = new double[DIM];
	}

    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);

    Mat img = imread(filename, IMREAD_GRAYSCALE);

	for (int i = 0; i < N_ROWS; i++){
	   	for (int j = 0; j < N_COLS; j++){
	        spatial[i][j] = img.at<uchar>(i,j);
		}
	}

	/*------------------------DFT--------------------------*/

	// OpenCV Implementation
	//dft(complexImg, complexImg);

	// another implementation
	//FFT2D(com,512,512,1);
	
	//Implementation For training (new)
	std::cout << "calling...." << std::endl;
	//dft_for_training(spectrum,spatial);
    std::cout << "done..." << std::endl;

	// NN Implementation
	std::cout << "calling...." << std::endl;
	//NN_dft(spectrum,spatial);
	NN_dft_naive(spectrum,spatial);
	std::cout << "done..." << std::endl;

	// change to log scale
	for (int i = 0; i < 512; i++) {
	    for (int j = 0; j < 512; j++) {
	        spectrum[i][j] = log(spectrum[i][j]+0.5);
	    }
	}


    Mat spect (512, 512, CV_64F, spectrum);

	int cx = 512/2;
	int cy = 512/2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	Mat tmp;
	Mat q0(spect, Rect(0, 0, cx, cy));
	Mat q1(spect, Rect(cx, 0, cx, cy));
	Mat q2(spect, Rect(0, cy, cx, cy));
	Mat q3(spect, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	
	// normalize to 0-1
    normalize(spect, spect, 0 , 1 , NORM_MINMAX);

	// show image        
    imshow("fsadfas", spect);
    waitKey();

    return 0;
}

int read_weights(double **weights_0_1, double **weights_1_2, 
					//double **layer_1_bias, double **output_bias,
						double **weights_0_1_imag, double **weights_1_2_imag
							//double **layer_1_bias_imag, double **output_bias_imag
){

	int i=0,j=0,k=0;
	// weights_0_1
	char *ptr;
	FILE *fp;
	char str[20];
	char* filename = "/home/junjian/Desktop/weights/input_layer_1_kernal.txt";
	fp = fopen(filename, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename);
		return 1;
	}
	i=0;j=0;
	while (fgets(str, 20, fp) != NULL){
		if (j==SIG_N){
			j = 0;
			i++;
		}
		weights_0_1[i][j] = strtod(str, &ptr);
		j++;
	}
	fclose(fp);

	// weights_1_2
	char* filename_4 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights/layer_1_2_kernal.txt";
	fp = fopen(filename_4, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename);
		return 1;
	}
	i=0;j=0;
	while (fgets(str, 20, fp) != NULL){
		if (j==NUM_NEURONS){
			j = 0;
			i++;
		}
		weights_1_2[i][j] = strtod(str, &ptr);
		j++;
	}
	fclose(fp);

	// weights_0_1 (imag)
	char* filename_5 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights_imag/input_layer_1_kernal_imag.txt";
	fp = fopen(filename_5, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename);
		return 1;
	}
	i=0;j=0;
	while (fgets(str, 20, fp) != NULL){
		if (j==NUM_NEURONS){
			j = 0;
			i++;
		}
		weights_0_1_imag[i][j] = strtod(str, &ptr);
		j++;
	}
	fclose(fp);

	// weights_1_2 (imag)
	char* filename_6 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights_imag/layer_1_2_kernal_imag.txt";
	fp = fopen(filename_6, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename);
		return 1;
	}
	i=0;j=0;
	while (fgets(str, 20, fp) != NULL){
		if (j==NUM_NEURONS){
			j = 0;
			i++;
		}
		weights_1_2_imag[i][j] = strtod(str, &ptr);
		j++;
	}
	fclose(fp);

	return 0;
}

int NN_dft_naive(double **spectrum, double **spatial){

	double sum_real_2;
	double sum_imag_2;

	double row_transforms_real[DIM];
	double row_transforms_imag[DIM];

	// NN matrices (real)
	double signal[SIG_N][1];
    double layer_1[NUM_NEURONS][1];
	double output[SIG_N][1];

	double layer_1_bias[NUM_NEURONS][1];
	double output_bias[SIG_N][1];
	double **weights_0_1;
	weights_0_1 = new double*[NUM_NEURONS]; 
	for (int i = 0; i < NUM_NEURONS; ++i) {
	  weights_0_1[i] = new double[SIG_N];
	}
	double **weights_1_2;
	weights_1_2 = new double*[SIG_N]; 
	for (int i = 0; i < SIG_N; ++i) {
	  weights_1_2[i] = new double[NUM_NEURONS];
	}

	// NN matrices (imag)
	double signal_imag[SIG_N][1];
    double layer_1_imag[NUM_NEURONS][1];
	double output_imag[SIG_N][1];

	double layer_1_bias_imag[NUM_NEURONS][1];
	double output_bias_imag[SIG_N][1];
	double **weights_0_1_imag;
	weights_0_1_imag = new double*[NUM_NEURONS]; 
	for (int i = 0; i < NUM_NEURONS; ++i) {
	  weights_0_1_imag[i] = new double[SIG_N];
	}
	double **weights_1_2_imag;
	weights_1_2_imag = new double*[SIG_N]; 
	for (int i = 0; i < SIG_N; ++i) {
	  weights_1_2_imag[i] = new double[NUM_NEURONS];
	}


	read_weights(weights_0_1, weights_1_2, 
						//layer_1_bias, output_bias,
							weights_0_1_imag, weights_1_2_imag 
								//layer_1_bias_imag, output_bias_imag
	);

	int i=0,j=0,k=0;
	char *ptr;
	FILE *fp;
	char str[20];
	// layer_1_bias
	char* filename_2 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights/input_layer_1_bias.txt";
	fp = fopen(filename_2, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename_2);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		layer_1_bias[i][0] = strtod(str, &ptr);
	}
	fclose(fp);

	// output_bias
	char* filename_3 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights/layer_1_2_bias.txt";
	fp = fopen(filename_3, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename_3);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		output_bias[i][0] = strtod(str, &ptr);
	}
	fclose(fp);

	// layer_1_bias (imag)
	char* filename_7 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights_imag/input_layer_1_bias_imag.txt";
	fp = fopen(filename_7, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename_7);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		layer_1_bias_imag[i][0] = strtod(str, &ptr);
	}
	fclose(fp);

	// output_bias (imag)
	char* filename_8 = "/home/junjian/Desktop/RedEyeRemover/496-modified/fft-2D/training/weights_imag/layer_1_2_bias_imag.txt";
	fp = fopen(filename_8, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename_8);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		output_bias_imag[i][0] = strtod(str, &ptr);
	}
	fclose(fp);




	// for each point in specturm
	for (int u=0;u<DIM;u++){
		for (int v=0;v<DIM;v++){

			// Inner transform: 
			// calculate the transform of each row and store them
			for (int row=0;row<DIM;row++){
				//---------------------------------------------------/
				//------------------Feed Forwarding-----------------/
				for (int n=0; n < SIG_N; n++) {
					signal[n][0] = spatial[row][n];
				}							
				// dgemm(layer_1,signal,weights_0_1);
				for(int i=0 ; i<NUM_NEURONS ; i++)
				{
					layer_1[i][0] = 0;
					for(int k=0 ; k<SIG_N ; k++ ){
						layer_1[i][0] += signal[k][0] * weights_0_1[i][k] ;
					}
				}
				// bias
					for(int i=0 ; i<NUM_NEURONS ; i++){
						layer_1[i][0] += layer_1_bias[i][0];
				}
				// dgemm(output,layer_1,weights_1_2);  
				for(int i=0 ; i<SIG_N ; i++)
				{
					output[i][0] = 0;
					for(int k=0 ; k<NUM_NEURONS; k++ ){
						output[i][0] += layer_1[k][0] * weights_1_2[i][k] ;
					}
					//printf("\n");
				}
				// bias
				for(int i=0 ; i<SIG_N ; i++){
					output[i][0] += output_bias[i][0];
				}
				//____________---(imag)---________________						
				// dgemm(layer_1,signal,weights_0_1);
				for(int i=0 ; i<NUM_NEURONS ; i++)
				{
					layer_1_imag[i][0] = 0;
					for(int k=0 ; k<SIG_N ; k++ ){
						layer_1_imag[i][0] += signal[k][0] * weights_0_1_imag[i][k] ;
					}
				}
				// bias
					for(int i=0 ; i<NUM_NEURONS ; i++){
						layer_1_imag[i][0] += layer_1_bias_imag[i][0];
				}
				// dgemm(output,layer_1,weights_1_2);  
				for(int i=0 ; i<SIG_N ; i++)
				{
					output_imag[i][0] = 0;
					for(int k=0 ; k<NUM_NEURONS; k++ ){
						output_imag[i][0] += layer_1_imag[k][0] * weights_1_2_imag[i][k] ;
					}
					//printf("\n");
				}
				// bias
				for(int i=0 ; i<SIG_N ; i++){
					output_imag[i][0] += output_bias_imag[i][0];
				}
				//________________________________________
		
		
				//-----------------------------------------------------/
				//----------------------------------------------------/
				//for (int v=0;v<DIM;v++){
					row_transforms_real[row] = output[v][0];
					row_transforms_imag[row] = output_imag[v][0];
				//}
			}
			//-----------NN 2 will learn this part-------------/
			// Outer tansform:
			for (int m=0;m<DIM;m++){
				sum_real_2 += (row_transforms_real[m] * 
						cos(2*PI*(1.0*m*u/DIM)) ) / DIM;
				sum_imag_2 -= (row_transforms_imag[m] * 
						sin(2*PI*(1.0*m*u/DIM)) ) / DIM;
			//-------------------------------------------------/
			}
			double amplitude = sqrt(sum_real_2*sum_real_2+sum_imag_2*sum_imag_2);
			spectrum[u][v] = amplitude;
			sum_real_2 = 0;
			sum_imag_2 = 0;
			if (!v){
				cout << "processing: (print once for every 512)";
				cout << "(" << u << "," << v << "): " << amplitude << endl;
			}
		}
	}


}
