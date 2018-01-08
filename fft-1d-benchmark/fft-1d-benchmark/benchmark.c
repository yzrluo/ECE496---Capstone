#include "fft.h"
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#define N 32
#define TIMES 100000
#define NUM_NEURONS 64

void timeSubtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;
}

double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int main(void) {
    struct timeval tvBegin, tvEnd, tvDiff;

    complex * input = (complex*) malloc(sizeof(struct complex_t) * N);
    complex * result;
    
	// naive
    gettimeofday(&tvBegin, NULL);
    for (int i=0; i < TIMES; i++) {
		// Init inputs
		for (int i=0; i < N; i++) {
			input[i].re = randfrom(0,32);
		    input[i].im = 0.0;
		}
        result = DFT_naive(input, N);
    }
    gettimeofday(&tvEnd, NULL);
    timeSubtract(&tvDiff, &tvEnd, &tvBegin);
    printf("%d x naive: \t %ld.%ld\n", TIMES, tvDiff.tv_sec, tvDiff.tv_usec);

	// Cooley-Tukey 
    gettimeofday(&tvBegin, NULL);
    for (int i=0; i < TIMES; i++) {
		// Init inputs
		for (int i=0; i < N; i++) {
			input[i].re = randfrom(0,32);
		    input[i].im = 0.0;
		}
        result = FFT_CooleyTukey(input, N, 6, 5);
    }
    gettimeofday(&tvEnd, NULL);
    timeSubtract(&tvDiff, &tvEnd, &tvBegin);
    printf("%d x Cooley-Tukey-fft: \t %ld.%ld\n", TIMES, tvDiff.tv_sec, tvDiff.tv_usec);

    // GoodThomas
    gettimeofday(&tvBegin, NULL);
    for (int i=0; i < TIMES; i++) {
		// Init inputs
		for (int i=0; i < N; i++) {
			input[i].re = randfrom(0,32);
		    input[i].im = 0.0;
		}
        result = FFT_GoodThomas(input, N, 6, 5);
    }
    gettimeofday(&tvEnd, NULL);
    timeSubtract(&tvDiff, &tvEnd, &tvBegin);
    printf("%d x Good-Thomas-fft: \t %ld.%ld\n", TIMES, tvDiff.tv_sec, tvDiff.tv_usec);

/*
    // print real
	printf("\n\n");
    printf("real: \n");
    for (int i=0; i < N; i++) {
        printf("%f,",result[i].re);
      
    }     
    printf("\n\n");
	
    // print imaginary
	printf("imaginary: \n");
    for (int i=0; i < N; i++) {
        printf("%f,",result[i].im);
      
    }  
	printf("\n\n");  
*/

/*--------------INITIALIZE MATRICES-------------------*/
	double signal[N][1];
    double weights_0_1[NUM_NEURONS][N];
	double layer_1_bias[NUM_NEURONS][1];
    double layer_1[NUM_NEURONS][1];
    double weights_1_2[N][NUM_NEURONS];
	double output_bias[N][1];
    double output[N][1];
	int i,j,k;

	// signal
	for (int i=0; i < N; i++) {
		signal[i][0] = input[i].re;
	}

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
		if (j==N){
			j = 0;
			i++;
		}
		weights_0_1[i][j] = strtod(str, &ptr);
		j++;
	}
	fclose(fp);

	// weights_1_2
	char* filename_4 = "/home/junjian/Desktop/weights/layer_1_2_kernal.txt";
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

	// layer_1_bias
	char* filename_2 = "/home/junjian/Desktop/weights/input_layer_1_bias.txt";
	fp = fopen(filename_2, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		layer_1_bias[i][0] = strtod(str, &ptr);
	}
	fclose(fp);

	// output_bias
	char* filename_3 = "/home/junjian/Desktop/weights/layer_1_2_bias.txt";
	fp = fopen(filename_3, "r");
	if (fp == NULL){
		printf("Could not open file %s",filename);
		return 1;
	}
	i=0;
	while (fgets(str, 20, fp) != NULL){
		output_bias[i][0] = strtod(str, &ptr);
	}
	fclose(fp);
	
/*-----------------------------------------------------*/

/*------------------FORWARD FEEDING--------------------*/
	gettimeofday(&tvBegin, NULL);
	for (int t=0; t < TIMES; t++) {	
		// Init inputs
		for (int i=0; i < N; i++) {
			input[i].re = randfrom(0,32);
			input[i].im = 0.0;
		}

		// dgemm(layer_1,signal,weights_0_1);
		for( i=0 ; i<NUM_NEURONS ; i++)
		{
			layer_1[i][0] = 0;
			for( k=0 ; k<N ; k++ ){
				layer_1[i][0] += signal[k][0] * weights_0_1[i][k] ;
			}
		}
		
		// + bias
		for( i=0 ; i<NUM_NEURONS ; i++){
			layer_1[i][0] += layer_1_bias[i][0];
		}
		

		//printf("\n");
		// dgemm(output,layer_1,weights_1_2);  
		for( i=0 ; i<N ; i++)
		{
			output[i][0] = 0;
			for( k=0 ; k<NUM_NEURONS; k++ ){
				output[i][0] += layer_1[k][0] * weights_1_2[i][k] ;
			}
			//printf("\n");
		}
		
		// + bias
		for( i=0 ; i<N ; i++){
			output[i][0] += output_bias[i][0];
		}
		
	}

	// time
	gettimeofday(&tvEnd, NULL);
	timeSubtract(&tvDiff, &tvEnd, &tvBegin);
	printf("%d x Nerual-Net-fft: \t %ld.%ld\n", TIMES, tvDiff.tv_sec, tvDiff.tv_usec);

/*-----------------------DEBUG-----------------------------*/
/*
	printf("\n\nsignal: \n");
    for(i=0;i<N;i++){
		printf("%lf\n",signal[i][0]);
    } 
	
	printf("\n\nweights_0_1: \n");
    for(i=0;i<NUM_NEURONS;i++){
		for (j=0;j<N;j++){
			printf("%lf\n",weights_0_1[i][j]);
		}
    } 

    printf("\n\nlayer_1: \n");
    for(i=0;i<NUM_NEURONS;i++){
		printf("%lf\n",layer_1[i][0]);
    }

	printf("\n\nweights_1_2: \n");
    for(i=0;i<N;i++){
		for (j=0;j<NUM_NEURONS;j++){
			printf("%lf\n",weights_1_2[i][j]);
		}    
	} 
*/
/*
	printf("\n\noutput: \n");
    for(i=0;i<N;i++){
		printf("%lf\n",output[i][0]);
    }
*/
/*-----------------------------------------------------*/
    return 0;
}

