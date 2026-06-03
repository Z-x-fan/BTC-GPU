#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
//#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <queue>
#include <bitset>
#include <list>
#include <ctime>
#include <random>
#include <fstream>
#include "tensor_SGD.h"
#include "sgd_Kernel.h"
#include "block.h"
#include <sstream>

using namespace std;

int I = 144;
int J = 288;
int K = 168;

int epochs = 2000;
int r = 20;
double lr = 0.001;   //LF 0.0001   0.004
double reg = 0.01;    //LF 0.04	0.005
int block_s = 1024;     //block           
int thread_size = 1024;
int flag_lockfree = 0;
int flag_preproccess = 1;
int parallel_sequence_size = 0; // 0: auto; >0: fixed blocks per parallel sequence
queue<double> num1;

//   
void getRand(double* Mat, int I, int r){
	static std::random_device rd;
    static std::mt19937 gen(rd());  // 初始化一次
    std::uniform_real_distribution<> distribution(0.1,0.9); // 均匀分布

    for (int i = 0; i < I; i++) {
        for (int j = 0; j < r; j++) {
            Mat[i * r + j] = distribution(gen); // 合法范围 [0.1, 0.9]
        }
    }
}

//ȡ  
//       
int Create_tmp( double* t, int I, int J, int K) {
	int nnz = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				if (t[i * J * K + j * K + m] > 0)
					nnz++;
			}
		}
	}
	return nnz;
}

void tensor_SGD(double* data_initial) {
	
	                                                         
	double rate = 0.5;

	printf("before: %lf %lf %lf %lf\n", data_initial[0], data_initial[1], data_initial[(I - 1) * J * K + (J - 1) * K + (K - 1) - 1], data_initial[(I - 1) * J * K + (J - 1) * K + (K - 1)]);
	

	double* a = new double[I * r];
	double* b = new double[J * r];
	double* c = new double[K * r];
	getRand(a, I, r);
	getRand(b, J, r);
	getRand(c, K, r);
	
	const int data_size = I * J * K;
	double* data_sampling = new double[data_size]();  //        
	string address_data("data/AbileneTM_Tensor_");
	string str;
	stringstream ss;
	ss << rate;
	ss >> str;
	str += ".txt";
	address_data += str;

	ifstream infile(address_data);
	if (!infile.is_open()) {
		string fallback_address("/root/fan/BTC-GPU/");
		fallback_address += address_data;
		infile.open(fallback_address);
	}
	if (!infile.is_open()) {
		cerr << "Failed to open " << address_data << endl;
		delete[] a;
		delete[] b;
		delete[] c;
		delete[] data_sampling;
		return;
	}
	int sum = 0;
	while (sum < data_size && infile >> data_sampling[sum]) {
		sum++;
	}
	if (sum < data_size) {
		cerr << "Warning: expected " << data_size << " values, got " << sum << endl;
	}

	int nnz = 0;
	nnz = Create_tmp(data_sampling, I, J, K);

	cout << rate << endl;

	block_problem(data_initial, data_sampling, a, b, c, nnz, rate);

	double* data_re = new double[I * J * K]; 
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				double sum_t_1 = 0;
				for (int n = 0; n < r; n++) {
					sum_t_1 += a[i * r + n] * b[j * r + n] * c[m * r + n];
				}
				data_re[i * J * K + j * K + m] = sum_t_1;
			}
		}
	}

	printf("after: %f %f %f %f\n", data_re[0], data_re[1], data_re[(I - 1) * J * K + (J - 1) * K + (K - 1) - 1], data_re[(I - 1) * J * K + (J - 1) * K + (K - 1)]);
	delete[]a;
	delete[]b;
	delete[]c;
	delete[]data_sampling;
	delete[]data_re;

}



