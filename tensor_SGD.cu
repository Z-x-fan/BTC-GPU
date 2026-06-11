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

int epochs = 1000;
int r = 4;
double lr = 0.001;   //LF 0.0001   0.004
double reg = 0.05;    //LF 0.04	0.005
int block_s = 1024;     //block
int thread_size = 1024;
int flag_lockfree = 0;
int flag_preproccess = 1;
int parallel_sequence_size = 0; // 0: auto; >0: fixed blocks per parallel sequence
int requested_gpu_count = 4; // 0: all visible GPUs; >0: at most this many GPUs
queue<double> num1;
string dataset_name = "tensor";
string output_root = "output";
double train_rate = 0.8;

void getRand(double* Mat, int I, int r){
	static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> distribution(0.02, 0.1);

    for (int i = 0; i < I; i++) {
        for (int j = 0; j < r; j++) {
            Mat[i * r + j] = distribution(gen);
        }
    }
}

void tensor_SGD(const vector<T_node>& train_entries, const vector<T_node>& test_entries) {
	cout << "dataset=" << dataset_name
		<< " dims=" << I << "x" << J << "x" << K
		<< " train_nnz=" << train_entries.size()
		<< " test_nnz=" << test_entries.size() << endl;

	double* a = new double[I * r];
	double* b = new double[J * r];
	double* c = new double[K * r];
	getRand(a, I, r);
	getRand(b, J, r);
	getRand(c, K, r);

	block_problem(
		train_entries.data(),
		(int)train_entries.size(),
		test_entries.data(),
		(int)test_entries.size(),
		a,
		b,
		c,
		train_rate);

	if (!train_entries.empty()) {
		const T_node& first = train_entries[0];
		double prediction = 0;
		for (int n = 0; n < r; n++) {
			prediction += a[first.x * r + n] * b[first.y * r + n] * c[first.z * r + n];
		}
		printf("first train entry: (%d,%d,%d) ref=%f pred=%f\n", first.x, first.y, first.z, first.rate, prediction);
	}

	delete[] a;
	delete[] b;
	delete[] c;
}
