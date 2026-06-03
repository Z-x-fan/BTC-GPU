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

using namespace std;

int main() {
	
	const int data_size = I * J * K;
	double* data_initial = new double[data_size]();  
	ifstream infile("data/AbileneTM_Tensor_0.5.txt");
	if (!infile.is_open()) {
		infile.open("/root/fan/BTC-GPU/data/AbileneTM_Tensor_0.5.txt");
	}
	if (!infile.is_open()) {
		cerr << "Failed to open data/AbileneTM_Tensor_0.5.txt" << endl;
		delete[] data_initial;
		return 1;
	}
	int sum = 0;
	while (sum < data_size && infile >> data_initial[sum]) {
		sum++;
	}
	if (sum < data_size) {
		cerr << "Warning: expected " << data_size << " values, got " << sum << endl;
	}
	tensor_SGD(data_initial);

	delete[] data_initial;
	
	return 0;
}
