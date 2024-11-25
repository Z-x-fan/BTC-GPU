#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include<iostream>
#include <vector>
#include<Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include<queue>
#include <bitset>
#include<list>
#include<ctime>
#include <random>
#include <fstream>
#include "tensor_SGD.h"

using namespace std;

int main() {

	double* t_2 = new double[I * J * K];
	ifstream infile("C:/Users/12625/Desktop/tensor/AbileneTM_Tensor.txt");
	int sum = 0;
	while (infile) {
		infile >> t_2[sum];
		sum++;
	}

	tensor_SGD(t_2);

	
	return 0;
}