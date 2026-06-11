#pragma once
#ifndef __SGD_KERNEL__
#define __SGD_KERNEL__
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include<iostream>
#include <vector>
//#include<Windows.h>
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

void sgd_train(const T_node* train_entries,
	int train_nnz,
	const T_node* test_entries,
	int test_nnz,
	double* a,
	double* b,
	double* c,
	int num_parallel,
	int max_parallel,
	LF_node* LF,
	int* num_LF,
	double rate,
	int *num_bs,
	b_node* bs,
	int num_scheduled_bs);
#endif
