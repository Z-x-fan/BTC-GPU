#pragma once
#ifndef __BLOCK__
#define __BLOCK__
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

void block_problem(double* t_1,
	double* t,
	double* a,
	double* b,
	double* c,
	LF_node* pre,
	int* num_parallel_pre,
	int num_block,
	int num_parallel,
	int max_parallel,
	LF_node* LF,
	int* num_LF,
	int nnz,
	double rate);
#endif