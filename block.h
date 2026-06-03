#pragma once
#ifndef __BLOCK__
#define __BLOCK__
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

void block_problem(double* data_initial, 
	double* data_sampling,
	double* a,
	double* b,
	double* c,
	int nnz,
	double rate);
#endif
