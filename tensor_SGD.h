#pragma once
#ifndef __TENSOR_SGD__
#define __TENSOR_SGD__

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

using namespace std;

extern int I;
extern int J;
extern int K;

extern int epochs;
extern int r;
extern double lr;
extern double reg;
extern int block_s;
extern int thread_size;
extern int flag_lockfree;
extern int flag_preproccess;
extern int parallel_sequence_size;
extern queue<double> num1;

struct BTnode{
	vector<bool> x, y, z;
	BTnode* lChild, * rChild;
	int h;
	int block;
};


struct T_node{
	int x, y, z;
};


struct Node_conflict{
	int id;
	int x, y, z;
	double coe_x;
	double coe_y;
	double coe_z;
	double rate;
};

struct BS_node {
	int id;
	vector<bool> x_id, y_id, z_id;
	//int* x_id, * y_id, * z_id;
	int level_x, level_y, level_z;
	int block_num;
	T_node* t;
};

struct Parallel {
	list<int> L;
	Parallel* next;
};

struct b_node {
	int *x;
	int *y;
	int *z;
	int block_num;
	int id;
	double coe_x;
	double coe_y;
	double coe_z;
};

struct LF_node{
	int x, y, z;
	double rate;
	double coe_x;
	double coe_y;
	double coe_z;
};

void tensor_SGD(double* data_initial);

#endif
