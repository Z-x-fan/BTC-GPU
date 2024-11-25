#pragma once
#ifndef __TENSOR_SGD__
#define __TENSOR_SGD__

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

using namespace std;

extern int I;
extern int J;
extern int K;

extern int epochs;
extern int r;
extern double lr;
extern double reg;
extern int block_s;
extern float sample_rate;
extern int thread_size;
extern int flag_lockfree;
extern int flag_preproccess;
extern int max_num;
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
	int x_start, x_end;
	int y_start, y_end;
	int z_start, z_end;
	int id;
};
struct LF_node{
	int x, y, z;
	double rate;
	double coe_x;
	double coe_y;
	double coe_z;
};

struct pre_node_1 {
	LF_node* t;
	int num_t;
	pre_node_1* next;
};
struct t_node {
	int num_b;
	int num_p;
	//pre_node_tmp* next;
};
struct pre_node {
	list<int> x, y, z;
	list<double> rate;
	list<double> coe_x;
	list<double> coe_y;
	list<double> coe_z;
	pre_node* next;
};
struct PB_node {
	list<int> Bid;
};


void tensor_SGD(double* t_1);

#endif