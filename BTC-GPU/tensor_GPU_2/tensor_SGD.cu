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
#include "sgd_Kernel.h"
#include "block.h"
#include <sstream>

using namespace std;

int I = 144;
int J = 288;
int K = 168;

int epochs = 1000;
int r = 4;
double lr = 0.0002;
double reg = 0.02;
int block_s = 4096;     //block存放数据上限
//float sample_rate = (float)0.5;
int thread_size = 256;
int flag_lockfree = 0;
int flag_preproccess = 1;
int max_num = 8;
queue<double> num1;

//随机生成矩阵
void getRand(double* Mat, int I, int r)
{
	for (int i = 0; i < I; i++)
	{
		for (int j = 0; j < r; j++)
		{
			random_device rd;
			mt19937 gen(rd());
			uniform_real_distribution<> distribution(0.1, 0.9);
			double random = distribution(gen);
			Mat[i * r + j] = random;

		}
	}

}
//取样
void sample(double* t, double* s, int* t_s, float num_p, int I, int J, int K) {

	int num_z = floor(num_p * K);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			int num = 0;
			int* tmp = new int[K]();
			while (num < num_z)
			{
				random_device rd;
				mt19937 gen(rd());
				uniform_real_distribution<> distribution(0, K * 1.1);
				int k = distribution(gen);

				if (k < K && tmp[k] == 0) {
					tmp[k] = 1;
					t_s[i * J * K + j * K + k] = 1;
					num++;
				}
			}
		}
	}

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				t[i * J * K + j * K + m] = t[i * J * K + j * K + m] * t_s[i * J * K + j * K + m];

			}
		}
	}

}
vector<vector<vector<double>>> sample_v(vector<vector<vector<double>>> S, double* t, float num_p, int I, int J, int K) {
	int num_z = floor(num_p * K);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			int num = 0;
			while (num < num_z)
			{
				random_device rd;
				mt19937 gen(rd());
				uniform_real_distribution<> distribution(0, K * 1.1);
				int k = distribution(gen);

				if (k < K && S[i][j][k] == 0) {
					S[i][j][k] = t[i * J * K + j * K + k];
					num++;
				}
			}
		}
	}

	return S;
}
int sample_vary(double* s, double* t, int* t_s, float num_p, int I, int J, int K) {
	int num_z = floor(num_p * K);
	int num = 0;
	int nnz = 0;
	while (num < num_z) {
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<> distribution_x(0, I - 1);
		int x = distribution_x(gen);
		uniform_real_distribution<> distribution_y(0, J - 1);
		int y = distribution_y(gen);
		uniform_real_distribution<> distribution_z(0, K - 1);
		int z = distribution_z(gen);

		if (t_s[x * J * K + y * K + z] == 0) {
			t_s[x * J * K + y * K + z] = 1;
			num++;
		}
	}

	for (int i = 0; i < I * J * K; i++) {
		s[i] = t[i] * t_s[i];
	}
	for (int i = 0; i < I * J * K; i++) {
		if (s[i] > 0)
			nnz++;
	}
	return nnz;
}

//样本S转为一维数组，记录非零个数
int Create_t(double* t, int I, int J, int K) {
	int nnz = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				if (t[i * J * K + j * K + m] > 0) {
					nnz++;
				}
				else {
					printf("%d %d %d %f \n", i, j, m, t[i * J * K + j * K + m]);
				}
			}
		}
	}
	return nnz;
}

//非零个数
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

//数据归一化
void Normalization(double* t_2, double* t_1, double& max, double& min) {
	max = min = t_2[0];
	for (int i = 0; i < I * J * K; i++) {
		max = max > t_2[i] ? max : t_2[i];
		min = min < t_2[i] ? min : t_2[i];
	}
	for (int i = 0; i < I * J * K; i++) {
		t_1[i] = 0.01 + 0.99 * (t_2[i] - min) / (max - min);
	}
}

void tensor_SGD(double* t_2) {
	
	double* t_1 = new double[I * J * K]();

	t_1 = t_2;

	double rate = 0.5;
	//	while (rate <= 1) {
	printf("before: %f %f %f %f\n", t_1[0], t_1[1], t_1[(I - 1) * J * K + (J - 1) * K + (K - 1) - 1], t_1[(I - 1) * J * K + (J - 1) * K + (K - 1)]);
	

	//从文件中载入特征矩阵
	double* a = new double[I * r];
	double* b = new double[J * r];
	double* c = new double[K * r];
	getRand(a, I, r);
	getRand(b, J, r);
	getRand(c, K, r);

	cout << rate << endl;
	double* t = new double[I * J * K];
	string address_data("C:/Users/12625/Desktop/tensor/AbileneTM_Tensor_");
	string str;
	stringstream ss;
	ss << rate;
	ss >> str;
	str += ".txt";
	address_data += str;

	ifstream infile(address_data);
	int sum = 0;
	while (infile) {
		infile >> t[sum];
		sum++;
	}

	int nnz = 0;

	//S = sample(t_1, rate, I, J, K);
	//S = sample_vary(S, t_1, 0.1, I, J, K);
	//nnz = Create_t(S, t, I, J, K);
	nnz = Create_tmp(t, I, J, K);
	//t = t_s;
	//nnz=Create_t(t, I, J, K);
	 
	/*if (rate < 1) {
		//S = sample(t_1, rate, I, J, K);
		//S = sample_vary(S, t_1, 0.1, I, J, K);
		//nnz = Create_t(S, t, I, J, K);
		nnz = Create_tmp(t_s, t, t_1, I, J, K);
		//t = t_s;
		//nnz=Create_t(t, I, J, K);
	}
	else {
		t = t_1;
		t_s = t_1;
		//nnz = Create_tmp(t, I, J, K);
		nnz = Create_t(t, I, J, K);
	}*/

	/*		ofstream ofs_s;
			ofs_s.open("C:\\Users\\12625\\Desktop\\tensor\\test_s.txt", ios::out | ios::in | ios::trunc);

			for (int i = 0; i < I; i++) {
				for (int j = 0; j < J; j++) {
					for (int m = 0; m < K; m++) {

						ofs_s << S[i][j][m] << " ";
					}
				}
			}
			ofs_s.close();*/

	LF_node* pre = NULL;
	int* num_parallel_pre = NULL;
	int num_block = 0;//block的数量
	int num_parallel = 0;//并行数量
	int max_parallel = 0;//最大并行block数
	LF_node* LF = NULL;
	int* num_LF = NULL;


	block_problem(t_1, t, a, b, c, pre, num_parallel_pre, num_block, num_parallel, max_parallel, LF, num_LF, nnz, rate);

	double* t_3 = new double[I * J * K];
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				double sum_t_1 = 0;
				for (int n = 0; n < r; n++) {
					sum_t_1 += a[i * r + n] * b[j * r + n] * c[m * r + n];
				}
				t_3[i * J * K + j * K + m] = sum_t_1;
			}
		}
	}

	printf("after: %f %f %f %f\n", t_3[0], t_3[1], t_3[(I - 1) * J * K + (J - 1) * K + (K - 1) - 1], t_3[(I - 1) * J * K + (J - 1) * K + (K - 1)]);
	delete[]a;
	delete[]b;
	delete[]c;

	//		delete []t;

	delete LF;
	delete num_LF;
	delete num_parallel_pre;
	delete[]t_3;
	rate += 0.1;
	//}

	delete[]t_1;
	/*
		double* a = new double[I * r];
		double* b = new double[J * r];
		double* c = new double[K * r];
		getRand(a, I, r);
		getRand(b, J, r);
		getRand(c, K, r);

		vector<vector<vector<double>>> S(I, vector<vector<double>>(J, vector<double>(K, 0)));
		double* t = new double[I * J * K];
		int nnz = 0;
		if (sample_rate < 1) {
			S = sample(t_1, 0.5, I, J, K);
			nnz = Create_t(S, t, I, J, K);
		}
		else {
			t = t_1;
			nnz = I * J * K;
		}

		b_node* bs = NULL;
		int* num_bs = NULL;
		int num_block;
		int num_parallel;
		int max_parallel;
		LF_node* LF = NULL;
		int* num_LF = NULL;
		LF_node* pre_t = NULL;
		int num_parallel_t;
		int* num_parallel_pre = NULL;

		block_problem(t_1, t, a, b, c, bs, num_bs, num_block, num_parallel, max_parallel, LF, num_LF, pre_t, num_parallel_t, num_parallel_pre, nnz, rate);

		for (int i = 0; i < I; i++) {
			for (int j = 0; j < J; j++) {
				for (int m = 0; m < K; m++) {
					for (int n = 0; n < r; n++) {
						t_1[i * J * K + j * K + m] = a[i * r + n] * b[j * r + n] * c[m * r + n];
					}

				}
			}
		}*/

		/*
			double* t_2 = new double[I * J * K];
			ofstream ofs;
			ofs.open("C:/Users/12625/Desktop/tensor/tensor_after.txt", ios::out | ios::in);
			for (int i = 0; i < I; i++) {
				for (int j = 0; j < J; j++) {
					for (int m = 0; m < K; m++) {
						double sum_t_1 = 0;
						for (int n = 0; n < r; n++) {
							sum_t_1 += a[i * r + n] * b[j * r + n] * c[m * r + n];
						}
						t_2[i * J * K + j * K + m] = sum_t_1;
						ofs << t_2[i * J * K + j * K + m] << " ";

					}
				}
			}
			ofs.close();*/

}



