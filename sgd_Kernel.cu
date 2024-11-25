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
#include<sstream>
#include "tensor_SGD.h"
#include "sgd_Kernel.h"
#include "block.h"

using namespace std;

double get_time_2(void)
{
	LARGE_INTEGER timer;
	static LARGE_INTEGER fre;
	static int init = 0;
	double t;

	if (init != 1) {
		QueryPerformanceFrequency(&fre);
		init = 1;
	}

	QueryPerformanceCounter(&timer);

	t = timer.QuadPart * 1. / (double)fre.QuadPart;

	return t;
}
double RMSE_GPU_S(double* s, double* a, double* b, double* c) {
	double total_error = 0;
	int num = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {

				if (s[i * J * K + j * K + m] > 0) {
					num++;
					double dot = 0;
					for (int n = 0; n < r; n++) {
						dot += a[i * r + n] * b[j * r + n] * c[m * r + n];
					}
					double error = s[i * J * K + j * K + m] - dot;
					total_error += pow(error, 2);
				}
			}

		}
	}
	double Error = sqrt(total_error / num);
	return Error;
}
double RMSE_GPU_T(double* s, double* t, double* a, double* b, double* c) {
	double total_error = 0;
	int num = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {

				if (s[i * J * K + j * K + m] == 0) {
					num++;
					double dot = 0;
					for (int n = 0; n < r; n++) {
						dot += a[i * r + n] * b[j * r + n] * c[m * r + n];
					}
					double error = t[i * J * K + j * K + m] - dot;
					total_error += pow(error, 2);
				}
			}

		}
	}
	double Error = sqrt(total_error / num);
	return Error;
}


__global__ void tensor_GPU(double* dX, b_node* d_bs, int* d_num_bs, double* dA, double* dB, double* dC, int num_parallel, int I, int J, int K, int r, double lr, double reg, Node_conflict* B_conf) {
	int bx = blockIdx.x;
	int len = 0;
	//printf("%d %d %d   ", bx, num_parallel, d_num_bs[0]);
	for (int i = 0; i < num_parallel; i++) {
		
		//printf("%d %d %d %d   ", bx, num_parallel, i, d_num_bs[0]);
		if (bx < d_num_bs[i]) {
			//printf("%d %d %d %d   ", bx, num_parallel, d_num_bs[i], d_bs[0].x_start);
			int x_start = d_bs[bx + len].x_start;
			int x_end = d_bs[bx + len].x_end;
			int x_s = x_end - x_start;
			int y_start = d_bs[bx + len].y_start;
			int y_end = d_bs[bx + len].y_end;
			int y_s = y_end - y_start;
			int z_start = d_bs[bx + len].z_start;
			int z_end = d_bs[bx + len].z_end;
			int z_s = z_end - z_start;
			int id = d_bs[bx + len].id;
			int cox = B_conf[id].coe_x;
			int coy = B_conf[id].coe_y;
			int coz = B_conf[id].coe_z;
			int tx = threadIdx.x;
			int ty = threadIdx.x;
			int tz = threadIdx.x;
			//printf("%d %d %d    ",bx, d_num_bs[1], d_bs[bx + len].x_start);
			if (tx < x_s && ty < y_s && tz < z_s && tx == ty && ty == tz) {
				int min = 0;
				min = x_s < y_s ? x_s : y_s;
				min = min < z_s ? min : z_s;
				int x = tx + x_start;
				int y = tx + y_start;
				int z = tz + z_start;
				//printf("%d %d %d %d %d %d %d %d %d\n", d_num_bs[i], bx, i, x_start, x_end, y_start, y_end, z_start, z_end);
				if (min == x_s) {
					for (int count = 0; count < y_s * z_s; count++) {

						if (dX[x * J * K + y * K + z] > 0) {
							//printf("%d %d %d %f\n", x, y, z, dX[x * J * K + y * K + z]);
							double dot = 0;
							for (int n = 0; n < r; n++) {
								dot += dA[x * r + n] * dB[y * r + n] * dC[z * r + n];
							}
							double error = dX[x * J * K + y * K + z] - dot;
							

							for (int n = 0; n < r; n++) {
								dA[x * r + n] += cox * lr * (error * dB[y * r + n] * dC[z * r + n] - reg * dA[x * r + n]);
								dB[y * r + n] += coy * lr * (error * dA[x * r + n] * dC[z * r + n] - reg * dB[y * r + n]);
								dC[z * r + n] += coz * lr * (error * dB[y * r + n] * dA[x * r + n] - reg * dC[z * r + n]);
							}
						}
						if (z == z_end - 1) {
							y = (y + 1) % y_end;
						}
						z = (z + 1) % z_end;

					}
				}

				if (min == y_s) {

					for (int count = 0; count < x_s * z_s; count++) {

						if (dX[x * J * K + y * K + z] > 0) {
							double dot = 0;
							for (int n = 0; n < r; n++) {
								dot += dA[x * r + n] * dB[y * r + n] * dC[z * r + n];
							}
							double error = dX[x * J * K + y * K + z] - dot;

							for (int n = 0; n < r; n++) {
								dA[x * r + n] += lr * (error * dB[y * r + n] * dC[z * r + n] - reg * dA[x * r + n]);
								dB[y * r + n] += lr * (error * dA[x * r + n] * dC[z * r + n] - reg * dB[y * r + n]);
								dC[z * r + n] += lr * (error * dB[y * r + n] * dA[x * r + n] - reg * dC[z * r + n]);
							}
						}

						if (z == x_end - 1) {
							x = (x + 1) % x_end;
						}
						z = (z + 1) % z_end;
					}

				}

				if (min == z_s) {

					for (int count = 0; count < x_s * y_s; count++) {

						if (dX[x * J * K + y * K + z] > 0) {
							double dot = 0;
							for (int n = 0; n < r; n++) {
								dot += dA[x * r + n] * dB[y * r + n] * dC[z * r + n];
							}
							double error = dX[x * J * K + y * K + z] - dot;

							for (int n = 0; n < r; n++) {
								dA[x * r + n] += lr * (error * dB[y * r + n] * dC[z * r + n] - reg * dA[x * r + n]);
								dB[y * r + n] += lr * (error * dA[x * r + n] * dC[z * r + n] - reg * dB[y * r + n]);
								dC[z * r + n] += lr * (error * dB[y * r + n] * dA[x * r + n] - reg * dC[z * r + n]);
							}
						}

						if (y == z_end - 1) {
							x = (x + 1) % x_end;
						}
						y = (y + 1) % y_end;
					}

				}
			}

		}
		len = len + d_num_bs[i];
	}

}

//lack_free
__global__ void tensor_LF(LF_node* d_LF, int* d_num_LF, double* dA, double* dB, double* dC, int num_parallel, int thread_size, int I, int J, int K, int r, double lr, double reg) {
	int len = 0;
	for (int i = 0; i < num_parallel; i++) {
		int tx = threadIdx.x;
		if (thread_size < d_num_LF[i]) {
			float tmp_num = d_num_LF[i];
			float tmp = tmp_num / thread_size;
			int count = ceil(tmp);
			for (int j = 0; j < count; j++) {
				if (tx < d_num_LF[i]) {
					int x = d_LF[len + tx].x;
					int y = d_LF[len + tx].y;
					int z = d_LF[len + tx].z;
					double dot = 0;
					for (int n = 0; n < r; n++) {
						dot += dA[x * r + n] * dB[y * r + n] * dC[z * r + n];
					}
					double error = d_LF[len + tx].rate - dot;

					for (int n = 0; n < r; n++) {
						dA[x * r + n] += lr * (error * dB[y * r + n] * dC[z * r + n] - reg * dA[x * r + n]);
						dB[y * r + n] += lr * (error * dA[x * r + n] * dC[z * r + n] - reg * dB[y * r + n]);
						dC[z * r + n] += lr * (error * dB[y * r + n] * dA[x * r + n] - reg * dC[z * r + n]);
					}
				}
				tx = tx + thread_size;
			}
		}
		else {
			if (tx < d_num_LF[i]) {
				int x = d_LF[len + tx].x;
				int y = d_LF[len + tx].y;
				int z = d_LF[len + tx].z;
				double dot = 0;
				for (int n = 0; n < r; n++) {
					dot += dA[x * r + n] * dB[y * r + n] * dC[z * r + n];
				}
				double error = d_LF[len + tx].rate - dot;

				for (int n = 0; n < r; n++) {
					dA[x * r + n] += lr * (error * dB[y * r + n] * dC[z * r + n] - reg * dA[x * r + n]);
					dB[y * r + n] += lr * (error * dA[x * r + n] * dC[z * r + n] - reg * dB[y * r + n]);
					dC[z * r + n] += lr * (error * dB[y * r + n] * dA[x * r + n] - reg * dC[z * r + n]);
				}
			}
		}
		len += d_num_LF[i];
	}

}
__global__ void tensor_pre_1(LF_node* d_pre, int* d_bsize, t_node* d_tsize, int* d_t_bnum, int* d_t_block_p, double* dA, double* dB, double* dC, int num_parallel, int thread_size, int I, int J, int K, int r, double lr, double reg) {
	int len = 0;
	int len_b = 0;
	int b_id = 0;
	printf("%d \n", d_pre[0].rate);
	for (int i = 0; i < num_parallel; i++) {
		int bx = blockIdx.x;

		if (bx < d_bsize[i]) {
			bx = bx + b_id;
			int t_id = 0;

			for (int j = 0; j < d_tsize[bx].num_b; j++) {
				int tx = threadIdx.x;
				if (tx < d_tsize[bx].num_p) {
					tx += t_id;

					printf("%d %d %d %d %f \n", i, bx, tx, d_t_bnum[bx], d_pre[0].rate);
					if (d_pre[d_t_bnum[bx] + tx].rate > 0) {
						//printf("%d %d %d %d\n", i, tx, len_b, d_pre[d_t_bnum[bx] + tx].rate);
						int x = d_pre[d_t_bnum[bx] + tx].x;
						int y = d_pre[d_t_bnum[bx] + tx].y;
						int z = d_pre[d_t_bnum[bx] + tx].z;
						double dot = 0;
						for (int n = 0; n < r; n++) {
							dot += dA[x * r + n] * dB[y * r + n] * dC[z * r + n];
						}
						double error = d_pre[d_t_bnum[bx] + tx].rate - dot;

						for (int n = 0; n < r; n++) {
							dA[x * r + n] += lr * (error * dB[y * r + n] * dC[z * r + n] - reg * dA[x * r + n]);
							dB[y * r + n] += lr * (error * dA[x * r + n] * dC[z * r + n] - reg * dB[y * r + n]);
							dC[z * r + n] += lr * (error * dB[y * r + n] * dA[x * r + n] - reg * dC[z * r + n]);
						}
					}
				}
				t_id += d_tsize[bx].num_p;
			}
			b_id += d_bsize[i];
		}
	}
}

//preproccess
__global__ void tensor_pre(LF_node* d_pre, int* d_num_parallel_pre, double* dA, double* dB, double* dC, int num_parallel_t, int thread_size, int I, int J, int K, int r, double lr, double reg) {
	int len = 0;
	for (int i = 0; i < num_parallel_t; i++) {
		if (thread_size < d_num_parallel_pre[i]) {
			int tx = threadIdx.x;
			//printf("%d ", d_num_parallel_pre[i]);
			float tmp_num = d_num_parallel_pre[i];
			float tmp = tmp_num / thread_size;
			int count = ceil(tmp);
			for (int j = 0; j < count; j++) {

				if (tx < d_num_parallel_pre[i]) {
					//printf("%d %d %d %d   ",tx, x, y, z);

					int x = d_pre[len + tx].x;
					int y = d_pre[len + tx].y;
					int z = d_pre[len + tx].z;
					double dot = 0;
					for (int n = 0; n < r; n++) {
						dot += dA[x * r + n] * dB[y * r + n] * dC[z * r + n];
					}
					double error = d_pre[len + tx].rate - dot;
					double coe_x = d_pre[len + tx].coe_x;
					double coe_y = d_pre[len + tx].coe_y;
					double coe_z = d_pre[len + tx].coe_z;

					for (int n = 0; n < r; n++) {
						dA[x * r + n] += coe_x * lr * (error * dB[y * r + n] * dC[z * r + n] - reg * dA[x * r + n]);
						dB[y * r + n] += coe_y * lr * (error * dA[x * r + n] * dC[z * r + n] - reg * dB[y * r + n]);
						dC[z * r + n] += coe_z * lr * (error * dB[y * r + n] * dA[x * r + n] - reg * dC[z * r + n]);
					}


				}
				tx = tx + thread_size;
			}
		}
		else {
			int tx = threadIdx.x;
			if (tx < d_num_parallel_pre[i]) {
				//printf("%d ", d_num_parallel_pre[i]);

				int x = d_pre[len + tx].x;
				int y = d_pre[len + tx].y;
				int z = d_pre[len + tx].z;

				//printf("%d %d %d %d %f\n   ",tx, x, y, z, d_pre[len + tx].rate);


				double dot = 0;
				for (int n = 0; n < r; n++) {
					dot += dA[x * r + n] * dB[y * r + n] * dC[z * r + n];
				}
				double error = d_pre[len + tx].rate - dot;
				double coe_x = d_pre[len + tx].coe_x;
				double coe_y = d_pre[len + tx].coe_y;
				double coe_z = d_pre[len + tx].coe_z;

				for (int n = 0; n < r; n++) {
					dA[x * r + n] += coe_x * lr * (error * dB[y * r + n] * dC[z * r + n] - reg * dA[x * r + n]);
					dB[y * r + n] += coe_y * lr * (error * dA[x * r + n] * dC[z * r + n] - reg * dB[y * r + n]);
					dC[z * r + n] += coe_z * lr * (error * dB[y * r + n] * dA[x * r + n] - reg * dC[z * r + n]);
				}

			}
		}
		len += d_num_parallel_pre[i];
	}

}
void clear(queue<double>& q) {
	queue<double> empty;
	swap(empty, q);
}

void sgd_train(double* t_1,
	double* t,
	double* a,
	double* b,
	double* c,
	LF_node* pre,
	int* num_parallel_pre,
	int num_parallel_t,
	int num_block,
	int num_parallel,
	int max_parallel,
	LF_node* LF,
	int* num_LF,
	int nnz,
	double rate,
	int* num_bs,
	b_node* bs,
	Node_conflict* B_conf) {

	ofstream ofs_time_1, ofs_error;
	string address("C:/Users/12625/Desktop/tensor/A_");
	string str;
	stringstream ss;
	ss << rate;
	ss >> str;
	str += "/";
	address += str;
	string add_error, add_time, add_block;
	add_error += address;
	add_error += "error.txt";
	add_time += address;
	add_time += "time.txt";
	add_block += address;
	add_block += "block.txt";

	double* dA, * dB, * dC;
	double* dX;

	int nbytesX = I * J * K * (sizeof(double));
	int nbytesA = I * r * (sizeof(double));
	int nbytesB = J * r * (sizeof(double));
	int nbytesC = K * r * (sizeof(double));

	cudaMalloc((void**)&dX, nbytesX);
	cudaMalloc((void**)&dA, nbytesA);
	cudaMalloc((void**)&dB, nbytesB);
	cudaMalloc((void**)&dC, nbytesC);
	if (dA == NULL || dB == NULL || dC == NULL) {
		printf("couldn't allocate GPU memory\n");
	}

	cudaMemcpy(dX, t, nbytesX, cudaMemcpyHostToDevice);
	cudaMemcpy(dA, a, nbytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, b, nbytesB, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, c, nbytesC, cudaMemcpyHostToDevice);

	clear(num1);

	ofs_error.open(add_error, ios::out | ios::in | ios::trunc);
	ofs_time_1.open(add_time, ios::out | ios::in | ios::app);

	if (flag_lockfree == 1) {
		LF_node* d_LF;
		int* d_num_LF;
		int nbytesLF = nnz * (sizeof(LF_node));
		int nbytesnumLF = num_parallel * (sizeof(int));
		cudaMalloc((void**)&d_LF, nbytesLF);
		cudaMalloc((void**)&d_num_LF, nbytesnumLF);

		cudaMemcpy(d_LF, LF, nbytesLF, cudaMemcpyHostToDevice);
		cudaMemcpy(d_num_LF, num_LF, nbytesnumLF, cudaMemcpyHostToDevice);

		for (int epoch = 0; epoch < epochs; epoch++) {

			double td2 = get_time_2();
			tensor_LF << <1, thread_size >> > (d_LF, d_num_LF, dA, dB, dC, num_parallel, thread_size, I, J, K, r, lr, reg);
			td2 = get_time_2() - td2;
			ofs_time_1 << td2 << endl;

			cudaMemcpy(a, dA, nbytesA, cudaMemcpyDeviceToHost);
			cudaMemcpy(b, dB, nbytesB, cudaMemcpyDeviceToHost);
			cudaMemcpy(c, dC, nbytesC, cudaMemcpyDeviceToHost);

			double Error_1 = RMSE_GPU_S(t, a, b, c);
			double Error_2 = RMSE_GPU_T(t, t_1, a, b, c);
			ofs_error << Error_1 << " " << Error_2 << endl;
			cout << "Epoch:" << epoch << ' ' << Error_1 << " " << Error_2 << '\n';

			num1.push(Error_1);
			if (num1.size() > 2) {
				num1.pop();
			}

			if (num1.size() >= 2) {
				if (num1.front() - num1.back() < 0.0000001) {
					break;
				}
			}
		}

		cudaFree(d_LF);
		cudaFree(d_num_LF);
	}
	/*
	if (flag_preproccess == 1) {
		
		b_node* d_bs;
		int* d_num_bs;
		Node_conflict* d_B_conf;
		int nbytesbs = nnz * (sizeof(b_node));
		int nbytesnumbs = num_parallel * (sizeof(int));
		int  nbytesconf = num_block * (sizeof(Node_conflict));
		cudaMalloc((void**)&d_bs, nbytesbs);
		cudaMalloc((void**)&d_num_bs, nbytesnumbs);
		cudaMalloc((void**)&d_B_conf, nbytesconf);

		cudaMemcpy(d_bs, bs, nbytesbs, cudaMemcpyHostToDevice);
		cudaMemcpy(d_num_bs, num_bs, nbytesnumbs, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B_conf, B_conf, nbytesconf, cudaMemcpyHostToDevice);

		for (int epoch = 0; epoch < epochs; epoch++) {
			int len_1 = 0;

			double td4 = get_time_2();
			tensor_GPU << <max_num, thread_size >> > (dX, d_bs, d_num_bs, dA, dB, dC, num_parallel, I, J, K, r, lr, reg, d_B_conf);
			td4 = get_time_2() - td4;
			ofs_time_1 << td4 << endl;

			cudaMemcpy(a, dA, nbytesA, cudaMemcpyDeviceToHost);
			cudaMemcpy(b, dB, nbytesB, cudaMemcpyDeviceToHost);
			cudaMemcpy(c, dC, nbytesC, cudaMemcpyDeviceToHost);

			double Error_1 = RMSE_GPU_S(t, a, b, c);
			double Error_2 = RMSE_GPU_T(t, t_1, a, b, c);
			ofs_error << Error_1 << " " << Error_2 << endl;
			cout << "Epoch:" << epoch << ' ' << Error_1 << " " << Error_2 << '\n';

			num1.push(Error_1);
			if (num1.size() > 2) {
				num1.pop();
			}

			if (num1.size() >= 2) {
				if (num1.front() - num1.back() < 0.0000001) {
					break;
				}
			}
		}

		

	}*/

	if (flag_preproccess == 1) {
		LF_node* d_pre;
		int* d_num_parallel_pre;
		int nbytespre = nnz * (sizeof(LF_node));
		int nbytesnumpre = num_parallel_t * (sizeof(int));
		cudaMalloc((void**)&d_pre, nbytespre);
		cudaMalloc((void**)&d_num_parallel_pre, nbytesnumpre);

		cudaMemcpy(d_pre, pre, nbytespre, cudaMemcpyHostToDevice);
		cudaMemcpy(d_num_parallel_pre, num_parallel_pre, nbytesnumpre, cudaMemcpyHostToDevice);

		for (int epoch = 0; epoch < epochs; epoch++) {
			int len_1 = 0;

			double td4 = get_time_2();
			tensor_GPU <<<max_parallel, thread_size >>> (d_pre, d_num_parallel_pre, dA, dB, dC, num_parallel_t, thread_size, I, J, K, r, lr, reg);
			td4 = get_time_2() - td4;
			ofs_time_1 << td4 << endl;

			cudaMemcpy(a, dA, nbytesA, cudaMemcpyDeviceToHost);
			cudaMemcpy(b, dB, nbytesB, cudaMemcpyDeviceToHost);
			cudaMemcpy(c, dC, nbytesC, cudaMemcpyDeviceToHost);

			double Error_1 = RMSE_GPU_S(t, a, b, c);
			double Error_2 = RMSE_GPU_T(t, t_1, a, b, c);
			ofs_error << Error_1 << " " << Error_2 << endl;
			cout << "Epoch:" << epoch << ' ' << Error_1 << " " << Error_2 << '\n';

			num1.push(Error_1);
			if (num1.size() > 2) {
				num1.pop();
			}

			if (num1.size() >= 2) {
				if (num1.front() - num1.back() < 0.0000001) {
					break;
				}
			}
		}

		cudaFree(d_pre);

	}
	/*if (flag_preproccess == 1) {
		LF_node* d_pre;
		int* d_bsize;
		t_node* d_tsize;
		int* d_t_bnum;
		int* d_t_block_p;
		int nbytespre = total * (sizeof(LF_node));
		int nbytesnumb = num_parallel * (sizeof(int));
		int nbytesnumt = num_block * (sizeof(t_node));
		int nbytesTbnum = num_block * (sizeof(int));
		int nbytesTbp = num_parallel * (sizeof(int));
		cudaMalloc((void**)&d_pre, nbytespre);
		cudaMalloc((void**)&d_bsize, nbytesnumb);
		cudaMalloc((void**)&d_tsize, nbytesnumt);
		cudaMalloc((void**)&d_t_bnum, nbytesTbnum);
		cudaMalloc((void**)&d_t_block_p, nbytesTbp);

		cudaMemcpy(d_pre, pre, nbytespre, cudaMemcpyHostToDevice);
		cudaMemcpy(d_bsize, b_size, nbytesnumb, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tsize, t_size, nbytesnumt, cudaMemcpyHostToDevice);
		cudaMemcpy(d_t_bnum, t_block_num, nbytesTbnum, cudaMemcpyHostToDevice);
		cudaMemcpy(d_t_block_p, t_block_p, nbytesTbp, cudaMemcpyHostToDevice);

		for (int epoch = 0; epoch < epochs; epoch++) {

			double td4 = get_time_2();
			tensor_pre << <max_parallel, thread_size >> > (d_pre, d_bsize, d_tsize, d_t_bnum, d_t_block_p, dA, dB, dC, num_parallel, thread_size, I, J, K, r, lr, reg);
			int len_1 = 0;
			td4 = get_time_2() - td4;
			ofs_time_1 << td4 << endl;

			cudaMemcpy(a, dA, nbytesA, cudaMemcpyDeviceToHost);
			cudaMemcpy(b, dB, nbytesB, cudaMemcpyDeviceToHost);
			cudaMemcpy(c, dC, nbytesC, cudaMemcpyDeviceToHost);

			double Error_1 = RMSE_GPU_S(t, a, b, c, I, J, K);
			double Error_2 = RMSE_GPU_T(t, t_1, a, b, c, I, J, K);
			ofs_error << Error_1 << " " << Error_2 << endl;
			cout << "Epoch:" << epoch << ' ' << Error_1 << " " << Error_2 << '\n';

			num1.push(Error_1);
			if (num1.size() > 2) {
				num1.pop();
			}

			if (num1.size() >= 2) {
				if (num1.front() - num1.back() < 0.0000001) {
					break;
				}
			}
		}

		cudaFree(d_pre);
		cudaFree(d_bsize);
		cudaFree(d_tsize);
	}*/

	ofs_time_1.close();
	cudaFree(dX);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

}