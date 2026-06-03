#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
//#include<Windows.h>
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
#include <sstream>
#include "tensor_SGD.h"
#include "sgd_Kernel.h"
#include "block.h"
#include <thread>
#include <chrono>
#include <iomanip>
#include <limits>
#ifndef _WIN32
#include <sys/time.h>
#endif
using namespace std;

double get_time_ms(void) {//s
#ifdef _WIN32
	return std::chrono::duration<double>(
		std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

struct ErrorMetrics {
	double rmse;
	double mae;
	double er;
	int count;
};

double undefined_metric() {
	return std::numeric_limits<double>::quiet_NaN();
}

ErrorMetrics compute_metrics(double* data_sampling, double* data_initial, double* a, double* b, double* c, bool train_entries) {
	double total_abs_error = 0;
	double total_squared_error = 0;
	double total_ref_squared = 0;
	int count = 0;

	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				int idx = i * J * K + j * K + m;
				bool selected = train_entries
					? data_sampling[idx] > 0
					: (data_sampling[idx] == 0 && data_initial[idx] > 0);
				if (!selected) {
					continue;
				}

				double dot = 0;
				for (int n = 0; n < r; n++) {
					dot += a[i * r + n] * b[j * r + n] * c[m * r + n];
				}

				double ref = train_entries ? data_sampling[idx] : data_initial[idx];
				double error = ref - dot;
				total_abs_error += fabs(error);
				total_squared_error += error * error;
				total_ref_squared += ref * ref;
				count++;
			}
		}
	}

	ErrorMetrics metrics;
	metrics.count = count;
	if (count == 0) {
		metrics.rmse = undefined_metric();
		metrics.mae = undefined_metric();
		metrics.er = undefined_metric();
		return metrics;
	}

	metrics.rmse = sqrt(total_squared_error / count);
	metrics.mae = total_abs_error / count;
	metrics.er = total_ref_squared > 0 ? total_squared_error / total_ref_squared : undefined_metric();
	return metrics;
}

void write_metrics_line(ofstream& ofs_error, int epoch, const ErrorMetrics& train_metrics, const ErrorMetrics& test_metrics) {
	ofs_error << epoch << ' '
		<< train_metrics.rmse << ' ' << test_metrics.rmse << ' '
		<< train_metrics.mae << ' ' << test_metrics.mae << ' '
		<< train_metrics.er << ' ' << test_metrics.er << ' '
		<< train_metrics.count << ' ' << test_metrics.count << endl;
}

void print_epoch_rmse(const char* prefix, int numDevices, int epoch, const ErrorMetrics& train_metrics, const ErrorMetrics& test_metrics) {
	if (prefix != NULL && prefix[0] != '\0') {
		cout << prefix << numDevices << ' ';
	}
	cout << "Epoch:" << epoch
		<< " RMSE(train/test)=" << train_metrics.rmse << ' ' << test_metrics.rmse << '\n';
}

void print_final_metrics(const ErrorMetrics& train_metrics, const ErrorMetrics& test_metrics) {
	cout << "Final train RMSE=" << train_metrics.rmse
		<< " MAE=" << train_metrics.mae
		<< " ER=" << train_metrics.er
		<< " count=" << train_metrics.count
		<< " | test RMSE=" << test_metrics.rmse
		<< " MAE=" << test_metrics.mae
		<< " ER=" << test_metrics.er
		<< " count=" << test_metrics.count << endl;
}


double MAE_GPU_S(double* s, double* a, double* b, double* c) {
	double total_error = 0;
	int num = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				int idx = i * J * K + j * K + m;
				if (s[idx] > 0) {
					num++;
					double dot = 0;
					for (int n = 0; n < r; n++) {
						dot += a[i * r + n] * b[j * r + n] * c[m * r + n];
					}
					double error = s[idx] - dot;
					total_error += fabs(error);
				}
			}
		}
	}
	return total_error / num;
}
double MAE_GPU_T(double* s, double* t, double* a, double* b, double* c) {
	double total_error = 0;
	int num = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				int idx = i * J * K + j * K + m;
				if (s[idx] == 0) { // test entry
					num++;
					double dot = 0;
					for (int n = 0; n < r; n++) {
						dot += a[i * r + n] * b[j * r + n] * c[m * r + n];
					}
					double error = t[idx] - dot;
					total_error += fabs(error);
				}
			}
		}
	}
	return total_error / num;
}

double ER_GPU_S(double* s, double* a, double* b, double* c) {
	double total_error = 0;
	double total_ref = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				int idx = i * J * K + j * K + m;
				if (s[idx] > 0) {
					double dot = 0;
					for (int n = 0; n < r; n++) {
						dot += a[i * r + n] * b[j * r + n] * c[m * r + n];
					}
					double error = s[idx] - dot;
					total_error += error * error;
					total_ref += s[idx] * s[idx];
				}
			}
		}
	}
	return total_error / total_ref;
}
double ER_GPU_T(double* s, double* t, double* a, double* b, double* c) {
	double total_error = 0;
	double total_ref = 0;
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			for (int m = 0; m < K; m++) {
				int idx = i * J * K + j * K + m;
				if (s[idx] == 0) { // test entry
					double dot = 0;
					for (int n = 0; n < r; n++) {
						dot += a[i * r + n] * b[j * r + n] * c[m * r + n];
					}
					double error = t[idx] - dot;
					total_error += error * error;
					total_ref += t[idx] * t[idx];
				}
			}
		}
	}
	return total_error / total_ref;
}

__global__ void GPUs( double* dX, b_node* d_block, int* d_parallel_num_gpu, double* dA, double* dB, double* dC, int num_parallel, int I, int J, int K, int r, double lr, double reg, int thread_size) {
	int bx = blockIdx.x;
	int len = 0;
	
	for (int i = 0; i < num_parallel-1; i++) {
		
		if (bx < d_parallel_num_gpu[i]) {
			int block_num = d_block[bx + len].block_num;
			double coe_x = __ldg(&d_block[bx + len].coe_x);
			double coe_y = __ldg(&d_block[bx + len].coe_y);
			double coe_z = __ldg(&d_block[bx + len].coe_z);
			if( thread_size > block_num ){
				int tx = threadIdx.x;
				if(tx < block_num){
					int x = d_block[bx + len].x[tx];
					int y = d_block[bx + len].y[tx];
					int z = d_block[bx + len].z[tx];
					
					if (dX[x * J * K + y * K + z] > 0) {
						double a_vals[100];  // 假设r <= 32
						double b_vals[100];
						double c_vals[100];
						double dot = 0;
                        for (int n = 0; n < r; n++) {
                            a_vals[n] = __ldg(&dA[x * r + n]);
                            b_vals[n] = __ldg(&dB[y * r + n]);
                            c_vals[n] = __ldg(&dC[z * r + n]);
							dot = fma(a_vals[n], b_vals[n] * c_vals[n], dot);
                        }
						double error = __ldg(&dX[x * J * K + y * K + z]) - dot;
                        
						for (int n = 0; n < r; n++) {
							double bc = b_vals[n] * c_vals[n];
							double ac = a_vals[n] * c_vals[n];
							double ab = a_vals[n] * b_vals[n];
                            
							atomicAdd(&dA[x * r + n], coe_x * lr * (error * bc - reg * a_vals[n]));
							atomicAdd(&dB[y * r + n], coe_y * lr * (error * ac - reg * b_vals[n]));
							atomicAdd(&dC[z * r + n], coe_z * lr * (error * ab - reg * c_vals[n]));
						}
					}
				}
				
			}
			else{
				
				int tx = threadIdx.x;	
				while (tx < block_num) {
					int x = d_block[bx + len].x[tx];
					int y = d_block[bx + len].y[tx];
					int z = d_block[bx + len].z[tx];
					if (x < 0 || x >= I || y < 0 || y >= J || z < 0 || z >= K) {
						printf("Invalid index: %d,%d,%d\n", x,y,z);
						continue; // 跳过无效索引
					}
					
					if (dX[x * J * K + y * K + z] > 0) {
						double a_vals[100];  // 假设r <= 32
						double b_vals[100];
						double c_vals[100];
						double dot = 0;
                        for (int n = 0; n < r; n++) {
                            a_vals[n] = __ldg(&dA[x * r + n]);
                            b_vals[n] = __ldg(&dB[y * r + n]);
                            c_vals[n] = __ldg(&dC[z * r + n]);
							dot = fma(a_vals[n], b_vals[n] * c_vals[n], dot);
                        }
						double error = __ldg(&dX[x * J * K + y * K + z]) - dot;
                        
						for (int n = 0; n < r; n++) {
							double bc = b_vals[n] * c_vals[n];
							double ac = a_vals[n] * c_vals[n];
							double ab = a_vals[n] * b_vals[n];
                            
							atomicAdd(&dA[x * r + n], coe_x * lr * (error * bc - reg * a_vals[n]));
							atomicAdd(&dB[y * r + n], coe_y * lr * (error * ac - reg * b_vals[n]));
							atomicAdd(&dC[z * r + n], coe_z * lr * (error * ab - reg * c_vals[n]));
						}
					}
					tx += thread_size; // 下一跳
				}
			}
		}
		len = len + d_parallel_num_gpu[i];
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
void clear(queue<double>& q) {
	queue<double> empty;
	swap(empty, q);
}


void sgd_train(double* data_initial,
	double* data_sampling,
	double* a,
	double* b,
	double* c,
	int num_parallel,
	int max_parallel,
	LF_node* LF,
	int* num_LF,
	int nnz,
	double rate,
	int *num_bs,
	b_node* bs,
	int num_scheduled_bs){

	ofstream ofs_time_1, ofs_error;
	string address("/root/fan/BTC-GPU/output/AbileneTM_Tensor_");
	//string address("/root/fan/output/NYC_LF/");
	string str;
	stringstream ss;
	ss << rate;
	ss >> str;
	str += "/";
	address += str;
	string add_error, add_time;
	add_error += address;
	add_error += "error.txt";
	add_time += address;
	add_time += "time.txt";
	
	clear(num1);
	int numDevices = 0;
	cudaGetDeviceCount(&numDevices);
	printf("Found %d GPU devices.\n", numDevices);
	

	

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

	//cudaMemcpy(dX, data_sampling, nbytesX, cudaMemcpyHostToDevice);
	cudaMemcpy(dX, data_sampling, nbytesX, cudaMemcpyHostToDevice);
	cudaMemcpy(dA, a, nbytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, b, nbytesB, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, c, nbytesC, cudaMemcpyHostToDevice);
	//cudaDeviceSynchronize();


	ofs_error.open(add_error, ios::out | ios::trunc);
	ofs_time_1.open(add_time, ios::out | ios::trunc);
	ofs_error << setprecision(10);
	ofs_error << "epoch train_RMSE test_RMSE train_MAE test_MAE train_ER test_ER train_count test_count" << endl;

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

			double td2 = get_time_ms();
			tensor_LF << <1, thread_size >> > (d_LF, d_num_LF, dA, dB, dC, num_parallel, thread_size, I, J, K, r, lr, reg);
			cudaDeviceSynchronize();
			td2 = get_time_ms() - td2;
			ofs_time_1 << td2 << endl;

			cudaMemcpy(a, dA, nbytesA, cudaMemcpyDeviceToHost);
			cudaMemcpy(b, dB, nbytesB, cudaMemcpyDeviceToHost);
			cudaMemcpy(c, dC, nbytesC, cudaMemcpyDeviceToHost);
			
			ErrorMetrics train_metrics = compute_metrics(data_sampling, data_initial, a, b, c, true);
			ErrorMetrics test_metrics = compute_metrics(data_sampling, data_initial, a, b, c, false);

			write_metrics_line(ofs_error, epoch, train_metrics, test_metrics);
			print_epoch_rmse("", numDevices, epoch, train_metrics, test_metrics);

			num1.push(train_metrics.rmse);
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
	
	if(flag_preproccess == 1 && num_scheduled_bs > 0){
		int* d_num_bs;
		int parallel_count = num_parallel > 1 ? num_parallel - 1 : 1;
		int nbytesnumbs = parallel_count * (sizeof(int));
		cudaMalloc(&d_num_bs, nbytesnumbs);
		cudaMemcpy(d_num_bs, num_bs, nbytesnumbs, cudaMemcpyHostToDevice);

		int nbytesd_bs = num_scheduled_bs * (sizeof(b_node));
		b_node* d_bs_tmp = (b_node*)malloc(nbytesd_bs);
		memcpy(d_bs_tmp, bs, nbytesd_bs);

		b_node* d_bs;
		cudaMalloc(&d_bs, nbytesd_bs);
		for (int i = 0; i < num_scheduled_bs; ++i) {
			int len = bs[i].block_num;
			int *d_x, *d_y, *d_z;
			cudaMalloc(&d_x, len * sizeof(int));
			cudaMalloc(&d_y, len * sizeof(int));
			cudaMalloc(&d_z, len * sizeof(int));
			
			cudaMemcpy(d_x, bs[i].x, len * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, bs[i].y, len * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_z, bs[i].z, len * sizeof(int), cudaMemcpyHostToDevice);
			d_bs_tmp[i].x = d_x;
			d_bs_tmp[i].y = d_y;
			d_bs_tmp[i].z = d_z;
		}

		cudaMemcpy(d_bs, d_bs_tmp, nbytesd_bs, cudaMemcpyHostToDevice);

		cout<< max_parallel << "   " << num_scheduled_bs <<endl;
		//if(numDevices == 1){
			for (int epoch = 0; epoch < epochs; epoch++) {
				double td4 = get_time_ms();
				GPUs<< <max_parallel, thread_size>>>(dX, d_bs, d_num_bs, dA, dB, dC, num_parallel, I, J, K, r, lr, reg, thread_size);
				cudaDeviceSynchronize();
				td4 = get_time_ms() - td4;
				ofs_time_1 << td4 << endl;
				
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) 
					printf("Kernel error: %s\n", cudaGetErrorString(err));
					
				cudaMemcpy(a , dA, nbytesA, cudaMemcpyDeviceToHost);
				cudaMemcpy(b, dB, nbytesB, cudaMemcpyDeviceToHost);
				cudaMemcpy(c, dC, nbytesC, cudaMemcpyDeviceToHost);

				ErrorMetrics train_metrics = compute_metrics(data_sampling, data_initial, a, b, c, true);
				ErrorMetrics test_metrics = compute_metrics(data_sampling, data_initial, a, b, c, false);
				
				write_metrics_line(ofs_error, epoch, train_metrics, test_metrics);
				print_epoch_rmse("GPU:", numDevices, epoch, train_metrics, test_metrics);
				
				num1.push(train_metrics.rmse);
				if (num1.size() > 2) {
					num1.pop();
				}
		
				if (num1.size() >= 2) {
					if (num1.front() - num1.back() < 0.00000001) {
						break;
					}
				}
			}
		for (int i = 0; i < num_scheduled_bs; ++i) {
			cudaFree(d_bs_tmp[i].x);
			cudaFree(d_bs_tmp[i].y);
			cudaFree(d_bs_tmp[i].z);
		}
		cudaFree(d_bs);
		cudaFree(d_num_bs);
		free(d_bs_tmp);
	}
	cudaMemcpy(a, dA, nbytesA, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, dB, nbytesB, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, dC, nbytesC, cudaMemcpyDeviceToHost);
	ErrorMetrics final_train_metrics = compute_metrics(data_sampling, data_initial, a, b, c, true);
	ErrorMetrics final_test_metrics = compute_metrics(data_sampling, data_initial, a, b, c, false);
	print_final_metrics(final_train_metrics, final_test_metrics);
	
	cudaFree(dX);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

}
