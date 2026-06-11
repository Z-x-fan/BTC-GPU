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
#include <algorithm>
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
	long long count;
};

double undefined_metric() {
	return std::numeric_limits<double>::quiet_NaN();
}

bool check_cuda(cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		cerr << operation << " failed: " << cudaGetErrorString(status) << endl;
		return false;
	}
	return true;
}

const int METRIC_THREADS = 256;
const int METRIC_MAX_BLOCKS = 4096;

__global__ void metrics_kernel(
	const T_node* entries,
	int entry_count,
	const double* dA,
	const double* dB,
	const double* dC,
	int r,
	double* partial_squared_error,
	double* partial_abs_error,
	double* partial_ref_squared) {

	extern __shared__ double shared[];
	double* shared_squared_error = shared;
	double* shared_abs_error = shared + blockDim.x;
	double* shared_ref_squared = shared + 2 * blockDim.x;

	int tx = threadIdx.x;
	double local_squared_error = 0;
	double local_abs_error = 0;
	double local_ref_squared = 0;

	for (int idx = blockIdx.x * blockDim.x + tx; idx < entry_count; idx += blockDim.x * gridDim.x) {
		int i = entries[idx].x;
		int j = entries[idx].y;
		int m = entries[idx].z;
		double dot = 0;
		for (int n = 0; n < r; n++) {
			dot = fma(__ldg(&dA[i * r + n]), __ldg(&dB[j * r + n]) * __ldg(&dC[m * r + n]), dot);
		}

		double ref = __ldg(&entries[idx].rate);
		double error = ref - dot;
		local_abs_error += fabs(error);
		local_squared_error += error * error;
		local_ref_squared += ref * ref;
	}

	shared_squared_error[tx] = local_squared_error;
	shared_abs_error[tx] = local_abs_error;
	shared_ref_squared[tx] = local_ref_squared;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tx < stride) {
			shared_squared_error[tx] += shared_squared_error[tx + stride];
			shared_abs_error[tx] += shared_abs_error[tx + stride];
			shared_ref_squared[tx] += shared_ref_squared[tx + stride];
		}
		__syncthreads();
	}

	if (tx == 0) {
		partial_squared_error[blockIdx.x] = shared_squared_error[0];
		partial_abs_error[blockIdx.x] = shared_abs_error[0];
		partial_ref_squared[blockIdx.x] = shared_ref_squared[0];
	}
}

int metric_block_count(int entry_count) {
	if (entry_count <= 0) {
		return 1;
	}
	int blocks = (entry_count + METRIC_THREADS - 1) / METRIC_THREADS;
	if (blocks < 1) {
		blocks = 1;
	}
	if (blocks > METRIC_MAX_BLOCKS) {
		blocks = METRIC_MAX_BLOCKS;
	}
	return blocks;
}

ErrorMetrics compute_metrics_gpu(
	const T_node* d_entries,
	int entry_count,
	double* dA,
	double* dB,
	double* dC,
	double* d_partial_squared_error,
	double* d_partial_abs_error,
	double* d_partial_ref_squared,
	int metric_blocks) {

	ErrorMetrics metrics;
	metrics.count = entry_count;
	if (entry_count == 0) {
		metrics.rmse = undefined_metric();
		metrics.mae = undefined_metric();
		metrics.er = undefined_metric();
		return metrics;
	}

	size_t shared_bytes = METRIC_THREADS * 3 * sizeof(double);
	metrics_kernel << <metric_blocks, METRIC_THREADS, shared_bytes >> > (
		d_entries,
		entry_count,
		dA,
		dB,
		dC,
		r,
		d_partial_squared_error,
		d_partial_abs_error,
		d_partial_ref_squared);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Metrics kernel error: %s\n", cudaGetErrorString(err));
	}

	vector<double> partial_squared_error(metric_blocks);
	vector<double> partial_abs_error(metric_blocks);
	vector<double> partial_ref_squared(metric_blocks);
	size_t partial_bytes = (size_t)metric_blocks * sizeof(double);
	if (!check_cuda(cudaMemcpy(partial_squared_error.data(), d_partial_squared_error, partial_bytes, cudaMemcpyDeviceToHost),
			"cudaMemcpy metric squared error") ||
		!check_cuda(cudaMemcpy(partial_abs_error.data(), d_partial_abs_error, partial_bytes, cudaMemcpyDeviceToHost),
			"cudaMemcpy metric absolute error") ||
		!check_cuda(cudaMemcpy(partial_ref_squared.data(), d_partial_ref_squared, partial_bytes, cudaMemcpyDeviceToHost),
			"cudaMemcpy metric reference squared")) {
		metrics.rmse = undefined_metric();
		metrics.mae = undefined_metric();
		metrics.er = undefined_metric();
		return metrics;
	}

	double total_squared_error = 0;
	double total_abs_error = 0;
	double total_ref_squared = 0;
	for (int i = 0; i < metric_blocks; i++) {
		total_squared_error += partial_squared_error[i];
		total_abs_error += partial_abs_error[i];
		total_ref_squared += partial_ref_squared[i];
	}

	metrics.rmse = sqrt(total_squared_error / entry_count);
	metrics.mae = total_abs_error / entry_count;
	metrics.er = total_ref_squared > 0 ? total_squared_error / total_ref_squared : undefined_metric();
	return metrics;
}

void write_metrics_line(ofstream& ofs_error, int epoch, const ErrorMetrics& test_metrics) {
	ofs_error << epoch << ' '
		<< test_metrics.rmse << ' '
		<< test_metrics.mae << ' '
		<< test_metrics.er << ' '
		<< test_metrics.count << endl;
}

void print_epoch_rmse(const char* prefix, int numDevices, int epoch, const ErrorMetrics& test_metrics) {
	if (prefix != NULL && prefix[0] != '\0') {
		cout << prefix << numDevices << ' ';
	}
	cout << "Epoch:" << epoch
		<< " test_RMSE=" << test_metrics.rmse << '\n';
}

void print_final_metrics(const ErrorMetrics& test_metrics) {
	cout << "Final test RMSE=" << test_metrics.rmse
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

__global__ void GPUs(b_node* d_block, int* d_parallel_num_gpu, double* dA, double* dB, double* dC, int num_parallel, int I, int J, int K, int r, double lr, double reg, int thread_size) {
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
					double target = __ldg(&d_block[bx + len].rate[tx]);
					
					double a_vals[100];
					double b_vals[100];
					double c_vals[100];
					double dot = 0;
                    for (int n = 0; n < r; n++) {
                        a_vals[n] = __ldg(&dA[x * r + n]);
                        b_vals[n] = __ldg(&dB[y * r + n]);
                        c_vals[n] = __ldg(&dC[z * r + n]);
						dot = fma(a_vals[n], b_vals[n] * c_vals[n], dot);
                    }
					double error = target - dot;
                    
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
			else{
				
				int tx = threadIdx.x;	
				while (tx < block_num) {
					int x = d_block[bx + len].x[tx];
					int y = d_block[bx + len].y[tx];
					int z = d_block[bx + len].z[tx];
					if (x < 0 || x >= I || y < 0 || y >= J || z < 0 || z >= K) {
						printf("Invalid index: %d,%d,%d\n", x,y,z);
						tx += thread_size;
						continue; // 跳过无效索引
					}
					double target = __ldg(&d_block[bx + len].rate[tx]);
					
					double a_vals[100];
					double b_vals[100];
					double c_vals[100];
					double dot = 0;
                    for (int n = 0; n < r; n++) {
                        a_vals[n] = __ldg(&dA[x * r + n]);
                        b_vals[n] = __ldg(&dB[y * r + n]);
                        c_vals[n] = __ldg(&dC[z * r + n]);
						dot = fma(a_vals[n], b_vals[n] * c_vals[n], dot);
                    }
					double error = target - dot;
                    
					for (int n = 0; n < r; n++) {
						double bc = b_vals[n] * c_vals[n];
						double ac = a_vals[n] * c_vals[n];
						double ab = a_vals[n] * b_vals[n];
                        
						atomicAdd(&dA[x * r + n], coe_x * lr * (error * bc - reg * a_vals[n]));
						atomicAdd(&dB[y * r + n], coe_y * lr * (error * ac - reg * b_vals[n]));
						atomicAdd(&dC[z * r + n], coe_z * lr * (error * ab - reg * c_vals[n]));
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

struct DeviceTrainingState {
	int device_id;
	int local_max_parallel;
	int local_scheduled_bs;
	vector<int> h_num_bs;
	vector<b_node> h_bs;
	vector<b_node> h_device_bs;
	int* d_num_bs;
	b_node* d_bs;
	double* dA;
	double* dB;
	double* dC;

	DeviceTrainingState()
		: device_id(0),
		  local_max_parallel(0),
		  local_scheduled_bs(0),
		  d_num_bs(NULL),
		  d_bs(NULL),
		  dA(NULL),
		  dB(NULL),
		  dC(NULL) {
	}
};

int choose_active_gpu_count(int visible_devices) {
	if (visible_devices <= 0) {
		return 0;
	}
	if (requested_gpu_count == 0) {
		return visible_devices;
	}
	return std::min(requested_gpu_count, visible_devices);
}

void print_cuda_inventory(int visible_devices) {
	cout << "CUDA visible GPU count=" << visible_devices << endl;
	for (int device = 0; device < visible_devices; device++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
			cout << "  GPU[" << device << "] name=\"" << prop.name << "\""
				<< " sm_count=" << prop.multiProcessorCount
				<< " memory_MB=" << (unsigned long long)(prop.totalGlobalMem / (1024ULL * 1024ULL))
				<< endl;
		}
	}
}

bool set_device_checked(int device_id, const char* operation) {
	return check_cuda(cudaSetDevice(device_id), operation);
}

double* select_factor_ptr(DeviceTrainingState& state, int factor_id) {
	if (factor_id == 0) {
		return state.dA;
	}
	if (factor_id == 1) {
		return state.dB;
	}
	return state.dC;
}

void free_device_training_state(DeviceTrainingState& state) {
	if (cudaSetDevice(state.device_id) != cudaSuccess) {
		return;
	}
	for (size_t i = 0; i < state.h_device_bs.size(); i++) {
		if (state.h_device_bs[i].x != NULL) {
			cudaFree(state.h_device_bs[i].x);
		}
		if (state.h_device_bs[i].y != NULL) {
			cudaFree(state.h_device_bs[i].y);
		}
		if (state.h_device_bs[i].z != NULL) {
			cudaFree(state.h_device_bs[i].z);
		}
		if (state.h_device_bs[i].rate != NULL) {
			cudaFree(state.h_device_bs[i].rate);
		}
	}
	if (state.d_bs != NULL) {
		cudaFree(state.d_bs);
	}
	if (state.d_num_bs != NULL) {
		cudaFree(state.d_num_bs);
	}
	if (state.dA != NULL) {
		cudaFree(state.dA);
	}
	if (state.dB != NULL) {
		cudaFree(state.dB);
	}
	if (state.dC != NULL) {
		cudaFree(state.dC);
	}
	state.h_device_bs.clear();
	state.d_bs = NULL;
	state.d_num_bs = NULL;
	state.dA = NULL;
	state.dB = NULL;
	state.dC = NULL;
}

void free_all_device_training_states(vector<DeviceTrainingState>& states) {
	for (size_t i = 0; i < states.size(); i++) {
		free_device_training_state(states[i]);
	}
}

bool allocate_device_factors(
	DeviceTrainingState& state,
	const double* a,
	const double* b,
	const double* c,
	size_t nbytesA,
	size_t nbytesB,
	size_t nbytesC) {

	if (!set_device_checked(state.device_id, "cudaSetDevice allocate factors")) {
		return false;
	}
	if (!check_cuda(cudaMalloc((void**)&state.dA, nbytesA), "cudaMalloc dA") ||
		!check_cuda(cudaMalloc((void**)&state.dB, nbytesB), "cudaMalloc dB") ||
		!check_cuda(cudaMalloc((void**)&state.dC, nbytesC), "cudaMalloc dC")) {
		return false;
	}
	if (!check_cuda(cudaMemcpy(state.dA, a, nbytesA, cudaMemcpyHostToDevice), "cudaMemcpy initial dA") ||
		!check_cuda(cudaMemcpy(state.dB, b, nbytesB, cudaMemcpyHostToDevice), "cudaMemcpy initial dB") ||
		!check_cuda(cudaMemcpy(state.dC, c, nbytesC, cudaMemcpyHostToDevice), "cudaMemcpy initial dC")) {
		return false;
	}
	return true;
}

bool upload_device_schedule(DeviceTrainingState& state) {
	if (!set_device_checked(state.device_id, "cudaSetDevice upload schedule")) {
		return false;
	}

	size_t nbytes_num_bs = state.h_num_bs.size() * sizeof(int);
	if (!check_cuda(cudaMalloc((void**)&state.d_num_bs, nbytes_num_bs), "cudaMalloc d_num_bs") ||
		!check_cuda(cudaMemcpy(state.d_num_bs, state.h_num_bs.data(), nbytes_num_bs, cudaMemcpyHostToDevice), "cudaMemcpy d_num_bs")) {
		return false;
	}

	state.local_scheduled_bs = (int)state.h_bs.size();
	if (state.local_scheduled_bs == 0) {
		return true;
	}

	size_t nbytes_bs = (size_t)state.local_scheduled_bs * sizeof(b_node);
	state.h_device_bs.resize(state.local_scheduled_bs);
	for (int i = 0; i < state.local_scheduled_bs; i++) {
		state.h_device_bs[i] = state.h_bs[i];
		state.h_device_bs[i].x = NULL;
		state.h_device_bs[i].y = NULL;
		state.h_device_bs[i].z = NULL;
		state.h_device_bs[i].rate = NULL;
	}
	if (!check_cuda(cudaMalloc((void**)&state.d_bs, nbytes_bs), "cudaMalloc d_bs")) {
		return false;
	}

	for (int i = 0; i < state.local_scheduled_bs; i++) {
		int len = state.h_bs[i].block_num;
		size_t len_int_bytes = (size_t)len * sizeof(int);
		size_t len_double_bytes = (size_t)len * sizeof(double);
		int* d_x = NULL;
		int* d_y = NULL;
		int* d_z = NULL;
		double* d_rate = NULL;

		if (!check_cuda(cudaMalloc((void**)&d_x, len_int_bytes), "cudaMalloc block x") ||
			!check_cuda(cudaMalloc((void**)&d_y, len_int_bytes), "cudaMalloc block y") ||
			!check_cuda(cudaMalloc((void**)&d_z, len_int_bytes), "cudaMalloc block z") ||
			!check_cuda(cudaMalloc((void**)&d_rate, len_double_bytes), "cudaMalloc block rate")) {
			if (d_x != NULL) {
				cudaFree(d_x);
			}
			if (d_y != NULL) {
				cudaFree(d_y);
			}
			if (d_z != NULL) {
				cudaFree(d_z);
			}
			if (d_rate != NULL) {
				cudaFree(d_rate);
			}
			return false;
		}
		if (!check_cuda(cudaMemcpy(d_x, state.h_bs[i].x, len_int_bytes, cudaMemcpyHostToDevice), "cudaMemcpy block x") ||
			!check_cuda(cudaMemcpy(d_y, state.h_bs[i].y, len_int_bytes, cudaMemcpyHostToDevice), "cudaMemcpy block y") ||
			!check_cuda(cudaMemcpy(d_z, state.h_bs[i].z, len_int_bytes, cudaMemcpyHostToDevice), "cudaMemcpy block z") ||
			!check_cuda(cudaMemcpy(d_rate, state.h_bs[i].rate, len_double_bytes, cudaMemcpyHostToDevice), "cudaMemcpy block rate")) {
			cudaFree(d_x);
			cudaFree(d_y);
			cudaFree(d_z);
			cudaFree(d_rate);
			return false;
		}

		state.h_device_bs[i].x = d_x;
		state.h_device_bs[i].y = d_y;
		state.h_device_bs[i].z = d_z;
		state.h_device_bs[i].rate = d_rate;
	}

	return check_cuda(cudaMemcpy(state.d_bs, state.h_device_bs.data(), nbytes_bs, cudaMemcpyHostToDevice), "cudaMemcpy d_bs");
}

bool prepare_multi_gpu_states(
	int active_gpu_count,
	int parallel_count,
	const int* num_bs,
	b_node* bs,
	const double* a,
	const double* b,
	const double* c,
	size_t nbytesA,
	size_t nbytesB,
	size_t nbytesC,
	vector<DeviceTrainingState>& states) {

	states.assign(active_gpu_count, DeviceTrainingState());
	for (int gpu = 0; gpu < active_gpu_count; gpu++) {
		states[gpu].device_id = gpu;
		states[gpu].h_num_bs.assign(parallel_count, 0);
	}

	int offset = 0;
	for (int sequence_id = 0; sequence_id < parallel_count; sequence_id++) {
		int sequence_blocks = num_bs[sequence_id];
		for (int block_id = 0; block_id < sequence_blocks; block_id++) {
			int gpu = block_id % active_gpu_count;
			states[gpu].h_num_bs[sequence_id]++;
			states[gpu].h_bs.push_back(bs[offset + block_id]);
		}
		offset += sequence_blocks;
	}

	for (int gpu = 0; gpu < active_gpu_count; gpu++) {
		states[gpu].local_scheduled_bs = (int)states[gpu].h_bs.size();
		for (size_t i = 0; i < states[gpu].h_num_bs.size(); i++) {
			states[gpu].local_max_parallel = std::max(states[gpu].local_max_parallel, states[gpu].h_num_bs[i]);
		}
		cout << "GPU[" << gpu << "] scheduled_blocks=" << states[gpu].local_scheduled_bs
			<< " max_parallel=" << states[gpu].local_max_parallel << endl;
		if (!allocate_device_factors(states[gpu], a, b, c, nbytesA, nbytesB, nbytesC) ||
			!upload_device_schedule(states[gpu])) {
			free_all_device_training_states(states);
			return false;
		}
	}
	return true;
}

bool copy_factor_to_all_devices(
	vector<DeviceTrainingState>& states,
	int factor_id,
	const double* host_factor,
	size_t nbytes,
	const char* label) {

	for (size_t i = 0; i < states.size(); i++) {
		if (!set_device_checked(states[i].device_id, "cudaSetDevice copy factor")) {
			return false;
		}
		string op = string("cudaMemcpy sync ") + label;
		if (!check_cuda(cudaMemcpy(select_factor_ptr(states[i], factor_id), host_factor, nbytes, cudaMemcpyHostToDevice), op.c_str())) {
			return false;
		}
	}
	return true;
}

bool combine_factor_from_devices(
	vector<DeviceTrainingState>& states,
	int factor_id,
	double* host_factor,
	size_t elements,
	size_t nbytes,
	vector<double>& scratch,
	vector<double>& combined,
	const char* label) {

	memcpy(combined.data(), host_factor, nbytes);
	for (size_t state_id = 0; state_id < states.size(); state_id++) {
		if (!set_device_checked(states[state_id].device_id, "cudaSetDevice combine factor")) {
			return false;
		}
		string op = string("cudaMemcpy gather ") + label;
		if (!check_cuda(cudaMemcpy(scratch.data(), select_factor_ptr(states[state_id], factor_id), nbytes, cudaMemcpyDeviceToHost), op.c_str())) {
			return false;
		}
		for (size_t i = 0; i < elements; i++) {
			combined[i] += scratch[i] - host_factor[i];
		}
	}
	memcpy(host_factor, combined.data(), nbytes);
	return copy_factor_to_all_devices(states, factor_id, host_factor, nbytes, label);
}

bool combine_all_factors_from_devices(
	vector<DeviceTrainingState>& states,
	double* a,
	double* b,
	double* c,
	size_t elemsA,
	size_t elemsB,
	size_t elemsC,
	size_t nbytesA,
	size_t nbytesB,
	size_t nbytesC,
	vector<double>& scratch,
	vector<double>& combined) {

	return combine_factor_from_devices(states, 0, a, elemsA, nbytesA, scratch, combined, "A") &&
		combine_factor_from_devices(states, 1, b, elemsB, nbytesB, scratch, combined, "B") &&
		combine_factor_from_devices(states, 2, c, elemsC, nbytesC, scratch, combined, "C");
}

bool run_preprocessed_multi_gpu_training(
	const T_node* test_entries,
	int test_nnz,
	double* a,
	double* b,
	double* c,
	int num_parallel,
	int max_parallel,
	int* num_bs,
	b_node* bs,
	int num_scheduled_bs,
	int active_gpu_count,
	int visible_gpu_count,
	ofstream& ofs_time_1,
	ofstream& ofs_kernel_time,
	ofstream& ofs_sync_time,
	ofstream& ofs_error) {

	int parallel_count = num_parallel > 1 ? num_parallel - 1 : 1;
	size_t elemsA = (size_t)I * (size_t)r;
	size_t elemsB = (size_t)J * (size_t)r;
	size_t elemsC = (size_t)K * (size_t)r;
	size_t nbytesA = elemsA * sizeof(double);
	size_t nbytesB = elemsB * sizeof(double);
	size_t nbytesC = elemsC * sizeof(double);
	size_t max_factor_elements = std::max(elemsA, std::max(elemsB, elemsC));

	cout << "multi_gpu_mode=epoch_sync visible_gpus=" << visible_gpu_count
		<< " requested_gpus=" << requested_gpu_count
		<< " active_gpus=" << active_gpu_count
		<< " max_parallel=" << max_parallel
		<< " scheduled_blocks=" << num_scheduled_bs << endl;

	vector<DeviceTrainingState> states;
	if (!prepare_multi_gpu_states(active_gpu_count, parallel_count, num_bs, bs, a, b, c, nbytesA, nbytesB, nbytesC, states)) {
		return false;
	}

	vector<double> scratch(max_factor_elements);
	vector<double> combined(max_factor_elements);

	T_node* d_test_entries = NULL;
	double* d_metric_squared_error = NULL;
	double* d_metric_abs_error = NULL;
	double* d_metric_ref_squared = NULL;
	int metric_blocks = metric_block_count(test_nnz);
	if (!set_device_checked(states[0].device_id, "cudaSetDevice metrics")) {
		free_all_device_training_states(states);
		return false;
	}
	if (test_nnz > 0) {
		size_t test_entry_bytes = (size_t)test_nnz * sizeof(T_node);
		size_t metric_bytes = (size_t)metric_blocks * sizeof(double);
		if (!check_cuda(cudaMalloc((void**)&d_test_entries, test_entry_bytes), "cudaMalloc d_test_entries") ||
			!check_cuda(cudaMemcpy(d_test_entries, test_entries, test_entry_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_test_entries") ||
			!check_cuda(cudaMalloc((void**)&d_metric_squared_error, metric_bytes), "cudaMalloc metric squared error") ||
			!check_cuda(cudaMalloc((void**)&d_metric_abs_error, metric_bytes), "cudaMalloc metric absolute error") ||
			!check_cuda(cudaMalloc((void**)&d_metric_ref_squared, metric_bytes), "cudaMalloc metric reference squared")) {
			cudaFree(d_test_entries);
			cudaFree(d_metric_squared_error);
			cudaFree(d_metric_abs_error);
			cudaFree(d_metric_ref_squared);
			free_all_device_training_states(states);
			return false;
		}
	}

	for (int epoch = 0; epoch < epochs; epoch++) {
		double kernel_time = get_time_ms();
		bool kernel_ok = true;

		for (int gpu = 0; gpu < active_gpu_count; gpu++) {
			if (states[gpu].local_scheduled_bs == 0) {
				continue;
			}
			if (!set_device_checked(states[gpu].device_id, "cudaSetDevice launch multi GPU")) {
				kernel_ok = false;
				break;
			}
			GPUs << <states[gpu].local_max_parallel, thread_size >> > (
				states[gpu].d_bs,
				states[gpu].d_num_bs,
				states[gpu].dA,
				states[gpu].dB,
				states[gpu].dC,
				num_parallel,
				I,
				J,
				K,
				r,
				lr,
				reg,
				thread_size);
			cudaError_t launch_err = cudaGetLastError();
			if (launch_err != cudaSuccess) {
				printf("GPU[%d] kernel launch error: %s\n", gpu, cudaGetErrorString(launch_err));
				kernel_ok = false;
				break;
			}
		}

		for (int gpu = 0; gpu < active_gpu_count; gpu++) {
			if (states[gpu].local_scheduled_bs == 0) {
				continue;
			}
			if (!set_device_checked(states[gpu].device_id, "cudaSetDevice synchronize multi GPU")) {
				kernel_ok = false;
				break;
			}
			cudaError_t sync_err = cudaDeviceSynchronize();
			if (sync_err != cudaSuccess) {
				printf("GPU[%d] kernel synchronize error: %s\n", gpu, cudaGetErrorString(sync_err));
				kernel_ok = false;
				break;
			}
		}

		kernel_time = get_time_ms() - kernel_time;
		double sync_time = get_time_ms();
		if (!kernel_ok ||
			!combine_all_factors_from_devices(states, a, b, c, elemsA, elemsB, elemsC, nbytesA, nbytesB, nbytesC, scratch, combined)) {
			cudaFree(d_test_entries);
			cudaFree(d_metric_squared_error);
			cudaFree(d_metric_abs_error);
			cudaFree(d_metric_ref_squared);
			free_all_device_training_states(states);
			return false;
		}

		sync_time = get_time_ms() - sync_time;
		ofs_kernel_time << kernel_time << endl;
		ofs_sync_time << sync_time << endl;
		ofs_time_1 << (kernel_time + sync_time) << endl;

		if (!set_device_checked(states[0].device_id, "cudaSetDevice metrics epoch")) {
			cudaFree(d_test_entries);
			cudaFree(d_metric_squared_error);
			cudaFree(d_metric_abs_error);
			cudaFree(d_metric_ref_squared);
			free_all_device_training_states(states);
			return false;
		}
		ErrorMetrics test_metrics = compute_metrics_gpu(
			d_test_entries,
			test_nnz,
			states[0].dA,
			states[0].dB,
			states[0].dC,
			d_metric_squared_error,
			d_metric_abs_error,
			d_metric_ref_squared,
			metric_blocks);

		write_metrics_line(ofs_error, epoch, test_metrics);
		print_epoch_rmse("GPU:", active_gpu_count, epoch, test_metrics);

		num1.push(test_metrics.rmse);
		if (num1.size() > 2) {
			num1.pop();
		}
		if (num1.size() >= 2) {
			if (num1.front() - num1.back() < 0.00000001) {
				break;
			}
		}
	}

	if (set_device_checked(states[0].device_id, "cudaSetDevice final metrics")) {
		ErrorMetrics final_test_metrics = compute_metrics_gpu(
			d_test_entries,
			test_nnz,
			states[0].dA,
			states[0].dB,
			states[0].dC,
			d_metric_squared_error,
			d_metric_abs_error,
			d_metric_ref_squared,
			metric_blocks);
		print_final_metrics(final_test_metrics);
	}

	cudaFree(d_test_entries);
	cudaFree(d_metric_squared_error);
	cudaFree(d_metric_abs_error);
	cudaFree(d_metric_ref_squared);
	free_all_device_training_states(states);
	return true;
}


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
	int num_scheduled_bs){

	ofstream ofs_time_1, ofs_kernel_time, ofs_sync_time, ofs_error;
	string address = output_root;
	if (!address.empty() && address[address.size() - 1] != '/' && address[address.size() - 1] != '\\') {
		address += "/";
	}
	address += dataset_name + "_";
	string str;
	stringstream ss;
	ss << rate;
	ss >> str;
	str += "/";
	address += str;
#ifdef _WIN32
	string make_dir_command = string("if not exist \"") + address + "\" mkdir \"" + address + "\"";
#else
	string make_dir_command = string("mkdir -p \"") + address + "\"";
#endif
	system(make_dir_command.c_str());
	string add_error, add_time, add_kernel_time, add_sync_time;
	add_error += address;
	add_error += "error.txt";
	add_time += address;
	add_time += "time.txt";
	add_kernel_time += address;
	add_kernel_time += "kernel_time.txt";
	add_sync_time += address;
	add_sync_time += "sync_time.txt";
	
	clear(num1);
	int numDevices = 0;
	cudaError_t device_count_status = cudaGetDeviceCount(&numDevices);
	if (device_count_status != cudaSuccess) {
		cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(device_count_status) << endl;
		return;
	}
	print_cuda_inventory(numDevices);
	if (numDevices <= 0) {
		cerr << "No visible CUDA GPU devices." << endl;
		return;
	}
	int active_gpu_count = choose_active_gpu_count(numDevices);
	if (active_gpu_count <= 0) {
		cerr << "No GPU selected for training." << endl;
		return;
	}
	if (requested_gpu_count > numDevices) {
		cout << "requested_gpus=" << requested_gpu_count
			<< " exceeds visible GPU count; using " << active_gpu_count << " GPU(s)." << endl;
	}
	cout << "Using " << active_gpu_count << " GPU(s) for training." << endl;

	ofs_error.open(add_error, ios::out | ios::trunc);
	ofs_time_1.open(add_time, ios::out | ios::trunc);
	ofs_kernel_time.open(add_kernel_time, ios::out | ios::trunc);
	ofs_sync_time.open(add_sync_time, ios::out | ios::trunc);
	ofs_error << setprecision(10);
	ofs_error << "epoch test_RMSE test_MAE test_ER test_count" << endl;

	double* dA = NULL;
	double* dB = NULL;
	double* dC = NULL;

	size_t nbytesA = (size_t)I * (size_t)r * sizeof(double);
	size_t nbytesB = (size_t)J * (size_t)r * sizeof(double);
	size_t nbytesC = (size_t)K * (size_t)r * sizeof(double);

	cout << "factor_matrix_bytes A=" << nbytesA
		<< " B=" << nbytesB
		<< " C=" << nbytesC << endl;

	if (active_gpu_count > 1 && flag_lockfree == 0 && flag_preproccess == 1 && num_scheduled_bs > 0) {
		bool multi_gpu_ok = run_preprocessed_multi_gpu_training(
			test_entries,
			test_nnz,
			a,
			b,
			c,
			num_parallel,
			max_parallel,
			num_bs,
			bs,
			num_scheduled_bs,
			active_gpu_count,
			numDevices,
			ofs_time_1,
			ofs_kernel_time,
			ofs_sync_time,
			ofs_error);
		if (!multi_gpu_ok) {
			cerr << "Multi-GPU training failed." << endl;
		}
		return;
	}

	if (active_gpu_count > 1) {
		cout << "Multi-GPU path requires preprocessed block training with lockfree disabled; falling back to GPU[0]." << endl;
	}
	if (!set_device_checked(0, "cudaSetDevice single GPU")) {
		return;
	}

	if (!check_cuda(cudaMalloc((void**)&dA, nbytesA), "cudaMalloc dA") ||
		!check_cuda(cudaMalloc((void**)&dB, nbytesB), "cudaMalloc dB") ||
		!check_cuda(cudaMalloc((void**)&dC, nbytesC), "cudaMalloc dC")) {
		cudaFree(dA);
		cudaFree(dB);
		cudaFree(dC);
		return;
	}

	if (!check_cuda(cudaMemcpy(dA, a, nbytesA, cudaMemcpyHostToDevice), "cudaMemcpy dA") ||
		!check_cuda(cudaMemcpy(dB, b, nbytesB, cudaMemcpyHostToDevice), "cudaMemcpy dB") ||
		!check_cuda(cudaMemcpy(dC, c, nbytesC, cudaMemcpyHostToDevice), "cudaMemcpy dC")) {
		cudaFree(dA);
		cudaFree(dB);
		cudaFree(dC);
		return;
	}
	//cudaDeviceSynchronize();

	T_node* d_test_entries = NULL;
	double* d_metric_squared_error = NULL;
	double* d_metric_abs_error = NULL;
	double* d_metric_ref_squared = NULL;
	int metric_blocks = metric_block_count(test_nnz);
	if (test_nnz > 0) {
		size_t test_entry_bytes = (size_t)test_nnz * sizeof(T_node);
		size_t metric_bytes = (size_t)metric_blocks * sizeof(double);
		if (!check_cuda(cudaMalloc((void**)&d_test_entries, test_entry_bytes), "cudaMalloc d_test_entries") ||
			!check_cuda(cudaMemcpy(d_test_entries, test_entries, test_entry_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_test_entries") ||
			!check_cuda(cudaMalloc((void**)&d_metric_squared_error, metric_bytes), "cudaMalloc metric squared error") ||
			!check_cuda(cudaMalloc((void**)&d_metric_abs_error, metric_bytes), "cudaMalloc metric absolute error") ||
			!check_cuda(cudaMalloc((void**)&d_metric_ref_squared, metric_bytes), "cudaMalloc metric reference squared")) {
			cudaFree(d_test_entries);
			cudaFree(d_metric_squared_error);
			cudaFree(d_metric_abs_error);
			cudaFree(d_metric_ref_squared);
			cudaFree(dA);
			cudaFree(dB);
			cudaFree(dC);
			return;
		}
	}

	if (flag_lockfree == 1) {
		LF_node* d_LF;
		int* d_num_LF;
		size_t nbytesLF = (size_t)train_nnz * sizeof(LF_node);
		size_t nbytesnumLF = (size_t)num_parallel * sizeof(int);
		if (!check_cuda(cudaMalloc((void**)&d_LF, nbytesLF), "cudaMalloc d_LF") ||
			!check_cuda(cudaMalloc((void**)&d_num_LF, nbytesnumLF), "cudaMalloc d_num_LF") ||
			!check_cuda(cudaMemcpy(d_LF, LF, nbytesLF, cudaMemcpyHostToDevice), "cudaMemcpy d_LF") ||
			!check_cuda(cudaMemcpy(d_num_LF, num_LF, nbytesnumLF, cudaMemcpyHostToDevice), "cudaMemcpy d_num_LF")) {
			cudaFree(d_LF);
			cudaFree(d_num_LF);
			cudaFree(d_test_entries);
			cudaFree(d_metric_squared_error);
			cudaFree(d_metric_abs_error);
			cudaFree(d_metric_ref_squared);
			cudaFree(dA);
			cudaFree(dB);
			cudaFree(dC);
			return;
		}

		// Warm up the same kernel path once, then restore factors so epoch 0 is unchanged.
		tensor_LF << <1, thread_size >> > (d_LF, d_num_LF, dA, dB, dC, num_parallel, thread_size, I, J, K, r, lr, reg);
		cudaDeviceSynchronize();
		cudaError_t warmup_err_lf = cudaGetLastError();
		if (warmup_err_lf != cudaSuccess) {
			printf("Warm-up kernel error: %s\n", cudaGetErrorString(warmup_err_lf));
		}
		check_cuda(cudaMemcpy(dA, a, nbytesA, cudaMemcpyHostToDevice), "cudaMemcpy warmup restore dA");
		check_cuda(cudaMemcpy(dB, b, nbytesB, cudaMemcpyHostToDevice), "cudaMemcpy warmup restore dB");
		check_cuda(cudaMemcpy(dC, c, nbytesC, cudaMemcpyHostToDevice), "cudaMemcpy warmup restore dC");

		for (int epoch = 0; epoch < epochs; epoch++) {

			double td2 = get_time_ms();
			tensor_LF << <1, thread_size >> > (d_LF, d_num_LF, dA, dB, dC, num_parallel, thread_size, I, J, K, r, lr, reg);
			cudaDeviceSynchronize();
			td2 = get_time_ms() - td2;
			ofs_kernel_time << td2 << endl;
			ofs_sync_time << 0 << endl;
			ofs_time_1 << td2 << endl;

			ErrorMetrics test_metrics = compute_metrics_gpu(
				d_test_entries,
				test_nnz,
				dA,
				dB,
				dC,
				d_metric_squared_error,
				d_metric_abs_error,
				d_metric_ref_squared,
				metric_blocks);

			write_metrics_line(ofs_error, epoch, test_metrics);
			print_epoch_rmse("", numDevices, epoch, test_metrics);

			num1.push(test_metrics.rmse);
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
		size_t nbytesnumbs = (size_t)parallel_count * sizeof(int);
		if (!check_cuda(cudaMalloc(&d_num_bs, nbytesnumbs), "cudaMalloc d_num_bs") ||
			!check_cuda(cudaMemcpy(d_num_bs, num_bs, nbytesnumbs, cudaMemcpyHostToDevice), "cudaMemcpy d_num_bs")) {
			cudaFree(d_num_bs);
			cudaFree(d_test_entries);
			cudaFree(d_metric_squared_error);
			cudaFree(d_metric_abs_error);
			cudaFree(d_metric_ref_squared);
			cudaFree(dA);
			cudaFree(dB);
			cudaFree(dC);
			return;
		}

		size_t nbytesd_bs = (size_t)num_scheduled_bs * sizeof(b_node);
		b_node* d_bs_tmp = (b_node*)malloc(nbytesd_bs);
		memcpy(d_bs_tmp, bs, nbytesd_bs);

		b_node* d_bs;
		if (!check_cuda(cudaMalloc(&d_bs, nbytesd_bs), "cudaMalloc d_bs")) {
			cudaFree(d_num_bs);
			free(d_bs_tmp);
			cudaFree(d_test_entries);
			cudaFree(d_metric_squared_error);
			cudaFree(d_metric_abs_error);
			cudaFree(d_metric_ref_squared);
			cudaFree(dA);
			cudaFree(dB);
			cudaFree(dC);
			return;
		}
		for (int i = 0; i < num_scheduled_bs; ++i) {
			int len = bs[i].block_num;
			int *d_x, *d_y, *d_z;
			double* d_rate;
			size_t len_int_bytes = (size_t)len * sizeof(int);
			size_t len_double_bytes = (size_t)len * sizeof(double);
			if (!check_cuda(cudaMalloc(&d_x, len_int_bytes), "cudaMalloc block x") ||
				!check_cuda(cudaMalloc(&d_y, len_int_bytes), "cudaMalloc block y") ||
				!check_cuda(cudaMalloc(&d_z, len_int_bytes), "cudaMalloc block z") ||
				!check_cuda(cudaMalloc(&d_rate, len_double_bytes), "cudaMalloc block rate")) {
				cudaFree(d_bs);
				cudaFree(d_num_bs);
				free(d_bs_tmp);
				cudaFree(d_test_entries);
				cudaFree(d_metric_squared_error);
				cudaFree(d_metric_abs_error);
				cudaFree(d_metric_ref_squared);
				cudaFree(dA);
				cudaFree(dB);
				cudaFree(dC);
				return;
			}
			
			check_cuda(cudaMemcpy(d_x, bs[i].x, len_int_bytes, cudaMemcpyHostToDevice), "cudaMemcpy block x");
			check_cuda(cudaMemcpy(d_y, bs[i].y, len_int_bytes, cudaMemcpyHostToDevice), "cudaMemcpy block y");
			check_cuda(cudaMemcpy(d_z, bs[i].z, len_int_bytes, cudaMemcpyHostToDevice), "cudaMemcpy block z");
			check_cuda(cudaMemcpy(d_rate, bs[i].rate, len_double_bytes, cudaMemcpyHostToDevice), "cudaMemcpy block rate");
			d_bs_tmp[i].x = d_x;
			d_bs_tmp[i].y = d_y;
			d_bs_tmp[i].z = d_z;
			d_bs_tmp[i].rate = d_rate;
		}

		check_cuda(cudaMemcpy(d_bs, d_bs_tmp, nbytesd_bs, cudaMemcpyHostToDevice), "cudaMemcpy d_bs");

		cout<< max_parallel << "   " << num_scheduled_bs <<endl;
		// Warm up the same kernel path once, then restore factors so epoch 0 is unchanged.
		GPUs<< <max_parallel, thread_size>>>(d_bs, d_num_bs, dA, dB, dC, num_parallel, I, J, K, r, lr, reg, thread_size);
		cudaDeviceSynchronize();
		cudaError_t warmup_err = cudaGetLastError();
		if (warmup_err != cudaSuccess) {
			printf("Warm-up kernel error: %s\n", cudaGetErrorString(warmup_err));
		}
		check_cuda(cudaMemcpy(dA, a, nbytesA, cudaMemcpyHostToDevice), "cudaMemcpy warmup restore dA");
		check_cuda(cudaMemcpy(dB, b, nbytesB, cudaMemcpyHostToDevice), "cudaMemcpy warmup restore dB");
		check_cuda(cudaMemcpy(dC, c, nbytesC, cudaMemcpyHostToDevice), "cudaMemcpy warmup restore dC");

		//if(numDevices == 1){
			for (int epoch = 0; epoch < epochs; epoch++) {
				double td4 = get_time_ms();
				GPUs<< <max_parallel, thread_size>>>(d_bs, d_num_bs, dA, dB, dC, num_parallel, I, J, K, r, lr, reg, thread_size);
				cudaDeviceSynchronize();
				td4 = get_time_ms() - td4;
				ofs_kernel_time << td4 << endl;
				ofs_sync_time << 0 << endl;
				ofs_time_1 << td4 << endl;
				
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) 
					printf("Kernel error: %s\n", cudaGetErrorString(err));
					
				ErrorMetrics test_metrics = compute_metrics_gpu(
					d_test_entries,
					test_nnz,
					dA,
					dB,
					dC,
					d_metric_squared_error,
					d_metric_abs_error,
					d_metric_ref_squared,
					metric_blocks);
				
				write_metrics_line(ofs_error, epoch, test_metrics);
				print_epoch_rmse("GPU:", numDevices, epoch, test_metrics);
				
				num1.push(test_metrics.rmse);
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
			cudaFree(d_bs_tmp[i].rate);
		}
		cudaFree(d_bs);
		cudaFree(d_num_bs);
		free(d_bs_tmp);
	}
	ErrorMetrics final_test_metrics = compute_metrics_gpu(
		d_test_entries,
		test_nnz,
		dA,
		dB,
		dC,
		d_metric_squared_error,
		d_metric_abs_error,
		d_metric_ref_squared,
		metric_blocks);
	print_final_metrics(final_test_metrics);

	check_cuda(cudaMemcpy(a, dA, nbytesA, cudaMemcpyDeviceToHost), "cudaMemcpy final a");
	check_cuda(cudaMemcpy(b, dB, nbytesB, cudaMemcpyDeviceToHost), "cudaMemcpy final b");
	check_cuda(cudaMemcpy(c, dC, nbytesC, cudaMemcpyDeviceToHost), "cudaMemcpy final c");
	
	if (d_test_entries != NULL) {
		cudaFree(d_test_entries);
	}
	if (d_metric_squared_error != NULL) {
		cudaFree(d_metric_squared_error);
	}
	if (d_metric_abs_error != NULL) {
		cudaFree(d_metric_abs_error);
	}
	if (d_metric_ref_squared != NULL) {
		cudaFree(d_metric_ref_squared);
	}
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

}
