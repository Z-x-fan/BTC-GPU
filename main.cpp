#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
//#include <Windows.h>
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
#include <algorithm>
#include <string>
#include "tensor_SGD.h"

using namespace std;

string join_path(const string& dir, const string& name) {
	if (dir.empty()) {
		return name;
	}
	char last = dir[dir.size() - 1];
	if (last == '/' || last == '\\') {
		return dir + name;
	}
	return dir + "/" + name;
}

bool open_first_existing(ifstream& infile, const vector<string>& candidates, string& opened_path) {
	for (size_t i = 0; i < candidates.size(); i++) {
		infile.open(candidates[i].c_str());
		if (infile.is_open()) {
			opened_path = candidates[i];
			return true;
		}
		infile.clear();
	}
	return false;
}

bool parse_entry_line(const string& line, T_node& entry, int coord_base) {
	if (line.empty() || line[0] == '#' || line[0] == '%') {
		return false;
	}

	stringstream ss(line);
	double value = 0;
	if (!(ss >> entry.x >> entry.y >> entry.z >> value)) {
		return false;
	}

	entry.x -= coord_base;
	entry.y -= coord_base;
	entry.z -= coord_base;
	entry.rate = value;
	return true;
}

vector<T_node> read_entries(
	const vector<string>& candidates,
	const string& label,
	int coord_base) {

	ifstream infile;
	string opened_path;
	if (!open_first_existing(infile, candidates, opened_path)) {
		cerr << "Failed to open " << label << " file. Tried:" << endl;
		for (size_t i = 0; i < candidates.size(); i++) {
			cerr << "  " << candidates[i] << endl;
		}
		exit(1);
	}

	vector<T_node> entries;
	string line;
	long long line_no = 0;
	while (getline(infile, line)) {
		line_no++;
		T_node entry;
		if (!parse_entry_line(line, entry, coord_base)) {
			continue;
		}
		if (entry.x < 0 || entry.x >= I || entry.y < 0 || entry.y >= J || entry.z < 0 || entry.z >= K) {
			cerr << "Invalid " << label << " coordinate at " << opened_path << ":" << line_no
				<< " -> (" << entry.x << "," << entry.y << "," << entry.z << ") after base conversion" << endl;
			exit(1);
		}
		entries.push_back(entry);
	}

	cout << "Loaded " << entries.size() << " " << label << " entries from " << opened_path << endl;
	return entries;
}

void print_usage(const char* program) {
	cerr << "Usage: " << program << " <dataset> <I> <J> <K> [data_dir] [coord_base] [output_dir] [options]" << endl;
	cerr << "Example: " << program << " delicious-3d 532924 17262471 2480308 /fan/data 1 /fan/output --epochs 1000 --rank 4 --lr 0.001 --reg 0.05 --block 1024 --parallel-sequence-size 8 --gpus 4" << endl;
	cerr << "Reads <dataset>.train and <dataset>.test from data_dir." << endl;
	cerr << "Options:" << endl;
	cerr << "  --epochs N                    Training epochs. Default: " << epochs << endl;
	cerr << "  --rank N, --r N               Factor rank r. Default: " << r << endl;
	cerr << "  --lr VALUE                    Learning rate. Default: " << lr << endl;
	cerr << "  --reg VALUE                   Regularization. Default: " << reg << endl;
	cerr << "  --block N, --block-s N, --block_s N" << endl;
	cerr << "                                Block size. Default: " << block_s << endl;
	cerr << "  --thread-size N, --thread_size N" << endl;
	cerr << "                                CUDA threads per block. Default: " << thread_size << endl;
	cerr << "  --parallel-sequence-size N, --parallel_sequence_size N" << endl;
	cerr << "                                Fixed blocks per parallel sequence; 0 means auto. Default: " << parallel_sequence_size << endl;
	cerr << "  --gpus N, --gpu-count N, --gpu_count N" << endl;
	cerr << "                                GPUs to use; 0 means all visible GPUs. Default: " << requested_gpu_count << endl;
}

bool read_option_value(const string& arg, int& index, int argc, char** argv, string& option, string& value) {
	size_t equal_pos = arg.find('=');
	if (equal_pos != string::npos) {
		option = arg.substr(0, equal_pos);
		value = arg.substr(equal_pos + 1);
		return true;
	}
	option = arg;
	if (index + 1 >= argc) {
		cerr << "Missing value for option " << option << endl;
		return false;
	}
	index++;
	value = argv[index];
	return true;
}

bool apply_option(const string& option, const string& value) {
	if (option == "--epochs") {
		epochs = atoi(value.c_str());
	}
	else if (option == "--rank" || option == "--r") {
		r = atoi(value.c_str());
	}
	else if (option == "--lr") {
		lr = atof(value.c_str());
	}
	else if (option == "--reg") {
		reg = atof(value.c_str());
	}
	else if (option == "--block" || option == "--block-s" || option == "--block_s") {
		block_s = atoi(value.c_str());
	}
	else if (option == "--thread-size" || option == "--thread_size") {
		thread_size = atoi(value.c_str());
	}
	else if (option == "--parallel-sequence-size" || option == "--parallel_sequence_size") {
		parallel_sequence_size = atoi(value.c_str());
	}
	else if (option == "--gpus" || option == "--gpu-count" || option == "--gpu_count") {
		requested_gpu_count = atoi(value.c_str());
	}
	else {
		cerr << "Unknown option: " << option << endl;
		return false;
	}
	return true;
}

bool validate_runtime_config() {
	if (epochs <= 0) {
		cerr << "epochs must be positive." << endl;
		return false;
	}
	if (r <= 0 || r > 100) {
		cerr << "rank r must be in [1, 100]. The current CUDA kernel uses fixed local arrays of length 100." << endl;
		return false;
	}
	if (lr <= 0) {
		cerr << "lr must be positive." << endl;
		return false;
	}
	if (reg < 0) {
		cerr << "reg must be non-negative." << endl;
		return false;
	}
	if (block_s <= 0) {
		cerr << "block must be positive." << endl;
		return false;
	}
	if (thread_size <= 0 || thread_size > 1024) {
		cerr << "thread-size must be in [1, 1024]." << endl;
		return false;
	}
	if (parallel_sequence_size < 0) {
		cerr << "parallel-sequence-size must be >= 0." << endl;
		return false;
	}
	if (requested_gpu_count < 0) {
		cerr << "gpus must be >= 0. Use 0 for all visible GPUs." << endl;
		return false;
	}
	return true;
}

void print_runtime_config() {
	cout << "Runtime config:"
		<< " epochs=" << epochs
		<< " r=" << r
		<< " lr=" << lr
		<< " reg=" << reg
		<< " block_s=" << block_s
		<< " thread_size=" << thread_size
		<< " parallel_sequence_size=" << parallel_sequence_size
		<< " requested_gpus=" << requested_gpu_count
		<< endl;
}

int main(int argc, char** argv) {
	if (argc > 1 && (string(argv[1]) == "--help" || string(argv[1]) == "-h")) {
		print_usage(argv[0]);
		return 0;
	}
	if (argc < 5) {
		print_usage(argv[0]);
		return 1;
	}

	dataset_name = argv[1];
	I = atoi(argv[2]);
	J = atoi(argv[3]);
	K = atoi(argv[4]);

	vector<string> positional;
	for (int index = 5; index < argc; index++) {
		string arg = argv[index];
		if (arg == "--help" || arg == "-h") {
			print_usage(argv[0]);
			return 0;
		}
		if (arg.size() > 2 && arg.substr(0, 2) == "--") {
			string option;
			string value;
			if (!read_option_value(arg, index, argc, argv, option, value) || !apply_option(option, value)) {
				print_usage(argv[0]);
				return 1;
			}
		}
		else {
			positional.push_back(arg);
		}
	}

	string data_dir = positional.size() > 0 ? positional[0] : "/fan/data";
	int coord_base = positional.size() > 1 ? atoi(positional[1].c_str()) : 1;
	output_root = positional.size() > 2 ? positional[2] : "output";

	if (I <= 0 || J <= 0 || K <= 0) {
		cerr << "I, J, and K must be positive." << endl;
		return 1;
	}
	if (coord_base != 0 && coord_base != 1) {
		cerr << "coord_base must be 0 or 1." << endl;
		return 1;
	}
	if (positional.size() > 3) {
		cerr << "Too many positional arguments." << endl;
		print_usage(argv[0]);
		return 1;
	}
	if (!validate_runtime_config()) {
		return 1;
	}

	int cuda_device_count = 0;
	cudaError_t device_status = cudaGetDeviceCount(&cuda_device_count);
	if (device_status != cudaSuccess) {
		cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(device_status) << endl;
		return 1;
	}
	cout << "CUDA visible GPU count=" << cuda_device_count << endl;
	if (cuda_device_count <= 0) {
		cerr << "No visible CUDA GPU devices." << endl;
		return 1;
	}
	print_runtime_config();

	vector<string> train_candidates;
	train_candidates.push_back(join_path(data_dir, dataset_name + ".train"));
	train_candidates.push_back(join_path(data_dir, dataset_name + "_tr.tns"));
	train_candidates.push_back(join_path(data_dir, dataset_name + ".tr.tns"));

	vector<string> test_candidates;
	test_candidates.push_back(join_path(data_dir, dataset_name + ".test"));
	test_candidates.push_back(join_path(data_dir, dataset_name + "_te.tns"));
	test_candidates.push_back(join_path(data_dir, dataset_name + ".te.tns"));

	vector<T_node> train_entries = read_entries(train_candidates, "train", coord_base);
	vector<T_node> test_entries = read_entries(test_candidates, "test", coord_base);
	if (train_entries.empty()) {
		cerr << "Train file has no entries." << endl;
		return 1;
	}
	if (train_entries.size() > 1) {
		std::mt19937 shuffle_rng(42);
		std::shuffle(train_entries.begin(), train_entries.end(), shuffle_rng);
		cout << "Shuffled train entries with seed=42 before block construction." << endl;
	}

	long long total_entries = (long long)train_entries.size() + (long long)test_entries.size();
	train_rate = total_entries > 0 ? (double)train_entries.size() / (double)total_entries : 0.0;
	cout << "Train rate=" << train_rate << " test rate=" << (1.0 - train_rate) << endl;

	tensor_SGD(train_entries, test_entries);

	return 0;
}
