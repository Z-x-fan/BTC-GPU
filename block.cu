#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <set>
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
#include <cstdlib>
#include "tensor_SGD.h"
#include "block.h"
#include "sgd_Kernel.h"
#include <chrono>
#include <algorithm>
#ifndef _WIN32
#include <sys/time.h>
#endif
using namespace std;


void initial_t(T_node* t_block, const T_node* train_entries, int nnz) {
	for (int i = 0; i < nnz; i++) {
		t_block[i] = train_entries[i];
	}
}


vector<bool> hx(int end, int start, int level, int num) {
	vector<bool> binary;

	float n_1 = (end - start + 1) / pow(2, level - 1);
	int n_2 = floor(num / n_1);

	bitset<100> b1(n_2);

	for (int i = level - 1; i >= 0; i--) {
		binary.push_back(b1[i]);
	}
	return binary;
}


int compare(vector<bool> id, vector<bool> x, int level) {
	for (int block_id = 1; block_id < level; block_id++) {
		if (x[block_id] ^ id[block_id]) {   
			return 1;   
		}
	}
	return 0;

}


BTnode* search_block(vector<bool> x, vector<bool> y, vector<bool> z, BTnode* BT) {
	while (BT->lChild != NULL) {
		int tmp_h = BT->h % 3;
		if (tmp_h == 1) {		
			if (compare(BT->lChild->x, x, BT->lChild->x.size()) == 0)
				BT = BT->lChild;
			else
				BT = BT->rChild;
		}
		else if (tmp_h == 2) {			
			if (compare(BT->lChild->y, y, BT->lChild->y.size()) == 0)
				BT = BT->lChild;
			else
				BT = BT->rChild;
		}
		else {
			if (compare(BT->lChild->z, z, BT->lChild->z.size()) == 0)
				BT = BT->lChild;
			else
				BT = BT->rChild;
		}
	}

	return BT;
}

void Create_Btree(BTnode* BT_block,int old_id, int new_id) {
	BT_block->block = -1;
	int tmp_h = BT_block->h % 3;

	if (tmp_h == 1) {
		BTnode* s_l = new BTnode();
		BTnode* s_r = new BTnode();
		s_l->h = BT_block->h + 1;
		s_r->h = BT_block->h + 1;
		s_l->x = BT_block->x;
		s_l->y = BT_block->y;
		s_l->z = BT_block->z;
		s_l->x.push_back(0);
		s_r->x = BT_block->x;
		s_r->y = BT_block->y;
		s_r->z = BT_block->z;
		s_r->x.push_back(1);
		s_l->block = old_id;
		s_r->block = new_id;
		s_l->lChild = s_l->rChild = s_r->lChild = s_r->rChild = NULL;
		BT_block->lChild = s_l;
		BT_block->rChild = s_r;
	}
	else if (tmp_h == 2) {
		BTnode* s_l = new BTnode();
		BTnode* s_r = new BTnode();
		s_l->h = BT_block->h + 1;
		s_r->h = BT_block->h + 1;
		s_l->x = BT_block->x;
		s_l->y = BT_block->y;
		s_l->z = BT_block->z;
		s_l->y.push_back(0);
		s_r->x = BT_block->x;
		s_r->y = BT_block->y;
		s_r->z = BT_block->z;
		s_r->y.push_back(1);
		s_l->block = old_id;
		s_r->block = new_id;
		s_l->lChild = s_l->rChild = s_r->lChild = s_r->rChild = NULL;
		BT_block->lChild = s_l;
		BT_block->rChild = s_r;
	}
	else if (tmp_h == 0) {
		BTnode* s_l = new BTnode();
		BTnode* s_r = new BTnode();
		s_l->h = BT_block->h + 1;
		s_r->h = BT_block->h + 1;
		s_l->x = BT_block->x;
		s_l->y = BT_block->y;
		s_l->z = BT_block->z;
		s_l->z.push_back(0);
		s_r->x = BT_block->x;
		s_r->y = BT_block->y;
		s_r->z = BT_block->z;
		s_r->z.push_back(1);
		s_l->block = old_id;
		s_r->block = new_id;
		s_l->lChild = s_l->rChild = s_r->lChild = s_r->rChild = NULL;
		BT_block->lChild = s_l;
		BT_block->rChild = s_r;
	}

}

int tensor_block(BS_node* Block, T_node* t, int nnz, int I, int J, int K, BTnode* BT, BTnode* BT_head) {
	int Block_num = 1;//Block����
	BT->h = 1;
	BT->block = 0;
	BT->x.push_back(0);
	BT->y.push_back(0);
	BT->z.push_back(0);
	BT->lChild = BT->rChild = NULL;	
	int g_x, g_y, g_z;
	g_x = 1;
	g_y = 1;
	g_z = 1;
	int start = 0;
	int end_x = I;
	int end_y = J;
	int end_z = K;
	for (int i = 0; i < nnz; i++) {		
		vector<bool> x = hx(end_x, start, g_x, t[i].x);
		vector<bool> y = hx(end_y, start, g_y, t[i].y);
		vector<bool> z = hx(end_z, start, g_z, t[i].z);

		BT = BT_head;
		BT = search_block(x, y, z, BT);
		int id = BT->block;

		if (Block[id].block_num < block_s) {
			Block[id].t[Block[id].block_num] = t[i];
			Block[id].block_num++;
		}
		else {
			int flag = 0;
			while (flag == 0) {

				if (Block[id].level_x == Block[id].level_y && Block[id].level_y == Block[id].level_z) {
					Block[id].level_x++;
					int max = 0;
					max = Block[id].level_x > g_x ? Block[id].level_x : g_x;
					g_x = max;

					Create_Btree(BT, id, Block_num);
					//Block[block_id_num].x_id = ID_change(Block[id].x_id, 1, Block[id].level_x);
					//Block[id].x_id = ID_change(Block[id].x_id, 0, Block[id].level_x);
					Block[Block_num].block_num = 0;
					Block[Block_num].level_x = Block[id].level_x;
					Block[Block_num].level_y = Block[id].level_y;
					Block[Block_num].level_z = Block[id].level_z;
					Block[Block_num].x_id = Block[id].x_id;
					Block[Block_num].y_id = Block[id].y_id;
					Block[Block_num].z_id = Block[id].z_id;
					Block[Block_num].id = Block_num;
					Block[Block_num].t = new T_node[block_s];
					Block[Block_num].x_id.push_back(1);
					Block[id].x_id.push_back(0);

					//�������·���
					int num_1 = 0;
					int num_2 = 0;
					int* tx = new int[block_s];
					T_node* t1 = new T_node[block_s];
					T_node* t2 = new T_node[block_s];
					for (int j = 0; j < Block[id].block_num; j++) {
						vector<bool> xt = hx(end_x, start, g_x, Block[id].t[j].x);
						if (xt.back() == 0) {
							t1[num_1] = Block[id].t[j];
							num_1++;
						}
						else {
							t2[num_2] = Block[id].t[j];
							num_2++;
						}					
					}
					vector<bool> xt = hx(end_x, start, g_x, t[i].x);
					if (num_1 < block_s && num_2 < block_s) {
						flag = 1;
						if (xt.back() == 0) {
							t1[num_1] = t[i];
							num_1++;
						}
						else {
							t2[num_2] = t[i];
							num_2++;
						}
					}
					else {
						flag = 0;
						if (xt.back() == 0) {
							id = Block[id].id;
							BT = BT->lChild;
						}							
						else {
							id = Block[Block_num].id;
							BT = BT->rChild;
						}							
					}
					

					Block[id].t = t1;
					Block[id].block_num = num_1;
					Block[Block_num].t = t2;
					Block[Block_num].block_num = num_2;
					Block_num++;
				}
				else if (Block[id].level_y < Block[id].level_x) {
					Block[id].level_y++;
					int max = 0;
					max = Block[id].level_y > g_y ? Block[id].level_y : g_y;
					g_y = max;


					Create_Btree(BT, id, Block_num);
					//Block[block_id_num].y_id = ID_change(Block[id].y_id, 1, Block[id].level_y);
					//Block[id].y_id = ID_change(Block[id].y_id, 0, Block[id].level_y);
					Block[Block_num].block_num = 0;
					Block[Block_num].level_x = Block[id].level_x;
					Block[Block_num].level_y = Block[id].level_y;
					Block[Block_num].level_z = Block[id].level_z;
					Block[Block_num].x_id = Block[id].x_id;
					Block[Block_num].y_id = Block[id].y_id;
					Block[Block_num].z_id = Block[id].z_id;
					Block[Block_num].id = Block_num;
					Block[Block_num].t = new T_node[block_s];
					Block[Block_num].y_id.push_back(1);
					Block[id].y_id.push_back(0);


					int* ty = new int[block_s];
					int num_1 = 0;
					int num_2 = 0;
					T_node* t1 = new T_node[block_s];
					T_node* t2 = new T_node[block_s];
					for (int j = 0; j < Block[id].block_num; j++) {
						vector<bool> yt = hx(end_y, start, g_y, Block[id].t[j].y);
						if (yt.back() == 0) {
							t1[num_1] = Block[id].t[j];
							num_1++;
						}
						else {
							t2[num_2] = Block[id].t[j];
							num_2++;
						}						                  
					}
					vector<bool> yt = hx(end_y, start, g_y, t[i].y);
					if (num_1 < block_s && num_2 < block_s) {
						flag = 1;
						if (yt.back() == 0) {
							t1[num_1] = t[i];
							num_1++;
						}
						else {
							t2[num_2] = t[i];
							num_2++;
						}
					}
					else {
						flag = 0;
						if (yt.back() == 0) {
							id = Block[id].id;
							BT = BT->lChild;
						}
						else {
							id = Block[Block_num].id;
							BT = BT->rChild;
						}
					}
					Block[id].t = t1;
					Block[id].block_num = num_1;
					Block[Block_num].t = t2;
					Block[Block_num].block_num = num_2;
					Block_num++;
				}
				else if (Block[id].level_z < Block[id].level_y) {

					Block[id].level_z++;
					int max = 0;
					max = Block[id].level_z > g_z ? Block[id].level_z : g_z;
					g_z = max;

					Create_Btree(BT, id, Block_num);
					//Block[block_id_num].z_id = ID_change(Block[id].z_id, 1, Block[id].level_z);
					//Block[id].z_id = ID_change(Block[id].z_id, 0, Block[id].level_z);
					Block[Block_num].block_num = 0;
					Block[Block_num].level_x = Block[id].level_x;
					Block[Block_num].level_y = Block[id].level_y;
					Block[Block_num].level_z = Block[id].level_z;
					Block[Block_num].x_id = Block[id].x_id;
					Block[Block_num].y_id = Block[id].y_id;
					Block[Block_num].z_id = Block[id].z_id;
					Block[Block_num].id = Block_num;
					Block[Block_num].t = new T_node[block_s];
					Block[Block_num].z_id.push_back(1);
					Block[id].z_id.push_back(0);


					int* tz = new int[block_s];
					int num_1 = 0;
					int num_2 = 0;
					T_node* t1 = new T_node[block_s];
					T_node* t2 = new T_node[block_s];					
					for (int j = 0; j < Block[id].block_num; j++) {
						vector<bool> zt = hx(end_z, start, g_z, Block[id].t[j].z);
						if (zt.back() == 0) {
							t1[num_1] = Block[id].t[j];
							num_1++;
						}
						else {
							t2[num_2] = Block[id].t[j];
							num_2++;
						}
					}
					vector<bool> zt = hx(end_z, start, g_z, t[i].z);
					if (num_1 < block_s && num_2 < block_s) {
						flag = 1;
						if (zt.back() == 0) {
							t1[num_1] = t[i];
							num_1++;
						}
						else {
							t2[num_2] = t[i];
							num_2++;
						}
					}
					else {
						flag = 0;
						if (zt.back() == 0) {
							id = Block[id].id;
							BT = BT->lChild;
						}
						else {
							id = Block[Block_num].id;
							BT = BT->rChild;
						}
					}
					Block[id].t = t1;
					Block[id].block_num = num_1;
					Block[Block_num].t = t2;
					Block[Block_num].block_num = num_2;
					Block_num++;
				}
			}
		}

		vector<bool>().swap(x);
		vector<bool>().swap(y);
		vector<bool>().swap(z);
	}

	return Block_num;
}

void Free_list(Parallel* head) {

	Parallel* freeNode;
	while (NULL != head) {
		freeNode = head;
		head = head->next;
		delete freeNode;
	}
}

int find_conflict_x(vector<bool> a, vector<bool> b, int level_a, int level_b) {
	int level = level_a < level_b ? level_a : level_b;

	if (level == 1)
		return 1;//�г�ͻ
	else {
		int block_x;
		for (block_x = 1; block_x < level; block_x++) {
			if (a[block_x] != b[block_x]) {
				return 0;//����ͻ
				break;
			}
		}
		if (block_x == level)
			return 1;
	}
	return -1;
}

int find_parent(vector<int>& parent, int index) {
	while (parent[index] != index) {
		parent[index] = parent[parent[index]];
		index = parent[index];
	}
	return index;
}

void union_parent(vector<int>& parent, int a, int b) {
	int root_a = find_parent(parent, a);
	int root_b = find_parent(parent, b);
	if (root_a != root_b) {
		parent[root_b] = root_a;
	}
}

int axis_conflict(BS_node* Block, int left, int right, int axis) {
	if (axis == 0) {
		return find_conflict_x(Block[left].x_id, Block[right].x_id, Block[left].level_x, Block[right].level_x);
	}
	if (axis == 1) {
		return find_conflict_x(Block[left].y_id, Block[right].y_id, Block[left].level_y, Block[right].level_y);
	}
	return find_conflict_x(Block[left].z_id, Block[right].z_id, Block[left].level_z, Block[right].level_z);
}

void set_axis_weight(Node_conflict* B_conf, int block_id, int axis, double weight) {
	if (axis == 0) {
		B_conf[block_id].coe_x = weight;
	}
	else if (axis == 1) {
		B_conf[block_id].coe_y = weight;
	}
	else {
		B_conf[block_id].coe_z = weight;
	}
}

void update_conflict_weights(Node_conflict* B_conf, const vector<int>& block_ids, BS_node* Block) {
	int n = block_ids.size();
	if (n == 0) {
		return;
	}

	for (int i = 0; i < n; i++) {
		int block_id = block_ids[i];
		B_conf[block_id].id = block_id;
		B_conf[block_id].coe_x = 1.0;
		B_conf[block_id].coe_y = 1.0;
		B_conf[block_id].coe_z = 1.0;
	}

	for (int axis = 0; axis < 3; axis++) {
		vector<int> parent(n);
		for (int i = 0; i < n; i++) {
			parent[i] = i;
		}

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				if (axis_conflict(Block, block_ids[i], block_ids[j], axis) == 1) {
					union_parent(parent, i, j);
				}
			}
		}

		vector<int> conflict_sum(n, 0);
		for (int i = 0; i < n; i++) {
			int root = find_parent(parent, i);
			int block_count = Block[block_ids[i]].block_num > 0 ? Block[block_ids[i]].block_num : 1;
			conflict_sum[root] += block_count;
		}

		for (int i = 0; i < n; i++) {
			int root = find_parent(parent, i);
			int block_count = Block[block_ids[i]].block_num > 0 ? Block[block_ids[i]].block_num : 1;
			//double weight = conflict_sum[root] > 0 ? (double)block_count / conflict_sum[root] : 1.0;
			double weight = conflict_sum[root] > 0 ? (double)block_count / conflict_sum[root] : 1.0;
			set_axis_weight(B_conf, block_ids[i], axis, weight);
		}
	}
}

void update_conflict_weights(Node_conflict* B_conf, Parallel* P, BS_node* Block) {
	vector<int> block_ids;
	list<int>::iterator it = P->L.begin();
	for (; it != P->L.end(); it++) {
		block_ids.push_back(*it);
	}
	update_conflict_weights(B_conf, block_ids, Block);
}

int sequence_conflict_score(BS_node* Block, const vector<int>& sequence, int candidate) {
	int score = 0;
	for (int i = 0; i < sequence.size(); i++) {
		for (int axis = 0; axis < 3; axis++) {
			if (axis_conflict(Block, sequence[i], candidate, axis) == 1) {
				score++;
			}
		}
	}
	return score;
}

bool sequence_contains(const vector<int>& sequence, int block_id) {
	for (int i = 0; i < sequence.size(); i++) {
		if (sequence[i] == block_id) {
			return true;
		}
	}
	return false;
}

int select_min_conflict_block(BS_node* Block, int block_num, const vector<int>& sequence) {
	int best_id = -1;
	int best_score = 0;
	int best_work = 0;

	for (int i = 0; i < block_num; i++) {
		if (Block[i].block_num == 0 || sequence_contains(sequence, i)) {
			continue;
		}

		int score = sequence_conflict_score(Block, sequence, i);
		int work = Block[i].block_num;
		if (best_id == -1 || score < best_score || (score == best_score && work < best_work)) {
			best_id = i;
			best_score = score;
			best_work = work;
		}
	}

	return best_id;
}

int fallback_blocks_per_sm(int major) {
	if (major >= 5) {
		return 32;
	}
	if (major >= 3) {
		return 16;
	}
	return 8;
}

int gpu_block_target(int observed_max_parallel, int block_num) {
	int target = observed_max_parallel > 0 ? observed_max_parallel : 1;
	int device = 0;
	int device_count = 0;

	if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
		if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= device_count) {
			device = 0;
		}

		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
			int blocks_per_sm = prop.maxBlocksPerMultiProcessor > 0
				? prop.maxBlocksPerMultiProcessor
				: fallback_blocks_per_sm(prop.major);
			int resident_blocks = prop.multiProcessorCount * blocks_per_sm;
			if (resident_blocks > 0) {
				target = target < resident_blocks ? target : resident_blocks;
			}
		}
	}

	if (target < 1) {
		target = 1;
	}
	if (block_num > 0 && target > block_num) {
		target = block_num;
	}
	return target;
}

int configured_gpu_block_target(int observed_max_parallel, int block_num) {
	int target = parallel_sequence_size > 0
		? parallel_sequence_size
		: gpu_block_target(observed_max_parallel, block_num);
	if (target < 1) {
		target = 1;
	}
	if (block_num > 0 && target > block_num) {
		target = block_num;
	}
	return target;
}

void collect_parallel_sequences(Parallel* head, vector<vector<int> >& sequences) {
	Parallel* node = head;
	while (node != NULL && node->next != NULL) {
		node = node->next;
		if (node->L.size() == 0) {
			continue;
		}

		vector<int> sequence;
		list<int>::iterator it = node->L.begin();
		for (; it != node->L.end(); it++) {
			sequence.push_back(*it);
		}
		sequences.push_back(sequence);
	}
}

Parallel* build_parallel_list(const vector<vector<int> >& sequences, int& num_parallel) {
	Parallel* head = new Parallel();
	head->next = NULL;
	Parallel* tail = head;

	for (int i = 0; i < sequences.size(); i++) {
		Parallel* node = new Parallel();
		node->next = NULL;
		for (int j = 0; j < sequences[i].size(); j++) {
			node->L.push_back(sequences[i][j]);
		}
		tail->next = node;
		tail = node;
	}

	Parallel* end_node = new Parallel();
	end_node->next = NULL;
	tail->next = end_node;
	num_parallel = sequences.size() + 1;
	return head;
}

Parallel* balance_parallel_sequences_for_gpu(Parallel* head, BS_node* Block, Node_conflict* B_conf, int block_num, int target_gpu_blocks, int& num_parallel) {
	vector<vector<int> > sequences;
	vector<vector<int> > balanced;
	collect_parallel_sequences(head, sequences);

	for (int i = 0; i < sequences.size(); i++) {
		vector<int> current;
		for (int j = 0; j < sequences[i].size(); j++) {
			int block_id = sequences[i][j];
			if (Block[block_id].block_num == 0) {
				continue;
			}

			current.push_back(block_id);
			if (current.size() == target_gpu_blocks) {
				balanced.push_back(current);
				current.clear();
			}
		}
		if (current.size() > 0) {
			balanced.push_back(current);
		}
	}

	for (int i = 0; i < balanced.size(); i++) {
		while (balanced[i].size() < target_gpu_blocks) {
			int block_id = select_min_conflict_block(Block, block_num, balanced[i]);
			if (block_id == -1) {
				break;
			}
			balanced[i].push_back(block_id);
		}
		update_conflict_weights(B_conf, balanced[i], Block);
	}

	Free_list(head);
	return build_parallel_list(balanced, num_parallel);
}

int count_scheduled_blocks(Parallel* head) {
	int count = 0;
	Parallel* node = head;
	while (node != NULL && node->next != NULL) {
		node = node->next;
		count += node->L.size();
	}
	return count;
}

struct ParallelStats {
	int sequence_count;
	int min_sequence_size;
	int max_sequence_size;
	int scheduled_blocks;
	int unique_blocks;
};

ParallelStats conflict_free_parallel_stats = {0, 0, 0, 0, 0};
ParallelStats after_conflict_parallel_stats = {0, 0, 0, 0, 0};
int report_target_gpu_blocks = 0;

ParallelStats get_parallel_stats(Parallel* head) {
	ParallelStats stats = {0, 0, 0, 0, 0};
	set<int> unique_blocks;
	Parallel* node = head;
	while (node != NULL && node->next != NULL) {
		node = node->next;
		int sequence_size = node->L.size();
		if (sequence_size == 0) {
			continue;
		}

		stats.sequence_count++;
		stats.scheduled_blocks += sequence_size;
		if (stats.min_sequence_size == 0 || sequence_size < stats.min_sequence_size) {
			stats.min_sequence_size = sequence_size;
		}
		if (sequence_size > stats.max_sequence_size) {
			stats.max_sequence_size = sequence_size;
		}
		for (list<int>::iterator it = node->L.begin(); it != node->L.end(); it++) {
			unique_blocks.insert(*it);
		}
	}
	stats.unique_blocks = unique_blocks.size();
	return stats;
}

string rate_output_dir(double rate) {
	stringstream ss;
	ss << rate;
	string dir = output_root;
	if (!dir.empty() && dir[dir.size() - 1] != '/' && dir[dir.size() - 1] != '\\') {
		dir += "/";
	}
	return dir + dataset_name + "_" + ss.str() + "/";
}

void ensure_output_dir(const string& dir) {
#ifdef _WIN32
	string command = string("if not exist \"") + dir + "\" mkdir \"" + dir + "\"";
#else
	string command = string("mkdir -p \"") + dir + "\"";
#endif
	system(command.c_str());
}

string bool_vector_to_string(const vector<bool>& values) {
	if (values.size() == 0) {
		return "-";
	}

	string result;
	for (int i = 0; i < values.size(); i++) {
		result += values[i] ? '1' : '0';
	}
	return result;
}

long long prefix_code(const vector<bool>& values) {
	long long code = 0;
	for (int i = 1; i < values.size(); i++) {
		code = code * 2 + (values[i] ? 1 : 0);
	}
	return code;
}

string axis_range_to_string(const vector<bool>& values, int axis_size) {
	if (axis_size <= 0) {
		return "[]";
	}

	int levels = values.size() > 0 ? (int)values.size() - 1 : 0;
	if (levels <= 0 || levels >= 62) {
		stringstream ss;
		ss << "[0," << axis_size - 1 << "]";
		return ss.str();
	}

	long long partitions = 1LL << levels;
	long long code = prefix_code(values);
	double width = (axis_size + 1.0) / partitions;
	int begin = (int)floor(code * width);
	int end = (int)floor((code + 1) * width) - 1;
	if (begin < 0) {
		begin = 0;
	}
	if (begin >= axis_size) {
		begin = axis_size - 1;
	}
	if (end < begin) {
		end = begin;
	}
	if (end >= axis_size) {
		end = axis_size - 1;
	}

	stringstream ss;
	ss << "[" << begin << "," << end << "]";
	return ss.str();
}

void write_block_parallel_report(BS_node* Block, int block_num, Parallel* head, int num_parallel, double rate) {
	string dir = rate_output_dir(rate);
	ensure_output_dir(dir);
	string report_path = dir + "block_parallel.txt";
	ofstream ofs(report_path.c_str(), ios::out | ios::trunc);
	if (!ofs.is_open()) {
		cerr << "Failed to open " << report_path << endl;
		return;
	}

	int non_empty_blocks = 0;
	int scheduled_blocks = 0;
	int max_sequence_size = 0;
	set<int> unique_scheduled_blocks;
	Parallel* node = head;
	while (node != NULL && node->next != NULL) {
		node = node->next;
		int sequence_size = node->L.size();
		if (sequence_size == 0) {
			continue;
		}
		max_sequence_size = max_sequence_size > sequence_size ? max_sequence_size : sequence_size;
		scheduled_blocks += sequence_size;
		for (list<int>::iterator it = node->L.begin(); it != node->L.end(); it++) {
			unique_scheduled_blocks.insert(*it);
		}
	}

	for (int i = 0; i < block_num; i++) {
		if (Block[i].block_num > 0) {
			non_empty_blocks++;
		}
	}

	ofs << "# block_s " << block_s << endl;
	ofs << "# parallel_sequence_size " << (parallel_sequence_size > 0 ? parallel_sequence_size : 0) << " (0 means auto)" << endl;
	ofs << "# num_block_total " << block_num << endl;
	ofs << "# non_empty_blocks " << non_empty_blocks << endl;
	ofs << "# num_parallel_nodes " << num_parallel << endl;
	ofs << "# scheduled_blocks " << scheduled_blocks << endl;
	ofs << "# unique_scheduled_blocks " << unique_scheduled_blocks.size() << endl;
	ofs << "# max_sequence_size " << max_sequence_size << endl;
	ofs << "# target_gpu_blocks " << report_target_gpu_blocks << endl;
	ofs << "# conflict_free_sequence_count " << conflict_free_parallel_stats.sequence_count << endl;
	ofs << "# conflict_free_min_parallel_blocks " << conflict_free_parallel_stats.min_sequence_size << endl;
	ofs << "# conflict_free_max_parallel_blocks " << conflict_free_parallel_stats.max_sequence_size << endl;
	ofs << "# conflict_free_scheduled_blocks " << conflict_free_parallel_stats.scheduled_blocks << endl;
	ofs << "# conflict_free_unique_blocks " << conflict_free_parallel_stats.unique_blocks << endl;
	ofs << "# after_conflict_sequence_count " << after_conflict_parallel_stats.sequence_count << endl;
	ofs << "# after_conflict_min_parallel_blocks " << after_conflict_parallel_stats.min_sequence_size << endl;
	ofs << "# after_conflict_max_parallel_blocks " << after_conflict_parallel_stats.max_sequence_size << endl;
	ofs << "# after_conflict_scheduled_blocks " << after_conflict_parallel_stats.scheduled_blocks << endl;
	ofs << "# after_conflict_unique_blocks " << after_conflict_parallel_stats.unique_blocks << endl;
	ofs << endl;

	ofs << "[blocks]" << endl;
	ofs << "block_id nnz level_x level_y level_z x_id x_range y_id y_range z_id z_range" << endl;
	for (int i = 0; i < block_num; i++) {
		if (Block[i].block_num == 0) {
			continue;
		}
		ofs << i << ' '
			<< Block[i].block_num << ' '
			<< Block[i].level_x << ' '
			<< Block[i].level_y << ' '
			<< Block[i].level_z << ' '
			<< bool_vector_to_string(Block[i].x_id) << ' '
			<< axis_range_to_string(Block[i].x_id, I) << ' '
			<< bool_vector_to_string(Block[i].y_id) << ' '
			<< axis_range_to_string(Block[i].y_id, J) << ' '
			<< bool_vector_to_string(Block[i].z_id) << ' '
			<< axis_range_to_string(Block[i].z_id, K) << endl;
	}

	ofs << endl;
	ofs << "[parallel_sequences]" << endl;
	ofs << "sequence_id size total_nnz block_ids" << endl;
	node = head;
	int sequence_id = 0;
	while (node != NULL && node->next != NULL) {
		node = node->next;
		if (node->L.size() == 0) {
			continue;
		}
		int total_nnz = 0;
		for (list<int>::iterator it = node->L.begin(); it != node->L.end(); it++) {
			total_nnz += Block[*it].block_num;
		}

		ofs << sequence_id << ' ' << node->L.size() << ' ' << total_nnz << ' ';
		bool first = true;
		for (list<int>::iterator it = node->L.begin(); it != node->L.end(); it++) {
			if (!first) {
				ofs << ',';
			}
			ofs << *it;
			first = false;
		}
		ofs << endl;
		sequence_id++;
	}

	cout << "block_parallel_report=" << report_path << endl;
}
int compare_tree(BTnode* BT, BS_node Block) {
	int flag = 0;
	int x_level = BT->x.size() < Block.x_id.size() ? BT->x.size() : Block.x_id.size();
	int y_level = BT->y.size() < Block.y_id.size() ? BT->y.size() : Block.y_id.size();
	int z_level = BT->z.size() < Block.z_id.size() ? BT->z.size() : Block.z_id.size();

	if (BT->block != -1) {
		for (int block_id = 1; block_id < x_level; block_id++) {
			if (BT->x[block_id] ^ Block.x_id[block_id]) {   //���  ��ͬΪ1����ͬΪ0
				flag = 1;   //����1Ϊ����ͻ,���Բ���
				break;
			}
		}
		if (flag == 0)
			return 0;

		flag = 0;
		for (int block_id = 1; block_id < y_level; block_id++) {
			if (BT->y[block_id] ^ Block.y_id[block_id]) {   //���  ��ͬΪ1����ͬΪ0
				flag = 1;   //����1Ϊ����ͻ,���Բ���
				break;
			}
		}
		if (flag == 0)
			return 0;

		flag = 0;
		for (int block_id = 1; block_id < z_level; block_id++) {
			if (BT->z[block_id] ^ Block.z_id[block_id]) {   //���  ��ͬΪ1����ͬΪ0
				flag = 1;   //����1Ϊ����ͻ,���Բ���
				break;
			}
		}
		if (flag == 0)
			return 0;
		else
			return 1;
	}
	else if (BT->block == -1) {
		if (BT->x.size() < Block.x_id.size() && BT->y.size() < Block.y_id.size() && BT->z.size() < Block.z_id.size()) {
			return 1;  //x,y,z�����껹���Լ������±Ƚ�				
		}

		if (BT->x.size() >= Block.x_id.size() ) {    //����ȥif flag=0
			for (int block_id = 1; block_id < Block.x_id.size(); block_id++) {
				if (BT->x[block_id] ^ Block.x_id[block_id]) {   //���  ��ͬΪ1����ͬΪ0
					flag = 1;
					break;
				}
			}
			if (flag == 0) //id��ͻ
				return 0;
		}
		
		flag = 0;
		if (BT->y.size() >= Block.y_id.size()) {
			for (int block_id = 1; block_id < Block.y_id.size(); block_id++) {
				if (BT->y[block_id] ^ Block.y_id[block_id]) {   //���  ��ͬΪ1����ͬΪ0
					flag = 1;
					break;
				}
			}
			if (flag == 0)
				return 0;
		}
		
		flag = 0;
		if (BT->z.size() >= Block.z_id.size()) {
			for (int block_id = 1; block_id < Block.z_id.size(); block_id++) {
				if (BT->z[block_id] ^ Block.z_id[block_id]) {   //���  ��ͬΪ1����ͬΪ0
					flag = 1;
					break;
				}
			}
			if (flag == 0)
				return 0;
		}
		
		if (flag == 1)
			return 1;		
	}
	
	return -1;
}

int tree_conflict(set<int>& block_parallel, BTnode* BT_conflict) {
	if (BT_conflict == NULL)
		return 0;

	if (BT_conflict->block != -1) {
		block_parallel.erase(BT_conflict->block);
	}

	tree_conflict(block_parallel, BT_conflict->lChild);
	tree_conflict(block_parallel, BT_conflict->rChild);
	return -1;
}

Parallel* search_parallel_block_Tree(BS_node* Block, Node_conflict* B_conf, int block_num, int& num_parallel, BTnode* BT) {
	Parallel* Parallel_head = new Parallel(); // 只创建头节点
	Parallel* Parallel_list = Parallel_head;  // 工作指针指向头节点
	Parallel* tmp = new Parallel();
	tmp->next = NULL;
	Parallel_list->next = tmp;
	Parallel_list = Parallel_list->next;
	
	int max_parallel = 0;
	num_parallel = 1;
	//vector<bool> parallel_block_id;	
	//Parallel* P, * head;
	//P = new Parallel();
	//head = new Parallel();
	//P->next = NULL;
	//head = P;

	set<int> block_id;
		
	for (int i = 0; i < block_num; i++) {
		if (Block[i].block_num != 0) {
			block_id.insert(i);
		}
	}

	while (block_id.size() != 0) {
		set<int> block_parallel;

		auto it = block_id.begin();
		int selected = *it;
		block_id.erase(selected);

		Parallel_list->L.push_back(selected); 
		
		BTnode* BT_tmp = new BTnode();
		BTnode* BT_tmp_head = new BTnode();
		BT_tmp = BT;		
		BT_tmp_head = BT_tmp;

		block_parallel = block_id;
		while (block_parallel.size() != 0) {
			
			queue<BTnode*> q;
			BT_tmp = BT_tmp_head;
			q.push(BT_tmp);
			while (!q.empty()) {   //������ȱ�����,�ҵ����Բ��е�һ��Block
				if (compare_tree(q.front(), Block[selected]) == 0) {
					
					BTnode* BT_conflict = new BTnode();
					BT_conflict = q.front();
					tree_conflict(block_parallel, BT_conflict);
					compare_tree(q.front(), Block[selected]);
				
					//q.front()->lChild = q.front()->rChild = NULL;
				}
				else {
					if (q.front()->lChild != NULL)
						q.push(q.front()->lChild);
					if (q.front()->rChild != NULL)
						q.push(q.front()->rChild);
				}
				
				q.pop();
				
			}
			
			while (block_parallel.size() != 0) {
				auto it_p = block_parallel.begin();
				selected = *it_p;
				if (Block[selected].block_num != 0) {
					Parallel_list->L.push_back(selected);
					block_parallel.erase(selected);
					block_id.erase(selected);
					break;
				}
				else {
					block_parallel.erase(selected);
					block_id.erase(selected);
				}
			}
					
		}
		
		Parallel* Parallel_tmp = new Parallel();
		Parallel_tmp->next = NULL;
		Parallel_list->next = Parallel_tmp;
		int num_p = Parallel_list->L.size();
		max_parallel = max_parallel > num_p ? max_parallel : num_p;
		Parallel_list = Parallel_list->next;
		num_parallel++;
	}

	Parallel_list = Parallel_head;
	while (Parallel_list->next != NULL) {
		Parallel_list = Parallel_list->next;
		list<int>::iterator it = Parallel_list->L.begin();
		for (; it != Parallel_list->L.end(); it++) {
			B_conf[*it].coe_x = 1;
			B_conf[*it].coe_y = 1;
			B_conf[*it].coe_z = 1;
		}
	}

	conflict_free_parallel_stats = get_parallel_stats(Parallel_head);
	int target_gpu_blocks = configured_gpu_block_target(max_parallel, block_num);
	report_target_gpu_blocks = target_gpu_blocks;
	cout << "max_parallel=" << max_parallel << " target_gpu_blocks=" << target_gpu_blocks;
	if (parallel_sequence_size > 0) {
		cout << " configured_parallel_sequence_size=" << parallel_sequence_size;
	}
	cout << endl;
	if (flag_preproccess == 1) {
		Parallel_head = balance_parallel_sequences_for_gpu(Parallel_head, Block, B_conf, block_num, target_gpu_blocks, num_parallel);
	}
	else {
		Parallel_list = Parallel_head;
		while (Parallel_list->next != NULL) {
			Parallel_list = Parallel_list->next;
			update_conflict_weights(B_conf, Parallel_list, Block);
		}
	}
	after_conflict_parallel_stats = get_parallel_stats(Parallel_head);

	return Parallel_head;
}


int Preproccess_list(Parallel* P, BS_node* Block, Node_conflict* B_conf, b_node* bs, int* num_bs, int I, int J, int K) {
	int count = 0;
	int size = 0;
	int max_parallel = 0;
	P = P->next;
	while (P->next != NULL) {
		update_conflict_weights(B_conf, P, Block);
		int num = P->L.size();
		max_parallel = max_parallel > num ? max_parallel : num;
		num_bs[size] = num;
		
    	for (auto it = P->L.begin(); it != P->L.end(); ++it) {

			int block_count = Block[*it].block_num;
			bs[count].x = new int[block_count];
			bs[count].y = new int[block_count];
			bs[count].z = new int[block_count];
			bs[count].rate = new double[block_count];
			bs[count].block_num = Block[*it].block_num;
			bs[count].id = *it;
			bs[count].coe_x = B_conf[*it].coe_x;
			bs[count].coe_y = B_conf[*it].coe_y;
			bs[count].coe_z = B_conf[*it].coe_z;
			
			for (int i = 0; i < Block[*it].block_num; ++i) {
				bs[count].x[i] = Block[*it].t[i].x;
				bs[count].y[i] = Block[*it].t[i].y;
				bs[count].z[i] = Block[*it].t[i].z;
				bs[count].rate[i] = Block[*it].t[i].rate;
				
			}
			count++;
   		}
		size++;
		P = P->next;
	}	
	
	return max_parallel;

}

//lock_freeԤ����
void ToMatrix_LF(Parallel* P, BS_node* Block, LF_node* LF, int* num_LS, int I, int J, int K) {
	int count = 0;
	int size = 0;
	while (P->next != NULL) {
		P = P->next;
		int num = 0;
		list<int>::iterator it = P->L.begin();
		for (; it != P->L.end(); it++) {
			for (int i = 0; i < Block[*it].block_num; i++) {
				int x_b, y_b, z_b;
				x_b = Block[*it].t[i].x;
				y_b = Block[*it].t[i].y;
				z_b = Block[*it].t[i].z;
				LF[count].rate = Block[*it].t[i].rate;
				LF[count].x = x_b;
				LF[count].y = y_b;
				LF[count].z = z_b;
				count++;
				num++;
			}
		}
		num_LS[size] = num;
		size++;
	}
}

//��ʼ��Block
void initial_B(BS_node* Block) {
	Block[0].id = 0;
	Block[0].level_x = 1;
	Block[0].level_y = 1;
	Block[0].level_z = 1;
	Block[0].x_id.push_back(0);
	Block[0].y_id.push_back(0);
	Block[0].z_id.push_back(0);
	Block[0].block_num = 0;
	Block[0].t = (T_node*)malloc(block_s * sizeof(T_node));
}


void block_problem(
	const T_node* train_entries,
	int train_nnz,
	const T_node* test_entries,
	int test_nnz,
	double* a,
	double* b,
	double* c,
	double rate) {

	T_node* t_block = new T_node[train_nnz];
	BS_node* Block;
	Parallel* head;
	Node_conflict* B_conf;
	int num_block = 0;
	int num_parallel = 0;
	int max_parallel = 1;
	LF_node* LF = NULL;
	int* num_LF = NULL;

	BTnode* BT = new BTnode();
	BTnode* BT_head = BT;
	
	Block = new BS_node[train_nnz > 0 ? train_nnz : 1];    

	initial_B(Block);
	initial_t(t_block, train_entries, train_nnz);
	
	num_block = tensor_block(Block, t_block, train_nnz, I, J, K, BT, BT_head);          
	
	BT = BT_head;
	B_conf = new Node_conflict[num_block];
	
	head = search_parallel_block_Tree(Block, B_conf, num_block, num_parallel, BT);
	write_block_parallel_report(Block, num_block, head, num_parallel, rate);
	
	int* num_bs = NULL;
	b_node* bs = NULL;
	int scheduled_bs = 0;
	if (flag_preproccess == 1) {
		int parallel_count = num_parallel > 1 ? num_parallel - 1 : 1;
		num_bs = new int[parallel_count];
		scheduled_bs = count_scheduled_blocks(head);
		bs = new b_node[scheduled_bs > 0 ? scheduled_bs : 1];
		max_parallel = Preproccess_list(head, Block, B_conf, bs, num_bs, I, J, K);
	}

	if (flag_lockfree == 1) {
		LF = new LF_node[train_nnz];
		num_LF = new int[num_parallel];
		ToMatrix_LF(head, Block, LF, num_LF, I, J, K);
	}
	sgd_train(train_entries, train_nnz, test_entries, test_nnz, a, b, c, num_parallel, max_parallel, LF, num_LF, rate, num_bs, bs, scheduled_bs);

}
