#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <set>
#include <Windows.h>
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


using namespace std;

//获取时间
double get_time(void)
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

//取样后重排
void initial_t(T_node* t_block, double* t, int I, int J, int K, int nnz) {
	int num = 0;

	for (int m = 0; m < K; m++) {
		for (int j = 0; j < J; j++) {
			for (int i = 0; i < I; i++) {
				if (t[i * J * K + j * K + m] > 0) {
					t_block[num].x = i;
					t_block[num].y = j;
					t_block[num].z = m;
					num++;
				}
			}
		}
	}
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> distribution(0, nnz * 0.6);
	int random = distribution(gen);

	int size = random;
	for (int i = size; i < nnz; i++) {
		uniform_real_distribution<> distribution(0, nnz);
		int random = distribution(gen);
		if (random < size) {
			T_node tmp_t;
			tmp_t.x = t_block[i].x;
			tmp_t.y = t_block[i].y;
			tmp_t.z = t_block[i].z;
			t_block[i].x = t_block[random].x;
			t_block[i].y = t_block[random].y;
			t_block[i].z = t_block[random].z;
			t_block[random].x = tmp_t.x;
			t_block[random].y = tmp_t.y;
			t_block[random].z = tmp_t.z;
		}
	}
}

//坐标化为二进制
vector<bool> hx(int end, int start, int level, int num) {
	vector<bool> binary;

	float n_1 = (end - start + 1) / pow(2, level - 1);
	int n_2 = floor(num / n_1);

	bitset<100> b1(n_2);

	for (int i = level - 1; i >= 0; i--) {
		binary.push_back(b1[i]);
	}
	return binary;
	
	/*
	bitset<100> b1(n_2);
	int* bit = new int[level];
	int num = 0;
	for (int i = level - 1; i >= 0; i--) {
		bit[num] = b1[i];
		num++;
	}
	return bit;
	*/

}

//坐标与索引是否匹配
int compare(vector<bool> id, vector<bool> x, int level) {
	for (int block_id = 1; block_id < level; block_id++) {
		if (x[block_id] ^ id[block_id]) {   //异或  不同为1，相同为0
			return 1;   //返回1为不冲突
		}
	}
	return 0;// id相同
/*
	for (int block_id = 0; block_id < level; block_id++) {
		if (x[block_id] != id[block_id]) {
			flag = 1;
			break;
		}
	}

	if (flag == 0)
		return 0;
	else if (flag == 1)
		return 1;
	else
		return false;*/
}

//找到与坐标匹配的BlockID
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
/*
	vector<bool> tmp_x = x;
	vector<bool> tmp_y = y;
	vector<bool> tmp_z = z;
	tmp_x.erase(tmp_x.begin());
	tmp_y.erase(tmp_y.begin());
	tmp_z.erase(tmp_z.begin());
	while (BT->lChild != NULL) {
		int tmp_h = BT->h % 3;
		if (tmp_h == 1) {
			int s_x = tmp_x.front();
			tmp_x.erase(tmp_x.begin());
			if (s_x == 0)
				BT = BT->lChild;
			else
 				BT = BT->rChild;
		}
		else if (tmp_h == 2) {
			int s_y = tmp_y.front();
			tmp_y.erase(tmp_y.begin());
			if (s_y == 0)
				BT = BT->lChild;
			else
				BT = BT->rChild;
		}
		else {
			int s_z = tmp_z.front();
			tmp_z.erase(tmp_z.begin());
			if (s_z == 0)
				BT = BT->lChild;
			else
				BT = BT->rChild;
		}
	}
*/	
	return BT;
}
/*
int search_block(BS_node* Block, int* x, int* y, int* z, int block_id_num) {
	
	int search_id = -1;
	int flag = 1;
	for (int i = 0; i < block_id_num; i++) {
		int flag = compare(Block[i].x_id, x, Block[i].level_x);
		if (flag == 0) {
			flag = compare(Block[i].y_id, y, Block[i].level_y);
		}
		if (flag == 0) {
			flag = compare(Block[i].z_id, z, Block[i].level_z);
		}
		if (flag == 0) {
			search_id = i;
			break;
		}
	}
	return search_id;
}*/

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
/*
int* ID_change(int* id, int number, int level) {
	int* s = new int[level];
	for (int i = 0; i < level - 1; i++) {
		s[i] = id[i];
	}
	s[level - 1] = number;
	return s;
}

//判断是否要继续分块
int continue_partition(int* t, int* id, int* x, int level, int end, int start, int g) {
	int flag = 1;
	int num = 0;
	flag = compare(id, x, level);
	if (flag == 0)
		num++;
	for (int i = 0; i < block_s; i++) {
		int* Bx = hx(end, start, g, t[i]);
		flag = compare(id, Bx, level);
		if (flag == 0)
			num++;
	}
	if (num == block_s + 1 || num == 0)
		return 0;
	else
		return 1;
}*/

//数据块划分
int tensor_block(BS_node* Block, T_node* t, int nnz, int I, int J, int K, BTnode* BT, BTnode* BT_head) {
	int Block_num = 1;//Block数量
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
			Block[id].t[Block[id].block_num].x = t[i].x;
			Block[id].t[Block[id].block_num].y = t[i].y;
			Block[id].t[Block[id].block_num].z = t[i].z;
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

					//数据重新分配
					int num_1 = 0;
					int num_2 = 0;
					int* tx = new int[block_s];
					T_node* t1 = new T_node[block_s];
					T_node* t2 = new T_node[block_s];
					for (int j = 0; j < Block[id].block_num; j++) {
						vector<bool> xt = hx(end_x, start, g_x, Block[id].t[j].x);
						//int flag_1 = compare(Block[id].x_id, xt, Block[id].level_x);
						if (xt.back() == 0) {
							t1[num_1].x = Block[id].t[j].x;
							t1[num_1].y = Block[id].t[j].y;
							t1[num_1].z = Block[id].t[j].z;
							num_1++;
						}
						else {
							t2[num_2].x = Block[id].t[j].x;
							t2[num_2].y = Block[id].t[j].y;
							t2[num_2].z = Block[id].t[j].z;
							num_2++;
						}					
					}
					vector<bool> xt = hx(end_x, start, g_x, t[i].x);
					if (num_1 < block_s && num_2 < block_s) {
						flag = 1;
						//int flag_1 = compare(Block[id].x_id, x, Block[id].level_x);
						
						if (xt.back() == 0) {
							t1[num_1].x = t[i].x;
							t1[num_1].y = t[i].y;
							t1[num_1].z = t[i].z;
							num_1++;
						}
						else {
							t2[num_2].x = t[i].x;
							t2[num_2].y = t[i].y;
							t2[num_2].z = t[i].z;
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
																
					/*flag = continue_partition(tx, Block[id].x_id, x, Block[id].level_x, end_x, start, g_x);

					if (flag == 0) {
						id = search_block(Block, x, y, z, block_id_num);
					}
					else {
						int num_1 = 0;
						int num_2 = 0;
						T_node* t1 = new T_node[block_s];
						T_node* t2 = new T_node[block_s];
						int flag_1 = compare(Block[id].x_id, x, Block[id].level_x);
						if (flag_1 == 0) {
							t1[num_1].x = t[i].x;
							t1[num_1].y = t[i].y;
							t1[num_1].z = t[i].z;
							num_1++;
						}
						else {
							t2[num_2].x = t[i].x;
							t2[num_2].y = t[i].y;
							t2[num_2].z = t[i].z;
							num_2++;
						}
						for (int j = 0; j < Block[id].block_num; j++) {
							int* xt = hx(end_x, start, g_x, Block[id].t[j].x);
							int flag_1 = compare(Block[id].x_id, xt, Block[id].level_x);
							if (flag_1 == 0) {
								t1[num_1].x = Block[id].t[j].x;
								t1[num_1].y = Block[id].t[j].y;
								t1[num_1].z = Block[id].t[j].z;
								num_1++;
							}
							else {
								t2[num_2].x = Block[id].t[j].x;
								t2[num_2].y = Block[id].t[j].y;
								t2[num_2].z = Block[id].t[j].z;
								num_2++;
							}

							delete xt;
						}

						Block[id].t = t1;
						Block[id].block_num = num_1;
						Block[block_id_num].t = t2;
						Block[block_id_num].block_num = num_2;
					}*/
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
						//int flag_1 = compare(Block[id].y_id, yt, Block[id].level_y);
						if (yt.back() == 0) {
							t1[num_1].x = Block[id].t[j].x;
							t1[num_1].y = Block[id].t[j].y;
							t1[num_1].z = Block[id].t[j].z;
							num_1++;
						}
						else {
							t2[num_2].x = Block[id].t[j].x;
							t2[num_2].y = Block[id].t[j].y;
							t2[num_2].z = Block[id].t[j].z;
							num_2++;
						}						                  
					}
					vector<bool> yt = hx(end_y, start, g_y, t[i].y);
					if (num_1 < block_s && num_2 < block_s) {
						flag = 1;
						//int flag_1 = compare(Block[id].x_id, x, Block[id].level_x);
						
						if (yt.back() == 0) {
							t1[num_1].x = t[i].x;
							t1[num_1].y = t[i].y;
							t1[num_1].z = t[i].z;
							num_1++;
						}
						else {
							t2[num_2].x = t[i].x;
							t2[num_2].y = t[i].y;
							t2[num_2].z = t[i].z;
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
					/*y = hx(end_y, start, g_y, t[i].y);
					for (int j = 0; j < Block[id].block_num; j++) {
						ty[j] = Block[id].t[j].y;
					}
					flag = continue_partition(ty, Block[id].y_id, y, Block[id].level_y, end_y, start, g_y);

					if (flag == 0) {
						id = search_block(Block, x, y, z, block_id_num);
					}
					else {
						int num_1 = 0;
						int num_2 = 0;
						T_node* t1 = new T_node[block_s];
						T_node* t2 = new T_node[block_s];
						int flag_1 = compare(Block[id].y_id, y, Block[id].level_y);
						if (flag_1 == 0) {
							t1[num_1].x = t[i].x;
							t1[num_1].y = t[i].y;
							t1[num_1].z = t[i].z;
							num_1++;
						}
						else {
							t2[num_2].x = t[i].x;
							t2[num_2].y = t[i].y;
							t2[num_2].z = t[i].z;
							num_2++;
						}
						for (int j = 0; j < Block[id].block_num; j++) {
							int* yt = hx(end_y, start, g_y, Block[id].t[j].y);
							int flag_1 = compare(Block[id].y_id, yt, Block[id].level_y);
							if (flag_1 == 0) {
								t1[num_1].x = Block[id].t[j].x;
								t1[num_1].y = Block[id].t[j].y;
								t1[num_1].z = Block[id].t[j].z;
								num_1++;
							}
							else {
								t2[num_2].x = Block[id].t[j].x;
								t2[num_2].y = Block[id].t[j].y;
								t2[num_2].z = Block[id].t[j].z;
								num_2++;
							}

							delete yt;
						}

						Block[id].t = t1;
						Block[id].block_num = num_1;
						Block[block_id_num].t = t2;
						Block[block_id_num].block_num = num_2;
					}*/
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
						//int flag_1 = compare(Block[id].z_id, zt, Block[id].level_z);
						if (zt.back() == 0) {
							t1[num_1].x = Block[id].t[j].x;
							t1[num_1].y = Block[id].t[j].y;
							t1[num_1].z = Block[id].t[j].z;
							num_1++;
						}
						else {
							t2[num_2].x = Block[id].t[j].x;
							t2[num_2].y = Block[id].t[j].y;
							t2[num_2].z = Block[id].t[j].z;
							num_2++;
						}
					}
					vector<bool> zt = hx(end_z, start, g_z, t[i].z);
					if (num_1 < block_s && num_2 < block_s) {
						flag = 1;
						//int flag_1 = compare(Block[id].x_id, x, Block[id].level_x);
						
						if (zt.back() == 0) {
							t1[num_1].x = t[i].x;
							t1[num_1].y = t[i].y;
							t1[num_1].z = t[i].z;
							num_1++;
						}
						else {
							t2[num_2].x = t[i].x;
							t2[num_2].y = t[i].y;
							t2[num_2].z = t[i].z;
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
		//delete x;		delete y;		delete z;

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

void destroyTree(BTnode* root) {
	if (root == NULL) {
		return;
	}
	destroyTree(root->lChild);
	destroyTree(root->rChild);
	free(root);
}

int compare_parallel(BS_node a, BS_node b) {
	int flag = 1;
	int level = 1;
	level = a.level_x < b.level_x ? a.level_x : b.level_x;
	if (level == 1)
		flag = 1;
	else {
		int block_x;
		for (block_x = 1; block_x < level; block_x++) {
			if (a.x_id[block_x] != b.x_id[block_x]) {
				flag = 0;
				break;
			}
		}
		if (block_x == level)
			return 1;
	}
	if (flag == 0) {
		level = a.level_y < b.level_y ? a.level_y : b.level_y;
		if (level == 1)
			flag = 1;
		else {
			int block_y;
			for (block_y = 1; block_y < level; block_y++) {
				if (a.y_id[block_y] != b.y_id[block_y]) {
					flag = 0;
					break;
				}
			}
			if (block_y == level)
				return 1;
		}
	}
	if (flag == 0) {
		level = a.level_z < b.level_z ? a.level_z : b.level_z;
		if (level == 1)
			flag = 1;
		else {
			int block_z;
			for (block_z = 1; block_z < level; block_z++) {
				if (a.z_id[block_z] != b.z_id[block_z]) {
					flag = 0;
					break;
				}
			}
			if (block_z == level)
				return 1;
		}
	}

	if (flag == 0)//可以并行
		return 0;
	else if (flag == 1)
		return 1;
	else
		return false;
}

int compare_parallel_2(BS_node a, BS_node b, int num) {
	int flag = 1;
	int level = 1;
	level = a.level_x < b.level_x ? a.level_x : b.level_x;
	if (level == 1)
		flag = 1;
	else {
		int block_x;
		for (block_x = 1; block_x < level; block_x++) {
			if (a.x_id[block_x] != b.x_id[block_x]) {
				flag = 0;
				break;
			}
		}
		if (block_x == level) {
			num++;
			flag == 1;
		}

	}
	if (flag == 0) {
		level = a.level_y < b.level_y ? a.level_y : b.level_y;
		if (level == 1)
			flag = 1;
		else {
			int block_y;
			for (block_y = 1; block_y < level; block_y++) {
				if (a.y_id[block_y] != b.y_id[block_y]) {
					flag = 0;
					break;
				}
			}
			if (block_y == level) {
				num++;
				flag = 1;
			}
		}
	}
	if (flag == 0) {
		level = a.level_z < b.level_z ? a.level_z : b.level_z;
		if (level == 1)
			flag = 1;
		else {
			int block_z;
			for (block_z = 1; block_z < level; block_z++) {
				if (a.z_id[block_z] != b.z_id[block_z]) {
					flag = 0;
					break;
				}
			}
			if (block_z == level) {
				num++;
				flag = 1;
			}
		}
	}

	if (flag == 0)//可以并行
		return 0;
	else if (flag == 1)
		return 1;
	else
		return false;
}

//寻找并行Block
Parallel* search_parallel_block_2(BS_node* Block, int block_num, int& num_parallel, int& max_parallel, int& block_pro) {
	num_parallel = 1;
	Parallel* P, * head, * tmp;
	P = new Parallel();
	head = new Parallel();
	tmp = new Parallel();
	P->next = NULL;
	head = P;

	tmp->L.push_back(0);
	tmp->next = P->next;
	P->next = tmp;
	for (int i = 1; i < block_num; i++) {
		int flag = 0;
		P = head;
		while (P->next != NULL) {
			P = P->next;
			int num = P->L.size();
			max_parallel = max_parallel > num ? max_parallel : num;

			list<int>::iterator it = P->L.begin();
			for (it; it != P->L.end(); it++) {
				flag = compare_parallel(Block[i], Block[*it]);
				if (flag == 1)
					break;
			}
			if (flag == 0) {
				P->L.push_back(i);
				break;
			}
		}
		if (flag == 1) {
			num_parallel++;
			Parallel* s;
			s = new Parallel();
			s->L.push_back(i);
			s->next = NULL;
			P->next = s;
			P = s;
		}
	}
	/**/
	P = head;
	list<int> L;
	while (P->next != NULL) {
		P = P->next;
		int num = P->L.size();
		if (num == 1) {
			int number = P->L.front();
			L.push_back(number); 
			P->L.pop_front();
		}
	}
	P = head;
	
	while (P->next != NULL) {
		P = P->next;
		int num = 0;
		if (P->L.size() < max_parallel) {
			list<int>::iterator it = P->L.begin();
			for (it; it != P->L.end(); it++) {
				if (num > 1)
					break;
				else {
					list<int>::iterator it_1 = L.begin();
					for (it_1; it_1 != P->L.end(); it_1++) {
//						compare_parallel_2()
					}
				}
				
			}
			
			for (int i = 0; i < block_num; i++) {
				if (P->L.size() == max_parallel)
					break;
				int flag = 1;
				list<int>::iterator it = P->L.begin();
				for (it; it != P->L.end(); it++) {
					flag = compare_parallel(Block[i], Block[*it]);
					if (flag == 1)
						break;
				}

				if (flag == 0 && P->L.size() < max_parallel) {
					block_pro++;
					P->L.push_back(i);
				}
			}
		}

	}


		while (P->next != NULL) {
			P = P->next;
			if (P->L.size() == max_parallel) {

			}
			if (P->L.size() < max_parallel) {

				for (int i = 0; i < block_num; i++) {
					if (P->L.size() == max_parallel)
						break;
					int flag = 1;
					list<int>::iterator it = P->L.begin();
					for (it; it != P->L.end(); it++) {
						flag = compare_parallel(Block[i], Block[*it]);
						if (flag == 1)
							break;
					}

					if (flag == 0 && P->L.size() < max_parallel) {
						block_pro++;
						P->L.push_back(i);
					}
				}
			}

		}
	
	return head;

}

int find_conflict_x(vector<bool> a, vector<bool> b, int level_a, int level_b) {
	int level = level_a < level_b ? level_a : level_b;

	if (level == 1)
		return 1;//有冲突
	else {
		int block_x;
		for (block_x = 1; block_x < level; block_x++) {
			if (a[block_x] != b[block_x]) {
				return 0;//不冲突
				break;
			}
		}
		if (block_x == level)
			return 1;
	}
}
void find_conflict(Node_conflict* B_conf, Parallel* P, BS_node* Block, int num_bid_conf) {//找并行块之间冲突的数量
	
	int flag_x, flag_y, flag_z;
	Parallel* x_conf, * y_conf, * z_conf;
	Parallel* x_head, * y_head, * z_head;
	//int* id_conf = new int[num];

	int id_i = 0;
	flag_x = flag_y = flag_z = 0;
	x_conf = new Parallel();
	y_conf = new Parallel();
	z_conf = new Parallel();

	x_head = x_conf;
	y_head = y_conf;
	z_head = z_conf;

		list<int>::iterator it = P->L.begin();
		for (it; it != P->L.end(); it++) {
		
			flag_x = find_conflict_x(Block[*it].x_id, Block[num_bid_conf].x_id, Block[*it].level_x, Block[num_bid_conf].level_x);
			flag_y = find_conflict_x(Block[*it].y_id, Block[num_bid_conf].y_id, Block[*it].level_y, Block[num_bid_conf].level_y);
			flag_z = find_conflict_x(Block[*it].z_id, Block[num_bid_conf].z_id, Block[*it].level_z, Block[num_bid_conf].level_z);
		
			if (flag_x == 1) {
				B_conf[*it].coe_x *= 0.5;
				B_conf[num_bid_conf].coe_x *= 0.5;
			}
			if (flag_y == 1) {
				B_conf[*it].coe_y *= 0.5;
 				B_conf[num_bid_conf].coe_y *= 0.5;
			}
			if (flag_z == 1) {
				B_conf[*it].coe_z *= 0.5;
				B_conf[num_bid_conf].coe_z *= 0.5;
			}
	
		}
/*	list<int>::iterator it = P->L.begin();
	for (it; it != P->L.end(); it++) {
		id_conf[id_i] = *it;
		id_i++;
	}

	x_conf->L.push_back(id_conf[0]);
	y_conf->L.push_back(id_conf[0]);
	z_conf->L.push_back(id_conf[0]);
	x_conf->next = NULL;
	y_conf->next = NULL;
	z_conf->next = NULL;

	for (int i = 1; i < num; i++) {

		x_conf = x_head;
		while (x_conf != NULL) {
			flag_x = find_conflict_x(Block[id_conf[i]], Block[x_conf->L.front()], Block[id_conf[i]].level_x, Block[x_conf->L.front()].level_x);

			if (flag_x == 1) {
				x_conf->L.push_front(id_conf[i]);
				break;
			}
			x_conf = x_conf->next;
		}
		if (flag_x == 0) {
			Parallel* s;
			s = new Parallel;
			s->L.push_back(id_conf[i]);
			s->next = NULL;
			//x_conf->next = s;
			x_conf = s;
		}

		y_conf = y_head;
		while (y_conf != NULL) {
			flag_x = find_conflict_x(Block[id_conf[i]], Block[y_conf->L.front()], Block[id_conf[i]].level_y, Block[y_conf->L.front()].level_y);

			if (flag_x == 1) {
				y_conf->L.push_front(id_conf[i]);
				break;
			}
			y_conf = y_conf->next;
		}
		if (flag_x == 0) {
			Parallel* s;
			s = new Parallel;
			s->L.push_back(id_conf[i]);
			s->next = NULL;
			//y_conf->next = s;
			y_conf = s;
		}

		z_conf = z_head;
		while (z_conf != NULL) {
			flag_x = find_conflict_x(Block[id_conf[i]], Block[z_conf->L.front()], Block[id_conf[i]].level_z, Block[z_conf->L.front()].level_z);

			if (flag_x == 1) {
				z_conf->L.push_front(id_conf[i]);
				break;
			}
			z_conf = z_conf->next;
		}
		if (flag_x == 0) {
			Parallel* s;
			s = new Parallel;
			s->L.push_back(id_conf[i]);
			s->next = NULL;
			//z_conf->next = s;
			z_conf = s;
		}

	}

	x_conf = x_head;
	y_conf = y_head;
	z_conf = z_head;
	int num_conf = 1;

	while (x_conf != NULL) {
		num_conf = x_conf->L.size();
		list<int>::iterator it = x_conf->L.begin();
		for (it; it != x_conf->L.end(); it++) {
			B_conf[*it].coe_x = 1 / num_conf;
		}
	}

	while (y_conf != NULL) {
		num_conf = y_conf->L.size();
		list<int>::iterator it = y_conf->L.begin();
		for (it; it != y_conf->L.end(); it++) {
			B_conf[*it].coe_y = 1 / num_conf;
		}
	}

	while (y_conf != NULL) {
		num_conf = z_conf->L.size();
		list<int>::iterator it = z_conf->L.begin();
		for (it; it != z_conf->L.end(); it++) {
			B_conf[*it].coe_z = 1 / num_conf;
		}
	}*/
}

void find_conflict_list(Node_conflict* B_conf, Parallel* P, BS_node* Block) {//找并行块之间冲突的数量

	int flag_x, flag_y, flag_z;
	Parallel* x_conf, * y_conf, * z_conf;
	Parallel* x_head, * y_head, * z_head;
	//int* id_conf = new int[num];

	int id_i = 0;
	flag_x = flag_y = flag_z = 0;
	x_conf = new Parallel();
	y_conf = new Parallel();
	z_conf = new Parallel();

	x_head = x_conf;
	y_head = y_conf;
	z_head = z_conf;

	list<int>::iterator it = P->L.begin();
	for (it; it != P->L.end(); it++) {

		list<int>::iterator it_t = it;
		it_t++;
		for (it_t; it_t != P->L.end(); it_t++) {
			flag_x = find_conflict_x(Block[*it].x_id, Block[*it_t].x_id, Block[*it].level_x, Block[*it_t].level_x);
			flag_y = find_conflict_x(Block[*it].y_id, Block[*it_t].y_id, Block[*it].level_y, Block[*it_t].level_y);
			flag_z = find_conflict_x(Block[*it].z_id, Block[*it_t].z_id, Block[*it].level_z, Block[*it_t].level_z);

			if (flag_x == 1) {
				B_conf[*it].coe_x *= 0.5;
				B_conf[*it_t].coe_x *= 0.5;
			}
			if (flag_y == 1) {
				B_conf[*it].coe_y *= 0.5;
				B_conf[*it_t].coe_y *= 0.5;
			}
			if (flag_z == 1) {
				B_conf[*it].coe_z *= 0.5;
				B_conf[*it_t].coe_z *= 0.5;
			}
		
		}

	}
	
}




Parallel* search_parallel_block(BS_node* Block, Node_conflict* B_conf, int block_num, int& num_parallel) {
	num_parallel = 1;
	list<int> size_one;//存储P中只有一个block的节点
	//int* size_one = new int[];
	int max_parallel = 0;
	Parallel* P, * head, * tmp;
	P = new Parallel();
	head = new Parallel();
	tmp = new Parallel();
	P->next = NULL;
	head = P;

	tmp->L.push_back(0);
	tmp->next = P->next;
	P->next = tmp;
	for (int i = 1; i < block_num; i++) {
		int flag = 0;
		P = head;
		while (P->next != NULL) {
			P = P->next;
			list<int>::iterator it = P->L.begin();
			for (it; it != P->L.end(); it++) {
				flag = compare_parallel(Block[i], Block[*it]);//判断是否可以并行
				if (flag == 1)
					break;
			}
			if (flag == 0) {
				P->L.push_back(i);
				int num_L = P->L.size();
				max_parallel = max_parallel > num_L ? max_parallel : num_L;
				break;
			}
		}
		if (flag == 1) {
			num_parallel++;
			Parallel* s;
			s = new Parallel;
			s->L.push_back(i);
			s->next = NULL;
			P->next = s;
			P = s;
		}
		
	}

	P = head;
	while (P->next != NULL) {
		P = P->next;
		list<int>::iterator it = P->L.begin();
		for (it; it != P->L.end(); it++) {
			B_conf[*it].coe_x = 1;
			B_conf[*it].coe_y = 1;
			B_conf[*it].coe_z = 1;
		}
	}

	//取出P中只有一个块的节点
	P = head;
	Parallel* front;

	while (P->next != NULL) {
		front = P;
		P = P->next;
		int num = P->L.size();
		if (num == 1) {
			size_one.push_back(P->L.front());
			front->next = P->next;
			P = front;
		}

	}

	//分配给并行数小于max_parallel的节点
	P = head;
	int size;
	while (P->next != NULL) {
		P = P->next;
		int num = P->L.size();
		size = size_one.size();
		if (size > 0) {
			for (int i = num; i < max_parallel; i++) {
				//找冲突
				find_conflict(B_conf, P, Block, size_one.front());
				P->L.push_back(size_one.front());
				size_one.pop_front();
				
			}			
		}
		size = size_one.size();
		if (size == 0)
			break;
	}

	if (size > 0) {
		while (P->next != NULL) {
			Parallel* tmp_p = new Parallel();
			tmp_p->L.push_back(size_one.back());
			size_one.pop_back();
			tmp_p->next = NULL;
			P->next = tmp_p;
			list<int>::iterator it = P->L.begin();
			for (it; it != P->L.end(); it++) {
				B_conf[*it].coe_x = 1;
				B_conf[*it].coe_y = 1;
				B_conf[*it].coe_z = 1;
			}
			P = P->next;
		}
	}
	return head;
}

/*
int compare_tree(vector<bool> id, vector<bool> x, int level_tree, int level_block, int block_id) {
	if (level_tree < level_block && block_id == -1) {
		return 1;
	}	
	else {
		for (int block_id = 1; block_id < level_tree; block_id++) {
			if (x[block_id] ^ id[block_id]) {   //异或  不同为1，相同为0
				return 1;   //返回1为不冲突,可以并行
			}
		}
	}
	return 0;
}
*/
int compare_tree(BTnode* BT, BS_node Block) {
	int flag = 0;
	int x_level = BT->x.size() < Block.x_id.size() ? BT->x.size() : Block.x_id.size();
	int y_level = BT->y.size() < Block.y_id.size() ? BT->y.size() : Block.y_id.size();
	int z_level = BT->z.size() < Block.z_id.size() ? BT->z.size() : Block.z_id.size();

	if (BT->block != -1) {
		for (int block_id = 1; block_id < x_level; block_id++) {
			if (BT->x[block_id] ^ Block.x_id[block_id]) {   //异或  不同为1，相同为0
				flag = 1;   //返回1为不冲突,可以并行
				break;
			}
		}
		if (flag == 0)
			return 0;

		flag = 0;
		for (int block_id = 1; block_id < y_level; block_id++) {
			if (BT->y[block_id] ^ Block.y_id[block_id]) {   //异或  不同为1，相同为0
				flag = 1;   //返回1为不冲突,可以并行
				break;
			}
		}
		if (flag == 0)
			return 0;

		flag = 0;
		for (int block_id = 1; block_id < z_level; block_id++) {
			if (BT->z[block_id] ^ Block.z_id[block_id]) {   //异或  不同为1，相同为0
				flag = 1;   //返回1为不冲突,可以并行
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
			return 1;  //x,y,z的坐标还可以继续向下比较				
		}

		if (BT->x.size() >= Block.x_id.size() ) {    //进不去if flag=0
			for (int block_id = 1; block_id < Block.x_id.size(); block_id++) {
				if (BT->x[block_id] ^ Block.x_id[block_id]) {   //异或  不同为1，相同为0
					flag = 1;
					break;
				}
			}
			if (flag == 0) //id冲突
				return 0;
		}
		
		flag = 0;
		if (BT->y.size() >= Block.y_id.size()) {
			for (int block_id = 1; block_id < Block.y_id.size(); block_id++) {
				if (BT->y[block_id] ^ Block.y_id[block_id]) {   //异或  不同为1，相同为0
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
				if (BT->z[block_id] ^ Block.z_id[block_id]) {   //异或  不同为1，相同为0
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
	
	/*
	if (level_tree < level_block && block_id == -1) {
		return 1;
	}
	else {
		for (int block_id = 1; block_id < level_tree; block_id++) {
			if (x[block_id] ^ id[block_id]) {   //异或  不同为1，相同为0
				return 1;   //返回1为不冲突,可以并行
			}
		}
	}
	return 0;*/
}

int tree_conflict(set<int>& block_parallel, BTnode* BT_conflict) {
	if (BT_conflict == NULL)
		return 0;

	if (BT_conflict->block != -1) {
		block_parallel.erase(BT_conflict->block);
	}

	tree_conflict(block_parallel, BT_conflict->lChild);
	tree_conflict(block_parallel, BT_conflict->rChild);

}

//找并行块，树方法
Parallel* search_parallel_block_Tree(BS_node* Block, Node_conflict* B_conf, int block_num, int& num_parallel, BTnode* BT) {
	Parallel* Parallel_list = new Parallel();
	Parallel* Parallel_head = new Parallel();
	Parallel_head = Parallel_list;
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
		
	srand(static_cast<unsigned int>(time(nullptr)));
	for (int i = 0; i < block_num; i++) {
		block_id.insert(i);
	}

	while (block_id.size() != 0) {
		set<int> block_parallel;

		auto it = block_id.begin();
		advance(it, rand() % block_id.size());   //随机选取block_id
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
			while (!q.empty()) {   //广度优先遍历树,找到可以并行的一个Block
				if (compare_tree(q.front(), Block[selected]) == 0) {
					BTnode* BT_conflict = new BTnode();
					BT_conflict = q.front();
					tree_conflict(block_parallel, BT_conflict);
					compare_tree(q.front(), Block[selected]);
					//q.front()->block=-
					//q.front()->lChild = q.front()->rChild = NULL;
				}
				else {
					if (q.front()->lChild != NULL)
						q.push(q.front()->lChild);
					if (q.front()->rChild != NULL)
						q.push(q.front()->rChild);
				}
				
				q.pop();
				/*
				int h_tmp = q.front()->h % 3;
				if (h_tmp == 2) {
					if (compare_tree(q.front()->x, Block[selected].x_id, q.front()->x.size(), Block[selected].level_x, q.front()->block) == 0) { //冲突
						BTnode* BT_conflict = new BTnode();
						BT_conflict = q.front();
						tree_conflict(block_parallel, BT_conflict);
						q.front()->lChild = q.front()->rChild = NULL;
					}
				}
				else if (h_tmp == 0) {
					if (compare_tree(q.front()->y, Block[selected].y_id, q.front()->y.size(), Block[selected].level_y, q.front()->block) == 0) {
						BTnode* BT_conflict = new BTnode();
						BT_conflict = q.front();
						tree_conflict(block_parallel, BT_conflict);
						q.front()->lChild = q.front()->rChild = NULL;
					}
				}
				else {
					if (compare_tree(q.front()->z, Block[selected].z_id, q.front()->z.size(), Block[selected].level_z, q.front()->block) == 0) {
						BTnode* BT_conflict = new BTnode();
						BT_conflict = q.front();
						tree_conflict(block_parallel, BT_conflict);
						q.front()->lChild = q.front()->rChild = NULL;
					}
				}*/
			}

			while (block_parallel.size() != 0) {
				auto it_p = block_parallel.begin();
				advance(it_p, rand() % block_parallel.size());   //随机选取block_id
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
			/*
			if (block_parallel.size() != 0) {
				auto it_p = block_parallel.begin();
				advance(it_p, rand() % block_parallel.size());   //随机选取block_id
				selected = *it_p;
				//selected = *block_parallel.begin();
				Parallel_list->L.push_back(selected);
				block_parallel.erase(selected);
				block_id.erase(selected);
			}*/				
		}

		Parallel* Parallel_tmp = new Parallel();
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
		for (it; it != Parallel_list->L.end(); it++) {
			B_conf[*it].coe_x = 1;
			B_conf[*it].coe_y = 1;
			B_conf[*it].coe_z = 1;
		}
	}

	list<int> size_one;
	//取出P中只有一个块的节点
	Parallel_list = Parallel_head;
	Parallel* front;

	while (Parallel_list->next != NULL) {
		front = Parallel_list;
		Parallel_list = Parallel_list->next;
		int num = Parallel_list->L.size();
		if (num == 1) {
			size_one.push_back(Parallel_list->L.front());
			front->next = Parallel_list->next;
			Parallel_list = front;
		}

	}

	//分配给并行数小于max_parallel的节点
	Parallel_list = Parallel_head;
	int size;
	while (Parallel_list->next != NULL) {
		Parallel_list = Parallel_list->next;
		int num = Parallel_list->L.size();
		size = size_one.size();
		if (size > 0) {
			for (int i = num; i < max_parallel; i++) {
				//找冲突
				find_conflict(B_conf, Parallel_list, Block, size_one.front());
				Parallel_list->L.push_back(size_one.front());
				size_one.pop_front();

			}
		}
		size = size_one.size();
		if (size == 0)
			break;
	}

	if (size > 0) {
		while (Parallel_list->next != NULL) {
			Parallel* tmp_p = new Parallel();
			tmp_p->L.push_back(size_one.back());
			size_one.pop_back();
			tmp_p->next = NULL;
			Parallel_list->next = tmp_p;
			list<int>::iterator it = Parallel_list->L.begin();
			for (it; it != Parallel_list->L.end(); it++) {
				B_conf[*it].coe_x = 1;
				B_conf[*it].coe_y = 1;
				B_conf[*it].coe_z = 1;
			}
			Parallel_list = Parallel_list->next;
		}
	}
	/*
	int block_parallel;
	while (block_id.size() != 0) {
		Parallel* Parallel_tmp = new Parallel();

		auto it = block_id.begin();
		advance(it, rand() % block_id.size());   //随机选取block_id
		int selected = *it;
		block_id.erase(selected);
		Parallel_list->L.push_back(selected);

		BTnode* BT_tmp = new BTnode();
		BTnode* BT_tmp_head = new BTnode();
		BT_tmp = BT;          
		BT_tmp_head = BT_tmp;
		int flag_1 = 1;
		int flag_2 = 1;

		BT_tmp = BT_tmp_head;
		while (flag_1 != 0) {  
			q.push(BT_tmp);
			while (!q.empty()) {   //广度优先遍历树,找到可以并行的一个Block
				int h_tmp = q.front()->h % 3;
				if (h_tmp == 2) {
					if (compare_tree(q.front()->x, Block[selected].x_id, q.front()->x.size(), Block[selected].level_x) == 0) { //冲突
						q.front()->lChild = q.front()->rChild = NULL;
						q.front()->block = -1;   //防止最后一个叶子节点也冲突
					}
				}
				else if (h_tmp == 0) {
					if (compare_tree(q.front()->y, Block[selected].y_id, q.front()->y.size(), Block[selected].level_y) == 0) {
						q.front()->lChild = q.front()->rChild = NULL;
						q.front()->block = -1;
					}
				}
				else {
					if (compare_tree(q.front()->z, Block[selected].z_id, q.front()->z.size(), Block[selected].level_z) == 0) {
						q.front()->lChild = q.front()->rChild = NULL;
						q.front()->block = -1;
					}
				}

				if (q.front()->block != -1) {	//可以并行的Block
					selected = q.front()->block;					
					Parallel_list->L.push_back(selected);
					block_id.erase(selected);
					flag_2 = 0;
					break;
				}
				
				if (q.front()->lChild != NULL)
					q.push(q.front()->lChild);
				if (q.front()->rChild != NULL)
					q.push(q.front()->rChild);

				q.pop();
			}

			if (q.size() == 0 && flag_2 == 1) {
				flag_1 = 0;
				Parallel* Parallel_tmp = new Parallel();
				Parallel_list->next = Parallel_tmp;
				int num_p = Parallel_list->L.size();
				num_parallel = num_parallel > num_p ? num_parallel : num_p;
				Parallel_tmp = Parallel_tmp->next;
			}
				

			queue<BTnode*> empty;
			swap(q, empty);
		}

		
	}
*/

	return Parallel_head;
}


Parallel* random_Block(BS_node* Block, int num_block, Node_conflict* B_conf, int& num_parallel) {
	Parallel* Parallel_head = new Parallel();
	Parallel* Parallel_list = new Parallel();
	Parallel_head = Parallel_list;
	set<int> block_id;
	num_parallel = 1;
	srand(static_cast<unsigned int>(time(nullptr)));
	for (int i = 0; i < num_block; i++) {
		block_id.insert(i);
	}
	while (block_id.size() != 0) {
		int num = max_num;
		while (num--) {
			if (block_id.size() == 0)
				break;
			auto it = block_id.begin();
			advance(it, rand() % block_id.size());   //随机选取block_id
			int selected = *it;
			block_id.erase(selected);
			Parallel_list->L.push_back(selected);
		}
		num_parallel++;
		list<int>::iterator it = Parallel_list->L.begin();
		for (it; it != Parallel_list->L.end(); it++) {
			B_conf[*it].coe_x = 1;
			B_conf[*it].coe_y = 1;
			B_conf[*it].coe_z = 1;
		}

		find_conflict_list(B_conf, Parallel_list, Block);

		Parallel* p_tmp = new Parallel();
		Parallel_list->next = p_tmp;
		Parallel_list = Parallel_list->next;

	}
	return Parallel_head;
}

//索引对应的坐标范围
b_node Local(BS_node Block, int I, int J, int K) {
	b_node bs;
	int x_start = 0; int x_end = I;
	int y_start = 0; int y_end = J;
	int z_start = 0; int z_end = K;
	for (int i = 1; i < Block.level_x; i++) {
		int n = (x_start + x_end) / 2;
		if (Block.x_id[i] == 0)
			x_end = n + 1;
		else if (Block.x_id[i] == 1)
			x_start = n + 1;
	}
	for (int i = 1; i < Block.level_y; i++) {
		int n = (y_start + y_end) / 2;
		if (Block.y_id[i] == 0)
			y_end = n + 1;
		else if (Block.y_id[i] == 1)
			y_start = n + 1;
	}
	for (int i = 1; i < Block.level_z; i++) {
		int n = (z_start + z_end) / 2;
		if (Block.z_id[i] == 0)
			z_end = n + 1;
		else if (Block.z_id[i] == 1)
			z_start = n + 1;
	}
	bs.x_start = x_start; bs.x_end = x_end;
	bs.y_start = y_start; bs.y_end = y_end;
	bs.z_start = z_start; bs.z_end = z_end;
	return bs;
}
int ToMatrix(Parallel* P, BS_node* Block, b_node* bs, int* num_bs, int I, int J, int K) {
	int count = 0;
	int size = 0;
	int max_parallel = 0;
	while (P->next != NULL) {

		P = P->next;
		int num = P->L.size();
		max_parallel = max_parallel > num ? max_parallel : num;
		num_bs[size] = num;
		list<int>::iterator it = P->L.begin();
		for (it; it != P->L.end(); it++) {
			b_node bs_t = Local(Block[*it], I, J, K);
			bs[count].x_end = bs_t.x_end;
			bs[count].x_start = bs_t.x_start;
			bs[count].y_end = bs_t.y_end;
			bs[count].y_start = bs_t.y_start;
			bs[count].z_end = bs_t.z_end;
			bs[count].z_start = bs_t.z_start;
			count++;
		}
		size++;
	}
	return max_parallel;
}

/*int find_parallel(BS_node Block, Parallel* a, int& max) {

	int num_t = 1;
	Parallel* tmp, * head;
	tmp = new Parallel();
	head = new Parallel();
	a->next = NULL;

	tmp->L.push_back(0);
	tmp->next = NULL;
	a->next = tmp;
	head = a;
	max = 0;

	for (int i = 1; i < Block.block_num; i++) {
		a = head;
		int flag = 0;

		while (a->next != NULL) {
			if (a->L.size() >= thread_size)
				break;

			flag = 0;
			a = a->next;

			int num = a->L.size();
			max = max > num ? max : num;
			list<int>::iterator it = a->L.begin();
			for (it; it != a->L.end(); it++) {
				if (Block.t[i].x == Block.t[*it].x || Block.t[i].y == Block.t[*it].y || Block.t[i].z == Block.t[*it].z) {
					flag = 1;
					break;
				}
			}
			if (flag == 0 && a->L.size() < thread_size) {
				a->L.push_back(i);
				break;
			}
			else {
				flag = 1;
			}

		}
		if (flag == 1) {
			num_t++;
			Parallel* s;
			s = new Parallel();
			s->L.push_back(i);
			s->next = NULL;
			a->next = s;
			a = s;
		}
	}



	a = head;
	while (a->next != NULL) {
		a = a->next;
		if (a->L.size() < thread_size) {

			for(int i=0;i<Block.block_num;i++){

				if (a->L.size() >= thread_size)
					break;
				int flag = 0;
				list<int>::iterator it = a->L.begin();
				for (it; it != a->L.end(); it++) {
					flag = 0;
					if (Block.t[i].x == Block.t[*it].x || Block.t[i].y == Block.t[*it].y || Block.t[i].z == Block.t[*it].z) {
						flag = 1;
						break;
					}
				}
				if (flag == 0 && a->L.size() < thread_size) {
					a->L.push_back(i);
				}
			}
		}
		if (a->L.size() < max) {
			while (a->L.size() < max) {
				a->L.push_back(-1);
			}
		}
	}

	return num_t;
}

int Preproccess_1(Parallel* P, BS_node* Block, double* t, LF_node* pre, int* b_s, t_node* t_s, int* t_block_num, pre_node* tmp, int max_parallel, int num_parallel, int I, int J, int K) {
	Parallel* head_p = new Parallel();
	head_p = P;

	t_block_num[0] = 0;
	int count = 0;
	int count_b = 1;
	int total = 0;
	int sum = 0;


	while (P->next != NULL) {
		P = P->next;

		b_s[count] = P->L.size();

		list<int>::iterator it = P->L.begin();
		for (it; it != P->L.end(); it++) {

			pre_node* tmp_s = new pre_node();
			tmp_s->next = NULL;
			Parallel* a = new Parallel();

			int max_a;
			int num_t = find_parallel(Block[*it], a, max_a);

			t_s[sum].num_b = num_t;
			t_s[sum].num_p = max_a;
			total += num_t * max_a;
			//t_block_p[count] += num_t * max_a;
			tmp_s->t = new LF_node[num_t * max_a];

			t_block_num[count_b] = t_block_num[count_b - 1] + num_t * max_a;
			int count_a = 0;

			while (a->next != NULL) {
				a = a->next;
				list<int>::iterator it_a = a->L.begin();
				for (it_a; it_a != a->L.end(); it_a++) {
					if (*it_a >= 0) {
					tmp_s->t[count_a].x= Block[*it].t[*it_a].x;
					tmp_s->t[count_a].y = Block[*it].t[*it_a].y;
					tmp_s->t[count_a].z = Block[*it].t[*it_a].z;
					tmp_s->t[count_a].rate = t[Block[*it].t[*it_a].x * J * K + Block[*it].t[*it_a].y * K + Block[*it].t[*it_a].z];
					count_a++;
					}
					else {
						tmp_s->t[count_a].x = 0;
						tmp_s->t[count_a].y = 0;
						tmp_s->t[count_a].z = 0;
						tmp_s->t[count_a].rate = 0;
						count_a++;
					}
				}
			}
			sum++;
			count_b++;
			tmp_s->num_t = count_a;
			tmp->next = tmp_s;
			tmp = tmp_s;
		}

		count++;

	}

	return total;
}
void P_pre(pre_node* tmp, LF_node* pre) {
	int sum = 0;

	while (tmp->next != NULL) {
		tmp = tmp->next;
		for (int i = 0; i < tmp->num_t; i++) {
			pre[sum].x = tmp->t[i].x;
			pre[sum].y = tmp->t[i].y;
			pre[sum].z = tmp->t[i].z;
			pre[sum].rate = tmp->t[i].rate;
			sum++;
		}

	}
}
*/
//找一个Block中可并行,返回list数量
int find_parallel(BS_node Block, pre_node* S, double* t, double coe_x, double coe_y, double coe_z) {
	pre_node* tmp_s = new pre_node();
	Parallel* b = new Parallel();
	Parallel* tmp_b = new Parallel();
	pre_node* head_s = new pre_node();
	head_s = S;
	Parallel* head;
	head = new Parallel();
	head = b;

	tmp_b->L.push_back(0);
	tmp_b->next = NULL;
	b->next = tmp_b;
	if (S->next == NULL) {
		tmp_s->x.push_back(Block.t[0].x);
		tmp_s->y.push_back(Block.t[0].y);
		tmp_s->z.push_back(Block.t[0].z);
		tmp_s->rate.push_back(t[Block.t[0].x * J * K + Block.t[0].y * K + Block.t[0].z]);
		tmp_s->coe_x.push_back(coe_x);
		tmp_s->coe_y.push_back(coe_y);
		tmp_s->coe_z.push_back(coe_z);
		tmp_s->next = NULL;
		S->next = tmp_s;
	}
	else {
		S->next->x.push_back(Block.t[0].x);
		S->next->y.push_back(Block.t[0].y);
		S->next->z.push_back(Block.t[0].z);
		S->next->rate.push_back(t[Block.t[0].x * J * K + Block.t[0].y * K + Block.t[0].z]);
		S->next->coe_x.push_back(coe_x);
		S->next->coe_y.push_back(coe_y);
		S->next->coe_z.push_back(coe_z);
	}


	int num_t = 1;
	int count_a = 0;
	for (int i = 1; i < Block.block_num; i++) {
		b = head;
		S = head_s;
		int flag = 0;

		while (b->next != NULL) {
			if (b->L.size() >= thread_size)//最大不超过线程数
				break;

			flag = 0;
			b = b->next;
			S = S->next;

			int num = b->L.size();
			list<int>::iterator it = b->L.begin();
			for (it; it != b->L.end(); it++) {//找到不同x,y,z的数放在一个list里
				if (Block.t[i].x == Block.t[*it].x || Block.t[i].y == Block.t[*it].y || Block.t[i].z == Block.t[*it].z) {
					flag = 1;
					break;
				}
			}
			if (flag == 0 && b->L.size() < thread_size) {
				b->L.push_back(i);

				S->x.push_back(Block.t[i].x);
				S->y.push_back(Block.t[i].y);
				S->z.push_back(Block.t[i].z);
				S->rate.push_back(t[Block.t[i].x * J * K + Block.t[i].y * K + Block.t[i].z]);
				S->coe_x.push_back(coe_x);
				S->coe_y.push_back(coe_y);
				S->coe_z.push_back(coe_z);

				break;
			}
			else {
				flag = 1;
			}

		}
		if (flag == 1) {
			num_t++;
			Parallel* s;
			s = new Parallel();
			s->L.push_back(i);
			s->next = NULL;
			b->next = s;
			b = s;

			if (S->next == NULL) {
				pre_node* tmp = new pre_node();
				tmp->x.push_back(Block.t[i].x);
				tmp->y.push_back(Block.t[i].y);
				tmp->z.push_back(Block.t[i].z);
				tmp->rate.push_back(t[Block.t[i].x * J * K + Block.t[i].y * K + Block.t[i].z]);
				tmp->coe_x.push_back(coe_x);
				tmp->coe_y.push_back(coe_y);
				tmp->coe_z.push_back(coe_z);
				tmp->next = NULL;
				S->next = tmp;
				S = tmp;
			}
			else {
				S = S->next;
				S->x.push_back(Block.t[i].x);
				S->y.push_back(Block.t[i].y);
				S->z.push_back(Block.t[i].z);
				S->rate.push_back(t[Block.t[i].x * J * K + Block.t[i].y * K + Block.t[i].z]);
				S->coe_x.push_back(coe_x);
				S->coe_y.push_back(coe_y);
				S->coe_z.push_back(coe_z);
			}

		}
	}
	Free_list(b);
	return num_t;
}
void Preproccess(Parallel* P, BS_node* Block, Node_conflict* B_conf, LF_node* pre, double* t, list<int>& num_parallel_p) {
	int count = 0;

	while (P->next != NULL) {
		P = P->next;
		int num = P->L.size();

		pre_node* S = new pre_node();
		S->next = NULL;
		pre_node* head_t = new pre_node;
		head_t = S;
		int max = 0;

		list<int>::iterator it = P->L.begin();
		for (it; it != P->L.end(); it++) {//找出并行block中所有可以并行的
			S = head_t;

			if (Block[*it].block_num == 0)
				continue;

			double coe_x = B_conf[*it].coe_x;
			double coe_y = B_conf[*it].coe_y;
			double coe_z = B_conf[*it].coe_z;
			int num_t = find_parallel(Block[*it], S, t, coe_x, coe_y, coe_z);
			max = max > num_t ? max : num_t;

		}

		while (S->next != NULL) {//所有数放在一维数组中，记录控制并行的数
			S = S->next;
			num_parallel_p.push_back(S->x.size());

			while (S->x.size() != 0) {
				pre[count].x = S->x.front();
				pre[count].y = S->y.front();
				pre[count].z = S->z.front();
				pre[count].rate = S->rate.front();
				pre[count].coe_x = S->coe_x.front();
				pre[count].coe_y = S->coe_y.front();
				pre[count].coe_z = S->coe_z.front();
				S->x.pop_front();
				S->y.pop_front();
				S->z.pop_front();
				S->rate.pop_front();
				S->coe_x.pop_front();
				S->coe_y.pop_front();
				S->coe_z.pop_front();

				count++;
			}

		}

	}

}


int Preproccess_list(Parallel* P, BS_node* Block, b_node* bs, int* num_bs, int I, int J, int K) {
	int count = 0;
	int size = 0;
	int max_parallel = 0;
	while (P->next != NULL) {

		
		int num = P->L.size();
		max_parallel = max_parallel > num ? max_parallel : num;
		num_bs[size] = num;
		list<int>::iterator it = P->L.begin();
		for (it; it != P->L.end(); it++) {
			b_node bs_t = Local(Block[*it], I, J, K);
			bs[count].x_end = bs_t.x_end;
			bs[count].x_start = bs_t.x_start;
			bs[count].y_end = bs_t.y_end;
			bs[count].y_start = bs_t.y_start;
			bs[count].z_end = bs_t.z_end;
			bs[count].z_start = bs_t.z_start;
			bs[count].id = *it;
			count++;
		}
		size++;
		P = P->next;
	}
	return max_parallel;

}

//lock_free预处理
void ToMatrix_LF(Parallel* P, BS_node* Block, double* t, LF_node* LF, int* num_LS, int I, int J, int K) {
	int count = 0;
	int size = 0;
	int max_parallel = 0;
	while (P->next != NULL) {
		P = P->next;
		int num = 0;
		list<int>::iterator it = P->L.begin();
		for (it; it != P->L.end(); it++) {
			for (int i = 0; i < Block[*it].block_num; i++) {
				int x_b, y_b, z_b;
				x_b = Block[*it].t[i].x;
				y_b = Block[*it].t[i].y;
				z_b = Block[*it].t[i].z;
				LF[count].rate = t[x_b * J * K + y_b * K + z_b];
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

//初始化Block
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
	double* t_1,
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
	double rate) {

	ofstream ofs_time, ofs_block, ofs_blockdetail, ofs_bnum;
	string address("C:/Users/12625/Desktop/tensor/A_");
	string str;
	stringstream ss;
	ss << rate;
	ss >> str;
	str += "/";
	address += str;
	string add_error, add_time, add_block, add_blockdetail;
	add_error += address;
	add_error += "error.txt";
	add_time += address;
	add_time += "time.txt";
	add_block += address;
	add_block += "block.txt";
	add_blockdetail += address;
	add_blockdetail += "detail.txt";
	T_node* t_block = new T_node[nnz];
	BS_node* Block;
	Parallel* P, * head;
	Node_conflict* B_conf;

	BTnode* BT = new BTnode();
	BTnode* BT_head = new BTnode(); 
	BT_head = BT;

	Block = new BS_node[nnz];    //数据块
	Block->t = new T_node[block_s];

	P = new Parallel();
	head = new Parallel();

	initial_B(Block);
	initial_t(t_block, t, I, J, K, nnz);

	ofs_blockdetail.open(add_blockdetail, ios::out | ios::in | ios::trunc);
	//ofs_time.open(add_time, ios::out | ios::in | ios::trunc);
	double td1 = get_time();
	num_block = tensor_block(Block, t_block, nnz, I, J, K, BT, BT_head);                   //数据块划分
	td1 = get_time() - td1;
	ofs_blockdetail<< "tensor_block=" << num_block << endl;
	//ofs_time << "tensor_block=" << td1 << endl;
	//ofs_time.close();

	BT = BT_head;
	B_conf = new Node_conflict[num_block];
	//head = random_Block(Block, num_block, B_conf, num_parallel);
	head = search_parallel_block_Tree(Block, B_conf, num_block, num_parallel, BT);
	//head = search_parallel_block(Block, B_conf, num_block, num_parallel);
	P = head;

	/*
	int *num_bs = new int[num_parallel];
	b_node* bs = new b_node[nnz];
	if (flag_preproccess == 1) {
		max_parallel = Preproccess_list(P, Block, bs, num_bs, I, J, K);
	}
*/
	pre = new LF_node[nnz];
	list<int> num_parallel_p;
	int num_parallel_t = 0;
	if (flag_preproccess == 1) {
	//	Preproccess(P, Block, B_conf, pre, t, num_parallel_p);
	}
	int sum = 0;
	int num_pre = num_parallel_p.size();
	num_parallel_pre = new int[num_pre];
	list<int>::iterator it = num_parallel_p.begin();
	for (it; it != num_parallel_p.end(); it++) {
		num_parallel_pre[sum] = *it;
		sum++;
	}
	num_parallel_t = num_pre;

	LF = new LF_node[nnz];
	num_LF = new int[num_parallel];
	if (flag_lockfree == 1) {
		ToMatrix_LF(head, Block, t, LF, num_LF, I, J, K);
	}


	//ofs_bnum.open("C:/Users/12625/Desktop/tensor/block_num.txt", ios::out | ios::in | ios::app);
	
	ofs_block.open(add_block, ios::out | ios::in | ios::trunc);
	//ofs_block << num_block << endl;
	ofs_blockdetail << num_block << endl;
//	ofs_bnum << num_block << endl;
	for (int i = 0; i < num_block; i++) {
		ofs_blockdetail << "BID" << Block[i].id << ":";

		for (int num_1 = 0; num_1 < Block[i].level_x; num_1++)
			ofs_blockdetail << Block[i].x_id[num_1];
		ofs_blockdetail << " ";
		for (int num_2 = 0; num_2 < Block[i].level_y; num_2++)
			ofs_blockdetail << Block[i].y_id[num_2];
		ofs_blockdetail << " ";
		for (int num_3 = 0; num_3 < Block[i].level_z; num_3++)
			ofs_blockdetail << Block[i].z_id[num_3];
		ofs_blockdetail << " ";
		ofs_blockdetail << Block[i].block_num << endl;
		ofs_block << Block[i].block_num << endl;
		//	ofs_block << 0 << ":" << Block[i].t[0].x << " " << Block[i].t[0].y << " " << Block[i].t[0].z << "  ";
		//	ofs_block << 1 << ":" << Block[i].t[1].x << " " << Block[i].t[1].y << " " << Block[i].t[1].z << "  ";
	}
	while (P->next != NULL) {
		P = P->next;
		list<int>::iterator it = P->L.begin();
		int n = P->L.size();
		for (it; it != P->L.end(); it++) {
			ofs_blockdetail << Block[*it].id << " ";
		}
		ofs_blockdetail << endl;
	}
	ofs_blockdetail << endl;

	ofs_block.close();
	ofs_blockdetail.close();
	//ofs_bnum.close();

	P = head;
	//delete[]t_block;
	//delete Block;
	Free_list(P);
	//	delete P;
	//	delete head;
	/*	double* a_t = new double[I * r];
		double* b_t = new double[J * r];
		double* c_t = new double[K * r];
		a_t = a;
		b_t = b;
		c_t = c;
	*/
	sgd_train(t_1, t, a, b, c, pre, num_parallel_pre, num_parallel_t, num_block, num_parallel, max_parallel, LF, num_LF, nnz, rate, num_bs, bs, B_conf);
	//	compare_train(rate, a_t, b_t, c_t, num_block, t, t_1,max,min);

}