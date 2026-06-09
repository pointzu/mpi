#pragma once

#include "matrix.h"

#include <string>

struct GkhRunStats
{
    std::string mode = "serial";
    int configured_threads = 1;
    int active_threads = 1;
    int iterations = 0;
    long long total_blocks = 0;
    long long nontrivial_blocks = 0;
    long long one_block_steps = 0;
    long long work_units = 0;
    long long max_thread_work = 0;
    long long min_active_thread_work = 0;
    int rounds_with_work = 0;
    int parallel_rounds = 0;
    int max_ready_tasks = 0;
    int mpi_world_size = 1;
    long long mpi_tasks_sent = 0;
    long long mpi_matrix_broadcast_bytes = 0;
    long long mpi_result_bytes = 0;
};

// 在已上二对角化结果 A = U * B * V^T 上执行 GKH 迭代。
// 额外处理“主对角线收敛到 0”的压缩情形。
//
// 输入要求：
// - B 为 m x n 且 m >= n，且近似上二对角
// - U 为 m x m，V 为 n x n
//
// 输出：
// - 保持 A = U * B * V^T
// - 若收敛，B 变为非负降序对角（m x n）
//
// 返回：是否在 max_iter 轮内收敛。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V,
                             int max_iter = 6000,
                             double tol = 1e-12);

// 返回最近一次 GKH 运行的线程调度统计，用于报告中的负载均衡分析。
const GkhRunStats &gkh_last_stats();
