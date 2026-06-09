#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <limits>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(SVD_DISABLE_SIMD) && defined(__AVX__)
#include <immintrin.h>
#elif !defined(SVD_DISABLE_SIMD) && defined(__SSE2__)
#include <emmintrin.h>
#elif !defined(SVD_DISABLE_SIMD) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVD_USE_MPI
#include <mpi.h>
#endif

namespace
{

    enum class GkhMode
    {
        Serial,
        Pthread,
        Openmp,
        Mpi
    };

    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，你可以认为通过这个抽象结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };

    GkhRunStats last_stats;

    static long long block_work(const Block &blk)
    {
        const long long len = static_cast<long long>(blk.r - blk.l + 1);
        return len * len;
    }

    static const char *mode_name(GkhMode mode)
    {
        switch (mode)
        {
        case GkhMode::Serial:
            return "serial";
        case GkhMode::Pthread:
            return "pthread";
        case GkhMode::Openmp:
            return "openmp";
        case GkhMode::Mpi:
            return "mpi";
        }
        return "serial";
    }

    static std::string env_string(const char *name)
    {
        const char *value = std::getenv(name);
        return value ? std::string(value) : std::string();
    }

    static GkhMode read_mode()
    {
        std::string mode = env_string("SVD_MODE");
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });

        if (mode == "serial")
        {
            return GkhMode::Serial;
        }
        if (mode == "openmp" || mode == "omp")
        {
            return GkhMode::Openmp;
        }
        if (mode == "mpi")
        {
            return GkhMode::Mpi;
        }
        if (mode == "pthread")
        {
            return GkhMode::Pthread;
        }
#ifdef SVD_USE_MPI
        return GkhMode::Mpi;
#else
        return GkhMode::Pthread;
#endif
    }

    static int parse_positive_int(const std::string &text)
    {
        if (text.empty())
        {
            return 0;
        }
        char *end = nullptr;
        const long value = std::strtol(text.c_str(), &end, 10);
        if (end == text.c_str() || *end != '\0' || value <= 0)
        {
            return 0;
        }
        return static_cast<int>(std::min<long>(value, 256));
    }

    static int count_pbs_threads()
    {
        const std::string nodefile = env_string("PBS_NODEFILE");
        if (nodefile.empty())
        {
            return 0;
        }

        std::ifstream in(nodefile);
        int count = 0;
        std::string line;
        while (std::getline(in, line))
        {
            if (!line.empty())
            {
                ++count;
            }
        }
        return count;
    }

    static int read_thread_count(GkhMode mode)
    {
        if (mode == GkhMode::Serial)
        {
            return 1;
        }
        if (mode == GkhMode::Mpi)
        {
#ifdef SVD_USE_MPI
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (initialized)
            {
                int world_size = 1;
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);
                return std::max(1, world_size);
            }
#endif
            return 1;
        }

        int threads = parse_positive_int(env_string("SVD_THREADS"));
        if (threads <= 0)
        {
            threads = count_pbs_threads();
        }
        if (threads <= 0)
        {
            const unsigned int hw = std::thread::hardware_concurrency();
            threads = hw > 0 ? static_cast<int>(hw) : 1;
        }

        return std::max(1, std::min(threads, 64));
    }

    static void finalize_stats(GkhRunStats &stats, const std::vector<long long> &thread_work)
    {
        stats.active_threads = 0;
        stats.max_thread_work = 0;
        stats.min_active_thread_work = 0;

        for (long long work : thread_work)
        {
            if (work <= 0)
            {
                continue;
            }
            ++stats.active_threads;
            stats.max_thread_work = std::max(stats.max_thread_work, work);
            if (stats.min_active_thread_work == 0 || work < stats.min_active_thread_work)
            {
                stats.min_active_thread_work = work;
            }
        }

        if (stats.active_threads == 0)
        {
            stats.active_threads = 1;
        }
    }

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows_range(Matrix &M, int r0, int r1, double c, double s,
                                      int col_begin, int col_end)
    {
        col_begin = std::max(0, col_begin);
        col_end = std::min(M.cols(), col_end);
        double *row0 = M.row_data(r0);
        double *row1 = M.row_data(r1);
        int j = col_begin;

#if !defined(SVD_DISABLE_SIMD) && defined(__AVX__)
        const __m256d vc = _mm256_set1_pd(c);
        const __m256d vs = _mm256_set1_pd(s);
        const __m256d vns = _mm256_set1_pd(-s);
        for (; j + 4 <= col_end; j += 4)
        {
            const __m256d a = _mm256_loadu_pd(row0 + j);
            const __m256d b = _mm256_loadu_pd(row1 + j);
            _mm256_storeu_pd(row0 + j, _mm256_add_pd(_mm256_mul_pd(vc, a), _mm256_mul_pd(vs, b)));
            _mm256_storeu_pd(row1 + j, _mm256_add_pd(_mm256_mul_pd(vns, a), _mm256_mul_pd(vc, b)));
        }
#elif !defined(SVD_DISABLE_SIMD) && defined(__SSE2__)
        const __m128d vc = _mm_set1_pd(c);
        const __m128d vs = _mm_set1_pd(s);
        const __m128d vns = _mm_set1_pd(-s);
        for (; j + 2 <= col_end; j += 2)
        {
            const __m128d a = _mm_loadu_pd(row0 + j);
            const __m128d b = _mm_loadu_pd(row1 + j);
            _mm_storeu_pd(row0 + j, _mm_add_pd(_mm_mul_pd(vc, a), _mm_mul_pd(vs, b)));
            _mm_storeu_pd(row1 + j, _mm_add_pd(_mm_mul_pd(vns, a), _mm_mul_pd(vc, b)));
        }
#elif !defined(SVD_DISABLE_SIMD) && defined(__aarch64__)
        const float64x2_t vc = vdupq_n_f64(c);
        const float64x2_t vs = vdupq_n_f64(s);
        const float64x2_t vns = vdupq_n_f64(-s);
        for (; j + 2 <= col_end; j += 2)
        {
            const float64x2_t a = vld1q_f64(row0 + j);
            const float64x2_t b = vld1q_f64(row1 + j);
            vst1q_f64(row0 + j, vaddq_f64(vmulq_f64(vc, a), vmulq_f64(vs, b)));
            vst1q_f64(row1 + j, vaddq_f64(vmulq_f64(vns, a), vmulq_f64(vc, b)));
        }
#endif

        for (; j < col_end; ++j)
        {
            const double a = row0[j];
            const double b = row1[j];
            row0[j] = c * a + s * b;
            row1[j] = -s * a + c * b;
        }
    }

    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        apply_left_rows_range(M, r0, r1, c, s, 0, M.cols());
    }

    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    // 即 M <- M * R，其中 R 只作用在第 c0/c1 两列上。
    static void apply_right_cols_range(Matrix &M, int c0, int c1, double c, double s,
                                       int row_begin, int row_end)
    {
        row_begin = std::max(0, row_begin);
        row_end = std::min(M.rows(), row_end);
        for (int i = row_begin; i < row_end; ++i)
        {
            double a = M.at(i, c0);
            double b = M.at(i, c1);
            M.at(i, c0) = a * c - b * s;
            M.at(i, c1) = a * s + b * c;
        }
    }

    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        apply_right_cols_range(M, c0, c1, c, s, 0, M.rows());
    }

    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们B <- L * B 时，我们必须维护的等式是 A = U * B * V^T
        // 如果 A = U * B * V^T 不成立，那么我们最终的SVD结果显然不是 A 的正确分解。
        // 由于正交矩阵和其转置的乘积是I，一个自然的想法是让 U <- U * L^T。
        // 这样就变成 A = (U * L^T) * (L * B) * V^T = U * B * V^T，等式得以保持。

        // 由于 L^T = [c -s; s c]，此处复用“右乘两列”接口并传入 -s。
        apply_right_cols(U, r0, r1, c, -s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // 偏移用于加速 QR 迭代收敛，并让 bulge chasing 过程更稳定。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 将上二对角结构以外、且绝对值很小的元素强制置零。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    // 对活动块 [l, r] 执行一次“单块 GKH bulge chasing”迭代。
    // 流程：首次右乘引入 bulge -> 首次左乘消 bulge -> 交替右乘/左乘将 bulge 追赶到块末端。
    static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘：由 (d_l^2-mu, d_l*e_l) 构造。
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols_range(B, l, l + 1, c, s, l, r + 1);
        apply_right_cols(V, l, l + 1, c, s);

        // 首次左乘：消去 (l+1, l)。
        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows_range(B, l, l + 1, c, s, l, r + 1);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘：消去 (k-1, k+1)
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols_range(B, k, k + 1, c, s, l, r + 1);
            apply_right_cols(V, k, k + 1, c, s);

            // 左乘：消去 (k+1, k)
            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows_range(B, k, k + 1, c, s, l, r + 1);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    static std::vector<Block> collect_nontrivial_blocks(const std::vector<Block> &blocks)
    {
        std::vector<Block> tasks;
        tasks.reserve(blocks.size());

        // 保留原串行实现“从右到左”的任务入队顺序；并行版本中任务之间互不依赖。
        for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
        {
            if (blocks[i].r > blocks[i].l)
            {
                tasks.push_back(blocks[i]);
            }
        }

        return tasks;
    }

    static void record_task_batch(const std::vector<Block> &blocks,
                                  const std::vector<Block> &tasks,
                                  bool can_parallel,
                                  GkhRunStats &stats)
    {
        stats.total_blocks += static_cast<long long>(blocks.size());
        stats.nontrivial_blocks += static_cast<long long>(tasks.size());
        stats.one_block_steps += static_cast<long long>(tasks.size());
        stats.max_ready_tasks = std::max(stats.max_ready_tasks, static_cast<int>(tasks.size()));

        if (!tasks.empty())
        {
            ++stats.rounds_with_work;
            if (can_parallel && tasks.size() > 1)
            {
                ++stats.parallel_rounds;
            }
        }

        for (const auto &task : tasks)
        {
            stats.work_units += block_work(task);
        }
    }

    static void process_blocks_serial(Matrix &U, Matrix &B, Matrix &V,
                                      const std::vector<Block> &tasks,
                                      std::vector<long long> &thread_work)
    {
        if (thread_work.empty())
        {
            thread_work.resize(1, 0);
        }

        for (const auto &task : tasks)
        {
            thread_work[0] += block_work(task);
            one_block_step(U, B, V, task.l, task.r);
        }
    }

    class PthreadBlockPool
    {
    public:
        explicit PthreadBlockPool(int thread_count)
        {
            const int worker_count = std::max(0, thread_count - 1);
            workers_.reserve(static_cast<size_t>(worker_count));
            for (int i = 0; i < worker_count; ++i)
            {
                const int slot = i + 1;
                workers_.emplace_back([this, slot]()
                                      { worker_loop(slot); });
            }
        }

        ~PthreadBlockPool()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
                ++generation_;
            }
            cv_start_.notify_all();
            for (auto &worker : workers_)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }
        }

        void process(Matrix &U, Matrix &B, Matrix &V,
                     const std::vector<Block> &tasks,
                     std::vector<long long> &thread_work)
        {
            if (tasks.empty())
            {
                return;
            }

            if (workers_.empty())
            {
                process_blocks_serial(U, B, V, tasks, thread_work);
                return;
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                U_ = &U;
                B_ = &B;
                V_ = &V;
                tasks_ = &tasks;
                thread_work_ = &thread_work;
                next_task_.store(0, std::memory_order_relaxed);
                pending_workers_ = static_cast<int>(workers_.size());
                ++generation_;
            }

            cv_start_.notify_all();
            run_tasks(0);

            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [this]()
                          { return pending_workers_ == 0; });
        }

    private:
        void worker_loop(int slot)
        {
            int seen_generation = 0;
            for (;;)
            {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_start_.wait(lock, [this, seen_generation]()
                                   { return stopping_ || generation_ != seen_generation; });
                    if (stopping_)
                    {
                        return;
                    }
                    seen_generation = generation_;
                }

                run_tasks(slot);

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --pending_workers_;
                    if (pending_workers_ == 0)
                    {
                        cv_done_.notify_one();
                    }
                }
            }
        }

        void run_tasks(int slot)
        {
            for (;;)
            {
                const int idx = next_task_.fetch_add(1, std::memory_order_relaxed);
                if (idx >= static_cast<int>(tasks_->size()))
                {
                    break;
                }

                const Block &task = (*tasks_)[idx];
                (*thread_work_)[slot] += block_work(task);
                one_block_step(*U_, *B_, *V_, task.l, task.r);
            }
        }

        std::vector<std::thread> workers_;
        std::mutex mutex_;
        std::condition_variable cv_start_;
        std::condition_variable cv_done_;
        bool stopping_ = false;
        int generation_ = 0;
        int pending_workers_ = 0;
        std::atomic<int> next_task_{0};
        Matrix *U_ = nullptr;
        Matrix *B_ = nullptr;
        Matrix *V_ = nullptr;
        const std::vector<Block> *tasks_ = nullptr;
        std::vector<long long> *thread_work_ = nullptr;
    };

    static void process_blocks_openmp(Matrix &U, Matrix &B, Matrix &V,
                                      const std::vector<Block> &tasks,
                                      int thread_count,
                                      std::vector<long long> &thread_work)
    {
        if (tasks.empty())
        {
            return;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
        for (int i = 0; i < static_cast<int>(tasks.size()); ++i)
        {
            const int tid = omp_get_thread_num();
            const Block task = tasks[static_cast<size_t>(i)];
            thread_work[static_cast<size_t>(tid)] += block_work(task);
            one_block_step(U, B, V, task.l, task.r);
        }
#else
        process_blocks_serial(U, B, V, tasks, thread_work);
#endif
    }

    // 处理“对角元 d_k 近零但超对角 e_k 未近零”的情况。
    // 思路与单块追赶类似：先右乘把 e_i 消掉，再左乘清理新引入的次对角 bulge，
    // 把这个问题逐步向右传递，直到块末端。
    static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
        // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
        // 然后把问题传递到下一行，直到末端。
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols(B, i, i + 1, c, s);
            apply_right_cols(V, i, i + 1, c, s);

            // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows(B, i, i + 1, c, s);
                accumulate_left_into_U(U, i, i + 1, c, s);
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    // 扫描所有 d_k≈0 的位置：若对应 e_k 仍显著非零，则调用追赶过程压缩该异常结构。
    // 返回值表示本轮是否对 B/U/V 做了实际更新。
    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal(U, B, V, k, tol))
                {
                    changed = true;
                }
            }
        }

        return changed;
    }

    // 根据超对角线是否“足够小”对问题进行分块。
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置可解耦并直接置零。
    // 最终会得到一系列小矩阵。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
    // 最终得到常见的 SVD 规范形式：sigma_1 >= sigma_2 >= ... >= 0。
    // 这个函数你不用太在意，后续任务也不会明确涉及它。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }

#ifdef SVD_USE_MPI
    static constexpr int MPI_CMD_STOP = 0;
    static constexpr int MPI_CMD_DISTRIBUTE = 1;
    static constexpr int MPI_CMD_IDLE_SERIAL = 2;
    static constexpr int MPI_TAG_TASK = 4101;
    static constexpr int MPI_TAG_RESULT_HEADER = 4102;
    static constexpr int MPI_TAG_RESULT_DATA = 4103;
    static bool mpi_initialized_here = false;
    static bool mpi_worker_stdout_suppressed = false;

    static void finalize_owned_mpi()
    {
        if (!mpi_initialized_here)
        {
            return;
        }

        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized)
        {
            MPI_Finalize();
        }
    }

    static bool mpi_world(int &rank, int &size)
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized)
        {
            rank = 0;
            size = 1;
            return false;
        }

        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int argc = 0;
            char **argv = nullptr;
            MPI_Init(&argc, &argv);
            mpi_initialized_here = true;
            std::atexit(finalize_owned_mpi);
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (rank != 0 && !mpi_worker_stdout_suppressed)
        {
            std::cout.setstate(std::ios_base::badbit);
            mpi_worker_stdout_suppressed = true;
        }
        return true;
    }

    static void mpi_bcast_matrix(Matrix &M)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int dims[2] = {M.rows(), M.cols()};
        MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0)
        {
            M = Matrix(dims[0], dims[1]);
        }

        const int count = M.size();
        if (count > 0)
        {
            MPI_Bcast(M.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    static long long mpi_matrix_broadcast_bytes(const Matrix &U, const Matrix &B, const Matrix &V, int world_size)
    {
        const long long payload = static_cast<long long>(U.size() + B.size() + V.size()) *
                                  static_cast<long long>(sizeof(double));
        return payload * static_cast<long long>(std::max(0, world_size - 1));
    }

    static int mpi_result_count(const Block &task, int m, int n)
    {
        const int len = task.r - task.l + 1;
        return len * len + m * len + n * len;
    }

    static std::vector<double> pack_mpi_result(const Matrix &U, const Matrix &B, const Matrix &V, const Block &task)
    {
        const int len = task.r - task.l + 1;
        std::vector<double> out;
        out.reserve(static_cast<size_t>(mpi_result_count(task, U.rows(), V.rows())));

        for (int i = task.l; i <= task.r; ++i)
        {
            for (int j = task.l; j <= task.r; ++j)
            {
                out.push_back(B.at(i, j));
            }
        }
        for (int i = 0; i < U.rows(); ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                out.push_back(U.at(i, task.l + j));
            }
        }
        for (int i = 0; i < V.rows(); ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                out.push_back(V.at(i, task.l + j));
            }
        }

        return out;
    }

    static void merge_mpi_result(Matrix &U, Matrix &B, Matrix &V, const Block &task,
                                 const std::vector<double> &payload)
    {
        const int len = task.r - task.l + 1;
        size_t pos = 0;

        for (int i = task.l; i <= task.r; ++i)
        {
            for (int j = task.l; j <= task.r; ++j)
            {
                B.at(i, j) = payload[pos++];
            }
        }
        for (int i = 0; i < U.rows(); ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                U.at(i, task.l + j) = payload[pos++];
            }
        }
        for (int i = 0; i < V.rows(); ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                V.at(i, task.l + j) = payload[pos++];
            }
        }
    }

    static void send_mpi_task(int worker, const Block &task)
    {
        int header[2] = {task.l, task.r};
        MPI_Send(header, 2, MPI_INT, worker, MPI_TAG_TASK, MPI_COMM_WORLD);
    }

    static void send_mpi_stop(int worker)
    {
        int header[2] = {-1, -1};
        MPI_Send(header, 2, MPI_INT, worker, MPI_TAG_TASK, MPI_COMM_WORLD);
    }

    static void mpi_worker_batch(Matrix &U, Matrix &B, Matrix &V)
    {
        int task_count = 0;
        MPI_Bcast(&task_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        for (;;)
        {
            int header[2] = {-1, -1};
            MPI_Recv(header, 2, MPI_INT, 0, MPI_TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (header[0] < 0)
            {
                break;
            }

            const Block task{header[0], header[1]};
            one_block_step(U, B, V, task.l, task.r);

            MPI_Send(header, 2, MPI_INT, 0, MPI_TAG_RESULT_HEADER, MPI_COMM_WORLD);
            std::vector<double> payload = pack_mpi_result(U, B, V, task);
            MPI_Send(payload.data(), static_cast<int>(payload.size()), MPI_DOUBLE, 0, MPI_TAG_RESULT_DATA, MPI_COMM_WORLD);
        }
    }

    static void process_blocks_mpi_root(Matrix &U, Matrix &B, Matrix &V,
                                        std::vector<Block> tasks,
                                        int world_size,
                                        GkhRunStats &stats,
                                        std::vector<long long> &rank_work)
    {
        if (tasks.empty())
        {
            return;
        }

        std::stable_sort(tasks.begin(), tasks.end(), [](const Block &a, const Block &b)
                         { return block_work(a) > block_work(b); });

        int task_count = static_cast<int>(tasks.size());
        MPI_Bcast(&task_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int next_task = 0;
        int active_workers = 0;
        const Block root_task = tasks[static_cast<size_t>(next_task++)];

        for (int worker = 1; worker < world_size; ++worker)
        {
            if (next_task < task_count)
            {
                send_mpi_task(worker, tasks[static_cast<size_t>(next_task++)]);
                ++active_workers;
                ++stats.mpi_tasks_sent;
            }
            else
            {
                send_mpi_stop(worker);
            }
        }

        std::vector<Block> root_tasks{root_task};
        process_blocks_serial(U, B, V, root_tasks, rank_work);

        while (active_workers > 0)
        {
            int header[2] = {-1, -1};
            MPI_Status status;
            MPI_Recv(header, 2, MPI_INT, MPI_ANY_SOURCE, MPI_TAG_RESULT_HEADER, MPI_COMM_WORLD, &status);
            const int worker = status.MPI_SOURCE;
            const Block task{header[0], header[1]};

            std::vector<double> payload(static_cast<size_t>(mpi_result_count(task, U.rows(), V.rows())));
            MPI_Recv(payload.data(), static_cast<int>(payload.size()), MPI_DOUBLE,
                     worker, MPI_TAG_RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            merge_mpi_result(U, B, V, task, payload);
            rank_work[static_cast<size_t>(worker)] += block_work(task);
            stats.mpi_result_bytes += static_cast<long long>(payload.size()) *
                                      static_cast<long long>(sizeof(double));

            --active_workers;
            if (next_task < task_count)
            {
                send_mpi_task(worker, tasks[static_cast<size_t>(next_task++)]);
                ++active_workers;
                ++stats.mpi_tasks_sent;
            }
            else
            {
                send_mpi_stop(worker);
            }
        }
    }

    static bool gkh_svd_from_bidiagonal_mpi(Matrix &U, Matrix &B, Matrix &V,
                                           int max_iter, double tol,
                                           int mpi_rank, int mpi_size)
    {
        const int n = B.cols();

        bool converged = false;
        std::vector<long long> rank_work(static_cast<size_t>(std::max(1, mpi_size)), 0);

        if (mpi_rank == 0)
        {
            last_stats = GkhRunStats{};
            last_stats.mode = mode_name(GkhMode::Mpi);
            last_stats.configured_threads = mpi_size;
            last_stats.mpi_world_size = mpi_size;

            for (int iter = 0; iter < max_iter; ++iter)
            {
                last_stats.iterations = iter + 1;

                cleanup_bidiagonal(B, tol);
                handle_diagonal_zeros(U, B, V, tol);

                std::vector<Block> blocks = split_active_blocks(B, n, tol);
                std::vector<Block> tasks = collect_nontrivial_blocks(blocks);
                const bool can_parallel = (mpi_size > 1 && tasks.size() > 1);
                record_task_batch(blocks, tasks, can_parallel, last_stats);

                if (tasks.empty())
                {
                    converged = true;
                    break;
                }

                if (!can_parallel)
                {
                    int command = MPI_CMD_IDLE_SERIAL;
                    MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    process_blocks_serial(U, B, V, tasks, rank_work);
                    continue;
                }

                int command = MPI_CMD_DISTRIBUTE;
                MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD);
                mpi_bcast_matrix(U);
                mpi_bcast_matrix(B);
                mpi_bcast_matrix(V);
                last_stats.mpi_matrix_broadcast_bytes += mpi_matrix_broadcast_bytes(U, B, V, mpi_size);
                process_blocks_mpi_root(U, B, V, tasks, mpi_size, last_stats, rank_work);
            }

            int command = MPI_CMD_STOP;
            MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD);

            cleanup_bidiagonal(B, tol);
            for (int i = 0; i < n - 1; ++i)
            {
                B.at(i, i + 1) = 0.0;
            }
            make_nonnegative_and_sort(U, B, V);
            finalize_stats(last_stats, rank_work);
        }
        else
        {
            for (;;)
            {
                int command = MPI_CMD_STOP;
                MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (command == MPI_CMD_STOP)
                {
                    break;
                }
                if (command == MPI_CMD_IDLE_SERIAL)
                {
                    continue;
                }

                mpi_bcast_matrix(U);
                mpi_bcast_matrix(B);
                mpi_bcast_matrix(V);
                mpi_worker_batch(U, B, V);
            }
        }

        int converged_flag = (mpi_rank == 0 && converged) ? 1 : 0;
        MPI_Bcast(&converged_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        mpi_bcast_matrix(U);
        mpi_bcast_matrix(B);
        mpi_bcast_matrix(V);

        if (mpi_rank != 0)
        {
            last_stats = GkhRunStats{};
            last_stats.mode = "mpi-worker";
            last_stats.configured_threads = mpi_size;
            last_stats.mpi_world_size = mpi_size;
        }

        return converged_flag != 0;
    }
#endif

} // namespace

// 从“上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代（改进版）：
// - 输入输出满足 A = U * B * V^T 不变；
// - 迭代中自动分块、处理对角近零、并在每个活动块上做 bulge chasing；
// - 成功收敛后，B 被整理为非负且降序的对角矩阵（其对角元即奇异值）。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }

    GkhMode mode = read_mode();
#ifndef _OPENMP
    if (mode == GkhMode::Openmp)
    {
        mode = GkhMode::Serial;
    }
#endif
#ifndef SVD_USE_MPI
    if (mode == GkhMode::Mpi)
    {
        mode = GkhMode::Serial;
    }
#else
    int mpi_rank = 0;
    int mpi_size = 1;
    if (mode == GkhMode::Mpi)
    {
        if (mpi_world(mpi_rank, mpi_size))
        {
            return gkh_svd_from_bidiagonal_mpi(U, B, V, max_iter, tol, mpi_rank, mpi_size);
        }
        mode = GkhMode::Serial;
    }
#endif

    const int thread_count = read_thread_count(mode);
    last_stats = GkhRunStats{};
    last_stats.mode = mode_name(mode);
    last_stats.configured_threads = thread_count;

    std::vector<long long> thread_work(static_cast<size_t>(thread_count), 0);
    PthreadBlockPool pthread_pool((mode == GkhMode::Pthread) ? thread_count : 1);

    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        last_stats.iterations = iter + 1;

        // 清理数值噪声，并优先处理 d_k≈0 的特殊情形。
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        // 根据超对角线断点拆分活动块
        // 这里子矩阵间是相互独立的，所以此处具有很大的并行潜力：你可以尝试多线程/多进程进行处理
        // 但根据算法，收集 Givens 旋转并更新 U/V 需要在每个块内顺序执行，所以这可能给并行带来麻烦。
        std::vector<Block> blocks = split_active_blocks(B, n, tol);
        std::vector<Block> tasks = collect_nontrivial_blocks(blocks);
        const bool can_parallel = (mode != GkhMode::Serial && thread_count > 1);
        record_task_batch(blocks, tasks, can_parallel, last_stats);

        // 若全部是 1x1 块，说明所有超对角都已收敛为 0。
        if (tasks.empty())
        {
            converged = true;
            break;
        }

        if (!can_parallel || tasks.size() == 1)
        {
            process_blocks_serial(U, B, V, tasks, thread_work);
        }
        else if (mode == GkhMode::Pthread)
        {
            pthread_pool.process(U, B, V, tasks, thread_work);
        }
        else
        {
            process_blocks_openmp(U, B, V, tasks, thread_count, thread_work);
        }
    }

    // 迭代结束后统一结构清理与标准化输出。
    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }
    make_nonnegative_and_sort(U, B, V);
    finalize_stats(last_stats, thread_work);

    return converged;
}

const GkhRunStats &gkh_last_stats()
{
    return last_stats;
}
