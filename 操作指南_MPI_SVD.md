# SVD MPI 操作指南

本文只写操作流程。实验原理、实验结果和分析见 `report.tex` / `report.pdf`。

## 1. 需要修改哪些代码

不要修改：

- `main.cpp`
- `test.sh`
- `qsub.sh`

需要修改或新增：

1. `matrix.h`
   - 增加 `data()`、`const data()`、`size()`。
   - 目的：MPI 广播矩阵时需要连续内存指针和元素总数。

2. `gkh.h`
   - 在 `GkhRunStats` 中增加 MPI 统计字段：
     - `mpi_world_size`
     - `mpi_tasks_sent`
     - `mpi_matrix_broadcast_bytes`
     - `mpi_result_bytes`
   - 目的：报告中分析 MPI 通信和任务调度开销。

3. `gkh.cpp`
   - 增加 `SVD_USE_MPI` 条件编译。
   - 增加 MPI 自动初始化和退出处理。
   - 增加 MPI 活动块任务池：
     - 0 号进程拆分 GKH 活动块。
     - 多个非平凡活动块同时存在时，广播当前 `U/B/V`。
     - 工作进程领取活动块，执行一次 bulge chasing。
     - 工作进程只回传对应的 `B` 子块和 `U/V` 相关列。
   - 若当前只有一个非平凡活动块，则 0 号进程串行处理，避免无意义通信。

4. `qsub_mpi.sh`
   - 使用官方模板，只把示例中的 `ntt` 替换成 `svd`。
   - 根据实验需要手动修改：
     - `#PBS -l nodes=...:ppn=...`
     - `/usr/local/bin/mpiexec -np ...`

## 2. 编译命令

在服务器的 `~/svd` 目录下执行：

```bash
mpic++ -std=c++17 -O2 -DSVD_USE_MPI main.cpp bidiagonalization.cpp gkh.cpp -o main
```

如果要做 SIMD 对比，可再编译一个禁用 SIMD 的版本：

```bash
mpic++ -std=c++17 -O2 -DSVD_USE_MPI -DSVD_DISABLE_SIMD main.cpp bidiagonalization.cpp gkh.cpp -o main_nosimd
```

正式提交时文件名应为 `main`。若测试 `main_nosimd`，需要临时改名或复制：

```bash
cp main_nosimd main
```

## 3. 官方脚本提交

确认 `qsub_mpi.sh` 中有如下结构：

```bash
#!/bin/sh
#PBS -N qsub_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=2:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)

for node in $NODES; do
    scp master_ubss1:/home/${USER}/svd/main ${node}:/home/${USER} 1>&2
    scp -r master_ubss1:/home/${USER}/svd/files ${node}:/home/${USER}/ 1>&2
done

/usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main

scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/svd/ 2>&1
```

提交：

```bash
qsub qsub_mpi.sh
```

查看任务：

```bash
qstat
```

查看输出：

```bash
cat test.o
cat test.e
```

## 4. 多组实验怎么做

不写额外运行脚本，手动改 `qsub_mpi.sh` 后重复提交。

### 4.1 不同 MPI 进程数

每次只改两处：

```bash
#PBS -l nodes=1:ppn=8
/usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main
```

建议实验组合：

| 实验 | nodes | ppn | np |
|---|---:|---:|---:|
| MPI-1 | 1 | 1 | 1 |
| MPI-2 | 1 | 2 | 2 |
| MPI-4 | 1 | 4 | 4 |
| MPI-8 | 1 | 8 | 8 |
| MPI-16 | 2 | 8 | 16 |
| MPI-32 | 4 | 8 | 32 |

约束：

```text
np <= nodes * ppn, nodes <= 4, ppn <= 8
```

### 4.2 不同编译优化等级

重复编译并提交：

```bash
mpic++ -std=c++17 -O0 -DSVD_USE_MPI main.cpp bidiagonalization.cpp gkh.cpp -o main
qsub qsub_mpi.sh

mpic++ -std=c++17 -O2 -DSVD_USE_MPI main.cpp bidiagonalization.cpp gkh.cpp -o main
qsub qsub_mpi.sh

mpic++ -std=c++17 -O3 -DSVD_USE_MPI main.cpp bidiagonalization.cpp gkh.cpp -o main
qsub qsub_mpi.sh
```

### 4.3 SIMD 对比

启用 SIMD：

```bash
mpic++ -std=c++17 -O2 -DSVD_USE_MPI main.cpp bidiagonalization.cpp gkh.cpp -o main
qsub qsub_mpi.sh
```

禁用 SIMD：

```bash
mpic++ -std=c++17 -O2 -DSVD_USE_MPI -DSVD_DISABLE_SIMD main.cpp bidiagonalization.cpp gkh.cpp -o main
qsub qsub_mpi.sh
```

### 4.4 不同随机种子

`main.cpp` 支持第一个命令行参数作为随机种子。若要换种子，修改 `qsub_mpi.sh` 的 mpiexec 行：

```bash
/usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main 20260408
```

可测试：

```text
20260408
20260409
20260410
```

## 5. 保存结果到文件夹

官方脚本默认输出到 `test.o` 和 `test.e`。每次任务结束后可手动保存：

```bash
mkdir -p results
cp test.o results/np8_O2_simd_seed20260408.o
cp test.e results/np8_O2_simd_seed20260408.e
```

更换进程数、优化等级或 SIMD 开关后，改文件名保存即可。

## 6. Profiling 数据怎么准备

服务器端不做 `perf`。实验手册明确说明本题不强制使用 `perf`，可以使用计时等方式做较浅层的 profiling。本文的服务器端 profiling 只基于官方程序输出的计时数据：`总上二对角化耗时(ms)`、`总GKH迭代耗时(ms)`、不同 `np` 下的重复运行均值和标准差。这样不会修改 `main.cpp`，也不会额外写运行脚本。

### 6.1 需要采集的典型配置

建议至少保留这些组：

| 目的 | nodes | ppn | np | 保存名示例 |
|---|---:|---:|---:|---|
| 串行基线 | 1 | 1 | 1 | `timing_np1` |
| 最优点 | 1 | 4 | 4 | `timing_np4` |
| 单节点满核心 | 1 | 8 | 8 | `timing_np8` |
| 跨节点对照 | 2 | 8 | 16 | `timing_np16` |
| 负优化代表 | 4 | 8 | 32 | `timing_np32` |

每组仍然只修改 `qsub_mpi.sh` 中两处：

```bash
#PBS -l nodes=1:ppn=4
/usr/local/bin/mpiexec -np 4 -machinefile $PBS_NODEFILE /home/${USER}/main 2412592
```

注意：带 `$PBS_NODEFILE` 的 `mpiexec` 行必须写在 `qsub_mpi.sh` 中通过 `qsub qsub_mpi.sh` 运行，不要直接粘贴到登录节点终端。

### 6.2 每组如何保存

以 `np=4` 为例：

```bash
mpic++ -std=c++17 -O2 -DSVD_USE_MPI main.cpp bidiagonalization.cpp gkh.cpp -o main
sed -i "s/^#PBS -l nodes=.*/#PBS -l nodes=1:ppn=4/" qsub_mpi.sh
sed -i "s#^/usr/local/bin/mpiexec .*#/usr/local/bin/mpiexec -np 4 -machinefile \\$PBS_NODEFILE /home/\\${USER}/main 2412592#" qsub_mpi.sh
jid=$(qsub qsub_mpi.sh)
while qstat "$jid" >/dev/null 2>&1; do sleep 10; done
mkdir -p profiling/timing_np4
cp test.o test.e profiling/timing_np4/
```

检查输出：

```bash
grep "总上二对角化耗时\|总GKH迭代耗时\|通过:" profiling/timing_np4/test.o
grep "FAIL\|converged                 : no" profiling/timing_np4/test.o
```

预期应看到：

```text
通过: 5 / 5
```

如果存在 `FAIL` 或 `converged : no`，该组不能作为有效数据。

### 6.3 重复运行

为了让 profiling 分析有统计依据，每个关键配置建议重复 3 次，至少包括：

```text
np=1, np=4, np=8, np=16, np=32
```

每次保存为不同目录，例如：

```bash
mkdir -p profiling/timing_np4_trial1
cp test.o test.e profiling/timing_np4_trial1/
```

报告中重点比较：

- `np=4` 是否比 `np=1` 的 GKH 时间更短；
- `np=8/16/32` 是否继续加速，还是出现回退；
- `np=32` 的标准差是否明显变大；
- 上二对角化是否随 `np` 增加而重复消耗；
- SIMD 对上二对角化和 GKH 的影响是否一致。

### 6.4 报告中如何解释 profiling

服务器端计时数据用于解释 MPI 层面的开销：

- 各进程内存独立，因此每个 rank 都持有自己的 `U/B/V` 矩阵副本；
- 进入并行分发轮次时，根进程需要广播当前 `U/B/V`，这是主要通信成本；
- worker 完成活动块后只回传局部 `B` 子块和 `U/V` 相关列，但当活动块很大或活动块数量很少时，局部回传优势会被等待和广播抵消；
- 如果 `np=4` 有加速，而 `np=8/16/32` 变慢，就说明任务池在中等并行度下有效，但高进程数下活动块不足、通信和等待占主导；
- 如果 `np=32` 标准差很大，可以作为跨节点通信和任务不均衡导致性能不稳定的证据。

本地 VTune 结果只用于补充说明“单个进程内部热点主要在 GKH 列更新和矩阵运算”，服务器端结论仍以真实 PBS 计时数据为主。

## 7. 预期结果

正常输出中，每个样例应看到：

```text
converged                 : yes
relative recon error      : 1e-13 左右
diagonal structure error  : 0
descending order error    : 0
nonnegative diagonal      : yes
结果: PASS
```

最后应看到：

```text
通过: 5 / 5
```

若出现：

```text
通过: 4 / 5
```

或某个样例 `converged : no`，说明该组实验不能作为正确结果，需要检查：

1. 是否用 `mpic++ -DSVD_USE_MPI` 编译；
2. `np <= nodes * ppn` 是否满足；
3. 是否有上一轮死锁任务未清理；
4. 是否用了过激优化选项导致数值异常。

## 8. 报告中应记录什么

每组实验至少记录：

- `np`
- `nodes:ppn`
- 编译选项：`O0/O2/O3/Ofast`
- 是否启用 SIMD
- 随机种子
- `总上二对角化耗时(ms)`
- `总GKH迭代耗时(ms)`
- `通过: 5 / 5`

分析时重点比较：

1. GKH 时间是否随 `np` 增加下降；
2. MPI 进程数过多时是否因为通信开销变慢；
3. SIMD 对上二对角化和 GKH 是否都有收益；
4. 不同随机种子下结论是否稳定。
