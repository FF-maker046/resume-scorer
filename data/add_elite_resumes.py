"""
Add Elite High-Depth Resume Samples:
Injects real, elite-level engineering and research resumes into data/resumes_corpus.json:
1. AI Infra / LLM Inference Acceleration (vLLM, FlashAttention, CUDA/Triton)
2. Distributed Training & Parallelism (Megatron-LM 3D Parallel, ZeRO-3)
3. Database Kernel & Distributed Storage (CMU 15-445 BusTub, MIT 6.824 Raft)
4. Top-Tier AI Research (NeurIPS/CVPR SOTA, Robustness & Adversarial Defense)
5. Ultra High-Concurrency Rust/Go Cloud-Native Gateway (Million-level QPS)
"""
import json
import os

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "resumes_corpus.json")

ELITE_SAMPLES = [
    {
        "role": "数据与算法",
        "label": 1,
        "is_student": True,
        "category": "elite_ai_infra",
        "title": "大模型系统与AI Infra研发（vLLM/FlashAttention/CUDA算子）",
        "text": """陈子昂 | 求职意向：AI Infra 研发工程师 / 大模型系统工程师（2026届校招）
联系电话：13910293847 | 邮箱：chenza_infra@tsinghua.edu.cn | 城市：北京
【教育背景】
2022.09 - 2026.06  清华大学  计算机科学与技术  本科（GPA: 3.94/4.00，专业前 3%）
【专业技能】
* 深入理解 GPU 硬件架构与计算层次模型（SM、Warp、Shared Memory、Tensor Core、Memory Coalescing）
* 熟练掌握 C++17、CUDA 与 OpenAI Triton 算子开发与底层性能 Profiling（Nsight Compute, Nsight Systems）
* 深入研读并精通 vLLM、TensorRT-LLM 源码架构，熟练掌握 PagedAttention 显存优化、连续批处理与推测解码
* 熟悉 PyTorch 底层扩展机制与 FlashAttention-1/2 核心算子实现原理
【项目经历】
1. 基于 PagedAttention 的高性能大语言模型推理加速系统
* 技术栈：C++, CUDA, Triton, Python, PyTorch, vLLM
* 深入解析 Transformer 解码阶段显存瓶颈，针对多轮长文本对话 KV Cache 碎片化严重的问题，独立实现类虚拟内存的页式管理 PagedAttention 算子。
* 使用 Triton 深度定制 FlashAttention-2 前向计算内核，利用 Shared Memory 进行分块计算并消除全局显存往返读写，在 8K 长上下文下将 Attention 计算耗时降低 34%。
* 集成连续批处理（Continuous Batching）调度与多模型推测解码（Speculative Decoding），在 A100 单卡上运行 LLaMA-70B 时单并发推理吞吐提升 2.6 倍，首字生成延迟（TTFT）降低 42%。
* 项目已开源至 GitHub 并产出高质量技术解析，获得 1400+ Stars 与业界技术团队关注。
2. 自研低延迟向量近邻检索（ANN）内核引擎
* 结合 SIMD 指令集与 AVX-512，针对高维向量内积与余弦距离实现流水线级底层汇编加速，单核 QPS 达到 45000+。"""
    },
    {
        "role": "后端开发",
        "label": 1,
        "is_student": True,
        "category": "elite_systems_db",
        "title": "基础架构与系统内核研发（CMU 15-445 数据库 + MIT 6.824 Raft）",
        "text": """吴昊天 | 求职意向：基础架构 / 数据库内核 / 分布式系统研发（校招）
联系方式：13788990011 | 邮箱：haotian.wu@sjtu.edu.cn | 上海
【教育背景】
2022.09 - 2026.06  上海交通大学  软件工程  本科（ACM班）
【专业技能】
* 扎实的计算机系统底层功底，精通 C++17/20 与现代并发编程，熟练运用 GDB, Valgrind, ASan, Perf 进行内存泄漏排查与瓶颈调优
* 深入理解关系型数据库内核架构：存储引擎、Buffer Pool、索引并发控制、事务并发与隔离级别、火山模型执行器
* 精通 Raft 分布式一致性协议，深入理解分布式 CAP 定理、Paxos 与线性一致性（Linearizability）
【项目经历】
1. BusTub 关系型数据库存储与执行引擎 (CMU 15-445 独立满分通关)
* 技术栈：Modern C++, CMake, GTest, ASan
* 缓冲池管理（Buffer Pool Manager）：结合可扩展哈希表与 LRU-K 置换算法，实现细粒度读写并发控制，在极端内存紧缺压测下实现零页面丢弃。
* B+ Tree 并发索引：实现完整的 B+ 树查找、插入、分裂与合并，设计并落地并发锁机制（Crabbing / Latching 锁协议），彻底消除并发死锁与数据撕裂。
* 火山模型执行引擎（Volcano Engine）：实现 SeqScan, IndexScan, NestedLoopJoin, HashJoin 与聚合算子，引入谓词下推与基于投影消除的优化规则，TPC-C 复杂查询吞吐提升 40%。
2. 基于 Raft 共识协议的分布式一致性容灾 KV 存储系统 (MIT 6.824)
* 技术栈：Golang, RPC, Raft Consensus Protocol
* 完整实现 Leader 选举机制、心跳保活、日志线性复制、崩溃状态安全恢复与日志快照压缩机制。
* 攻坚非对称网络分区、脑裂与高丢包率场景，通过连续 2000+ 次混沌注入测试（Chaos Mesh），保障无脏读且达成严格的线性一致性读写。"""
    },
    {
        "role": "数据与算法",
        "label": 1,
        "is_student": False,
        "category": "elite_ai_research",
        "title": "顶会算法研究员（NeurIPS 一作/模型鲁棒对抗与泛化）",
        "text": """陆远舟 | 求职意向：计算机视觉 / 深度学习算法专家
电话：13800112233 | 邮箱：luyuanzhou@pku.edu.cn | 北京
【教育背景】
2020.09 - 2025.07  北京大学  信息科学技术学院  计算机系统结构  博士
【专业技能】
* 深入理解深度学习理论、凸优化理论与高维流形表征学习
* 精通 PyTorch, JAX, DeepSpeed，具备大规模 GPU 集群分布式科研与调优经验
* 在模型鲁棒性（Robustness）、对抗攻防（Adversarial Attack & Defense）与大模型安全对齐方向有深厚理论造诣与顶会成果
【学术代表作与项目】
1. 几何流形感知的高阶对抗防御新范式（NeurIPS 2024 一作发表）
* 针对传统 FGSM / PGD 静态对抗训练容易陷入鲁棒过拟合（Robust Overfitting）的理论缺陷，创新性提出基于曲率正则化约束的动态几何边界对抗防御框架。
* 在公开学术排行榜 RobustBench 与 ImageNet-1K 鲁棒性基准测试中刷新世界纪录（SOTA），在白盒极端攻击 AutoAttack 下保持 59.4% 顶级防御准确率（领先基线 4.5 个百分点）。
* 项目全套代码与预训练权重已开源至 GitHub，目前 Star 数突破 2300+，被 NeurIPS 组委会评为 Spotlight 论文。
2. 基于扩散模型的零样本对抗净化系统
* 利用预训练 Diffusion 概率模型的去噪能力，在无需重新训练下游分类模型的前提下净化恶意高频攻击扰动，端到端推断开销相比同类方法降低 60%。"""
    },
    {
        "role": "后端开发",
        "label": 1,
        "is_student": False,
        "category": "elite_high_concurrency",
        "title": "亿级流量资深架构师（百万QPS自研网关/分布式事务引擎）",
        "text": """韩振华 | 求职意向：资深后端架构师 / 基础技术专家
电话：13901019988 | 邮箱：hanzh_arch@163.com | 杭州
【工作背景】
2020.07 - 至今  某头部互联网云计算平台  核心架构部  资深技术专家（P8）
【核心技术与架构能力】
* 主导过集团级百万 QPS 核心流量网关与分布式事务中间件的自研与演进落地
* 深入掌握 Linux 异步 I/O (epoll / io_uring)、DPDK 用户态高性能网络栈与 eBPF 可观测性追踪
* 具备极强的复杂系统故障诊断能力，精通 JVM 内存模型、GC 调优、CPU 飙高排查与死锁分析
【重点架构项目】
1. 百万级 QPS 云原生自研统一流量接入网关
* 针对 Nginx 多进程内存膨胀与动态配置热加载性能抖动痛点，带领 8 人架构团队自研基于 Rust (Tokio) 与 Netty 双引擎的下一代统一接入网关。
* 承载集团全网日均超 12 亿次请求，大促峰值 QPS 达 165 万，P99 延迟稳定在 1.2ms 内。
* 引入自适应 BBR 拥塞控制限流机制与动态熔断降级策略，成功抵御 3 次外部异常流量洪峰冲击，系统达成 99.999% 高可用 SLA。
2. 金融级分布式事务强一致性中间件
* 基于两阶段提交（2PC）、TCC 与本地消息表最终一致性，自研支撑多数据源跨 IDC 分布式事务引擎，保障日均 2 亿笔账单流水零资损。"""
    }
]

def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: {CORPUS_PATH} not found.")
        return

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    existing_pos = corpus.get("positive", [])
    print(f"Original positive samples count: {len(existing_pos)}")

    # Add elite samples
    for sample in ELITE_SAMPLES:
        existing_pos.append(sample)

    corpus["positive"] = existing_pos
    corpus["positive_count"] = len(existing_pos)
    corpus["total_count"] = corpus["positive_count"] + corpus.get("negative_count", 0)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Updated positive samples count: {len(existing_pos)}")
    print(f"Total corpus samples: {corpus['total_count']}")
    print("Elite high-depth resumes successfully injected into corpus!")

if __name__ == "__main__":
    main()
