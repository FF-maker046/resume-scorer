"""
Project Deep Evaluation Engine (项目四维深度诊断引擎):
Evaluates:
1. 岗位匹配度 (Relevance / Alignment)
2. 技术难度与深度 (Difficulty & Complexity: L1 基础入门 -> L4 架构创新)
3. 工程完成度 (Engineering Completeness: 架构设计 -> 核心编码 -> 测试压测 -> 部署上线 -> 文档开源)
4. 真实性与可信度 (Authenticity & Anti-Fabrication: 开源代码/量化指标可信/真实问题解决推导)
"""
import re
from typing import Dict, List, Any, Optional


# Level 1: 基础入门 Demo / 模板课设 (Toy & Coursework)
TOY_PROJECT_PATTERNS = [
    r"(?:图书|学生|员工|宿舍|教务|选课|成绩|超市|酒店|车辆|仓库)?\s*(?:信息)?管理系统",
    r"(?:简易|简单|个人|基础)?\s*(?:博客|留言板|记事本|待办|todolist|todo)",
    r"(?:豆瓣|网易云|淘宝|微博)?\s*(?:电影|音乐)?\s*(?:TOP250|top250|简单)?爬虫",
    r"(?:仿)?(?:网易云|微信|淘宝|京东|小米)\s*(?:前端)?(?:界面|官网|商城|静态页)",
    r"(?:计算器|天气预报|贪吃蛇|五子棋|俄罗斯方块)"
]

# Level 2: 常见教程复现 / API包装应用 (Tutorial & API Wrapper)
TUTORIAL_WRAPPER_PATTERNS = [
    r"(?:FGSM|快速梯度符号)\s*(?:对抗)?\s*(?:攻击)?\s*(?:演示|可视化|demo|Demo|小工具|系统)?",
    r"(?:Streamlit|Gradio)?\s*(?:调用|对接|使用)?\s*(?:大模型|GLM|OpenAI|Kimi|GPT|通义)?\s*(?:API)?\s*(?:智能|简易)?(?:问答|对话|助手|聊天)",
    r"(?:单机|单体)?\s*(?:基础)?\s*(?:商城|商城系统|商城后台|增删改查|CRUD)",
    r"(?:MNIST|CIFAR|手写数字)?\s*(?:识别|分类)\s*(?:demo|Demo|小项目)?"
]

# Level 3: 工业级落地应用 / 知名硬核实训 (Industrial & Heavyweight Coursework)
INDUSTRIAL_HEAVYWEIGHT_PATTERNS = [
    # 算法与硬核攻防/推荐
    r"(?:PGD|C&W|AutoAttack|对抗训练|对抗鲁棒性|模型加固)",
    r"(?:RAG|检索增强).*(?:重排|Rerank|Cross-Encoder|混合检索|切片治理|长文档优化)",
    r"(?:LoRA|QLoRA|微调|Fine-tuning|DPO|RLHF|量化|AWQ|GPTQ|ONNX|TensorRT)",
    r"(?:CTR预估|DIN|DIEN|DSSM|BERT4Rec|协同过滤|召回排序|AUC调优)",
    # 名校硬核实训 (CMU 15-445, MIT 6.824, xv6)
    r"(?:CMU\s*15-445|BusTub|Buffer\s*Pool|B\+\s*Tree|LRU-K|火山模型|Volcano)",
    r"(?:MIT\s*6\.824|Raft|线性一致性|共识算法|网络分区|混沌工程)",
    r"(?:MIT\s*6\.S081|xv6|写时复制|虚拟内存分页|系统调用)",
    # 高并发核心工程
    r"(?:秒杀|防超卖|分布式锁|RedLock|Lua脚本|缓存穿透|布隆过滤器)",
    r"(?:分库分表|Sharding|读写分离|分布式事务|Seata|TCC|两阶段提交)",
    r"(?:Netty|自研RPC|连接池|心跳保活|异步编排)",
    r"(?:JMeter|压测|TPS\s*\d+|QPS\s*\d+|P99|性能瓶颈排查)"
]

# Level 4: 顶级自研 / AI Infra / 顶会SOTA (Top-Tier Architecture & SOTA Research)
TOP_TIER_SOTA_PATTERNS = [
    # AI Infra 与底层高性能算子
    r"(?:vLLM|TensorRT-LLM|PagedAttention|连续批处理|Continuous\s*Batching|推测解码|Speculative\s*Decoding)",
    r"(?:FlashAttention|Triton\s*算子|CUDA\s*算子|Shared\s*Memory\s*优化|算子融合|Kernel)",
    r"(?:Megatron|3D并行|流水线并行|张量并行|ZeRO-3|MFU调优)",
    # 顶会一作与学术SOTA
    r"(?:NeurIPS|ICML|CVPR|ICLR|ACL|KDD|AAAI|IJCAI).*(?:一作|SOTA|Spotlight|Oral)",
    r"(?:刷新世界纪录|开源.*(?:1000|2000|3000|Star|Stars))",
    # 工业级百万QPS与内核自研
    r"(?:百万级?\s*QPS|自研.*网关|自研.*存储引擎|eBPF.*零拷贝|DPDK.*协议栈|Linux内核模块)"
]

# Full Lifecycle signals
LIFECYCLE_SIGNALS = {
    "需求与架构选型": [r"需求分析", r"技术选型", r"从0到1", r"架构设计", r"全流程"],
    "核心实现与攻坚": [r"核心开发", r"独立负责", r"主导", r"实现", r"重构", r"算法", r"模型训练"],
    "质量验证与压测": [r"压测", r"JMeter", r"单元测试", r"回归", r"验证", r"指标", r"可视化展示"],
    "交付与部署上线": [r"上线", r"部署", r"Docker", r"Streamlit", r"Web应用", r"网页应用", r"持久化", r"演示"],
    "规范与开源维护": [r"github\.com", r"gitee\.com", r"README", r"说明文档", r"开源", r"软著"]
}


def extract_project_blocks(text: str) -> List[Dict[str, str]]:
    """Split resume text into distinct project blocks intelligently."""
    # Locate project section
    proj_sec_match = re.search(
        r"(?:【?\s*项目经历\s*】?|【?\s*主要项目\s*】?|【?\s*核心项目\s*】?)[\s\S]*?(?=【?\s*(?:教育背景|专业技能|工作经历|实习经历|获奖证书|荣誉奖项|自我评价)\s*】?|$)",
        text
    )
    proj_text = proj_sec_match.group(0) if proj_sec_match else text

    # Split by explicit date periods or numbered items
    splits = re.split(r"(?=\b\d{4}[\.\-/]\d{2}\s*[-~至到]\s*(?:\d{4}[\.\-/]\d{2}|至今)?)", proj_text)
    
    # If date split gave multiple blocks
    if len(splits) > 1 and len(splits[1].strip()) > 30:
        raw_blocks = [s.strip() for s in splits if len(s.strip()) > 30 and "项目经历" != s.strip()]
    else:
        # Fallback split by project titles/bullets
        lines = proj_text.split("\n")
        blocks = []
        cur = []
        for line in lines:
            line_s = line.strip()
            if not line_s or "项目经历" in line_s:
                continue
            if re.search(r"^(?:\d+[\.、]|【|◆|●|#)\s*[\u4e00-\u9fa5A-Za-z0-9]", line_s) and cur:
                blocks.append("\n".join(cur))
                cur = [line_s]
            else:
                cur.append(line_s)
        if cur:
            blocks.append("\n".join(cur))
        raw_blocks = blocks

    if not raw_blocks:
        raw_blocks = [proj_text]

    projects = []
    for idx, b_text in enumerate(raw_blocks):
        # Extract and clean project title
        first_line = b_text.split("\n")[0].strip()
        # Remove leading dates
        cleaned_first = re.sub(r"^\d{4}[\.\-/]\d{2}\s*[-~至到]\s*(?:\d{4}[\.\-/]\d{2}|至今)?\s*", "", first_line)
        cleaned_first = re.sub(r"^\d+[\.、]\s*", "", cleaned_first)
        # Split by keywords like 描述, 技术栈, 【
        title_candidate = re.split(r"描述|技术栈|项目介绍|职责|【|\|", cleaned_first)[0].strip()
        
        # If title is too long or empty
        if not title_candidate or len(title_candidate) > 25:
            match = re.search(r"[A-Za-z0-9\u4e00-\u9fa5\s]{2,20}(?:系统|助手|平台|爬虫|引擎|应用|项目|小程序|APP)", cleaned_first)
            title = match.group(0).strip() if match else f"核心技术项目 {idx + 1}"
        else:
            title = title_candidate

        projects.append({
            "title": title,
            "raw_text": b_text
        })

    return projects


def evaluate_single_project(proj: Dict[str, str], target_role: str = "数据与算法") -> Dict[str, Any]:
    text = proj["raw_text"]
    title = proj["title"]

    # 1. 难度与深度评级 (Technical Difficulty & Depth: L1 -> L4)
    is_top_tier = any(re.search(p, text, re.I) for p in TOP_TIER_SOTA_PATTERNS)
    is_industrial = any(re.search(p, text, re.I) for p in INDUSTRIAL_HEAVYWEIGHT_PATTERNS)
    is_tutorial = any(re.search(p, text, re.I) for p in TUTORIAL_WRAPPER_PATTERNS)
    is_toy = any(re.search(p, text, re.I) for p in TOY_PROJECT_PATTERNS)

    if is_top_tier:
        tier = "L4"
        tier_title = "L4 顶级自研 / AI Infra / 顶会SOTA"
        diff_score = 96
        diff_desc = "具备顶级技术壁垒与底层突破（识别到底层算子/分布式训练/顶会科研等核心创新），达到大厂 SSP / 顶尖核心团队直通车水准。"
        upgrade_tip = "技术壁垒极高。建议重点梳理系统级架构权衡（如 GPU 访存局部性、分布式跨节点通信拓扑），准备应对首席架构师级别的深度面试追问。"
    elif is_industrial:
        tier = "L3"
        tier_title = "L3 工业级落地 / 名校硬核实训"
        diff_score = 88
        diff_desc = "具备扎实工业级工程落地或名校硬核系统实现（覆盖关键复杂机制），在名企技术面试中具备强技术区分度。"
        upgrade_tip = "项目具备良好技术深度。建议在简历中进一步丰富‘极限压测指标、性能瓶颈 Profiling 耗时拆解’，用更硬核的数据说话。"
    elif is_tutorial or any(k in text for k in ["FGSM", "Streamlit", "大模型API", "GLM API", "智谱GLM"]):
        tier = "L2"
        tier_title = "L2 教程复现 / API包装应用"
        diff_score = 72
        diff_desc = "具备基础动手实践与流程打通能力，但整体属于开源教程复现或第三方 API 封装调用，在大厂技术面试中技术壁垒偏薄弱。"
        if "FGSM" in text:
            upgrade_tip = "当前属于基础的单文件梯度攻击教程复现（核心十几行代码即可完成）。建议升级为‘高阶对抗攻防加固系统’：引入 PGD/AutoAttack 强攻击，加入对抗训练加固，并在大模型/实际视觉系统上测试鲁棒性。"
        elif any(k in text for k in ["Streamlit", "大模型", "GLM", "问答"]):
            upgrade_tip = "当前属于基础的大模型 API + Streamlit 套壳开发。建议升级为‘企业级低延迟 RAG 检索增强引擎’：引入 BGE-Reranker 重排模型、自建长文档切片治理、实现连续流式并发与 P99 延迟治理。"
        else:
            upgrade_tip = "建议深入底层原理，补充真实高并发压测、性能调优或复杂场景容灾逻辑，脱离单纯的教程代码复现范畴。"
    elif is_toy:
        tier = "L1"
        tier_title = "L1 入门Demo / 常见课设"
        diff_score = 58
        diff_desc = "属于基础入门或常见初学者爬虫/增删改查模板，在应届求职中同质化极其严重，名企技术初筛区分度极低。"
        upgrade_tip = "此项目在应届求职中同质化较为严重。建议在描述中强化‘反爬对抗/高并发异步/断点续传/自动化ETL入库’等进阶工程特性，或将其定位为辅助项目让位给核心技术项目。"
    else:
        tier = "L2"
        tier_title = "L2 基础业务项目"
        diff_score = 70
        diff_desc = "具备基本的开发链路，建议进一步深挖底层原理与性能调优环节。"
        upgrade_tip = "建议在项目中补充具体的技术难点攻坚过程与量化数据支撑。"

    # 2. 真实性与可信度 (Authenticity & Anti-Fabrication)
    has_github = bool(re.search(r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text, re.I))
    has_demo_url = bool(re.search(r"https?://[^\s]+", text))
    has_specific_metrics = bool(re.search(r"\d+(?:\.\d+)?%|QPS\s*\d+|TPS\s*\d+|\d+\s*ms|提升\s*\d+|降低\s*\d+|\d+\s*万", text))
    has_problem_solution = any(w in text for w in ["针对", "痛点", "瓶颈", "解决", "优化", "重构", "引入", "缓解", "清洗与预处理"])

    auth_score = 60
    auth_evidence = []
    if has_github:
        auth_score += 25
        auth_evidence.append("附带真实开源 GitHub 代码仓库")
    elif has_demo_url:
        auth_score += 15
        auth_evidence.append("附带在线演示/文档链接")

    if has_specific_metrics:
        auth_score += 15
        auth_evidence.append("具备具体量化指标佐证")

    if has_problem_solution:
        auth_score += 10
        auth_evidence.append("具备真实业务痛点分析与落地推导")

    auth_score = min(100, auth_score)
    if auth_score >= 85:
        auth_level = "极高 (代码开源且数据翔实)"
    elif auth_score >= 70:
        auth_level = "良好 (有具体实现与数据支撑)"
    else:
        auth_level = "一般 (缺乏代码地址与具体技术细节)"

    # 3. 完成度 (Completeness & Full Lifecycle)
    lifecycle_found = []
    for stage, pats in LIFECYCLE_SIGNALS.items():
        if any(re.search(p, text) for p in pats):
            lifecycle_found.append(stage)

    comp_score = min(100, int((len(lifecycle_found) / 5.0) * 100) + 15)

    # 4. 岗位匹配度 (Relevance)
    relevance_score = 80
    if "算法" in target_role or "数据" in target_role:
        if any(w in text for w in ["PyTorch", "FGSM", "LLM", "FAISS", "对抗", "大模型", "爬虫", "数据清洗", "算法", "Streamlit"]):
            relevance_score = 95
        elif any(w in text for w in ["Java", "Spring", "微服务"]):
            relevance_score = 65
    elif "后端" in target_role:
        if any(w in text for w in ["Java", "Spring", "MySQL", "Redis", "微服务", "并发", "API", "Go"]):
            relevance_score = 95
        else:
            relevance_score = 70
    elif "前端" in target_role:
        if any(w in text for w in ["Vue", "React", "HTML", "CSS", "TypeScript", "Streamlit", "前端", "页面"]):
            relevance_score = 95
        else:
            relevance_score = 65

    # Composite project quality score
    proj_composite = int(diff_score * 0.30 + auth_score * 0.25 + comp_score * 0.25 + relevance_score * 0.20)

    # Append authenticity advice if missing link
    if not has_github and not has_demo_url:
        upgrade_tip += " 另外建议在简历中附带公开可访问的 GitHub 仓库或在线 Demo 地址以增强工程可信度。"
    elif not has_specific_metrics and tier in ["L2", "L3"]:
        upgrade_tip += " 另外建议补充具体的实验/压测数字（如延迟毫秒数、吞吐量或准确率变化），用数据支撑技术成果。"

    return {
        "title": title,
        "composite_score": proj_composite,
        "difficulty_tier": tier,
        "difficulty_title": tier_title,
        "difficulty_score": diff_score,
        "difficulty_desc": diff_desc,
        "authenticity_score": auth_score,
        "authenticity_level": auth_level,
        "authenticity_evidence": auth_evidence,
        "completeness_score": comp_score,
        "lifecycle_stages": lifecycle_found,
        "relevance_score": relevance_score,
        "upgrade_tip": upgrade_tip
    }


def evaluate_all_projects(text: str, target_role: str = "数据与算法") -> Dict[str, Any]:
    """Extract and comprehensively evaluate all resume projects."""
    raw_projects = extract_project_blocks(text)
    if not raw_projects:
        raw_projects = [{"title": "核心技术项目", "raw_text": text}]

    evaluated = [evaluate_single_project(p, target_role) for p in raw_projects]

    avg_score = int(sum(p["composite_score"] for p in evaluated) / len(evaluated))
    avg_diff = int(sum(p["difficulty_score"] for p in evaluated) / len(evaluated))
    avg_auth = int(sum(p["authenticity_score"] for p in evaluated) / len(evaluated))
    avg_comp = int(sum(p["completeness_score"] for p in evaluated) / len(evaluated))
    avg_rel = int(sum(p["relevance_score"] for p in evaluated) / len(evaluated))

    return {
        "overall_score": avg_score,
        "project_count": len(evaluated),
        "avg_difficulty": avg_diff,
        "avg_authenticity": avg_auth,
        "avg_completeness": avg_comp,
        "avg_relevance": avg_rel,
        "projects": evaluated
    }
