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


# Typical beginner / tutorial / toy project patterns
TOY_PROJECT_PATTERNS = [
    r"(?:图书|学生|员工|宿舍|教务|选课|成绩|超市|酒店|车辆|仓库)?\s*(?:信息)?管理系统",
    r"(?:简易|简单|个人|基础)?\s*(?:博客|留言板|记事本|待办|todolist|todo)",
    r"(?:豆瓣|网易云|淘宝|微博)?\s*(?:电影|音乐)?\s*(?:TOP250|top250|简单)?爬虫",
    r"(?:仿)?(?:网易云|微信|淘宝|京东|小米)\s*(?:前端)?(?:界面|官网|商城|静态页)",
    r"(?:计算器|天气预报|贪吃蛇|五子棋|俄罗斯方块)"
]

# Advanced engineering / research / technical depth indicators (L3 & L4)
ADVANCED_TECH_INDICATORS = [
    # 算法与AI硬核深度
    r"(?:对抗攻击|对抗样本|FGSM|PGD|鲁棒性|模型加固)",
    r"(?:RAG|检索增强|向量数据库|FAISS|Milvus|Chroma|Embedding|语义召回)",
    r"(?:LoRA|微调|Fine-tuning|模型量化|蒸馏|剪枝|ONNX|TensorRT)",
    r"(?:Transformer|BERT|GPT|LLM|大模型|Prompt\s*Engineering)",
    r"(?:特征工程|交叉特征|CTR预估|召回排序|AUC调优)",
    # 后端高并发与架构深度
    r"(?:分布式锁|RedLock|Lua脚本|缓存穿透|缓存雪崩|缓存击穿|布隆过滤器)",
    r"(?:分库分表|Sharding|主从复制|读写分离|分布式事务|Seata|TCC)",
    r"(?:RPC|gRPC|Netty|异步编排|CompletableFuture|响应式编程)",
    r"(?:JVM调优|GC调优|线程池参数|OOM排查|CPU飙高排查)",
    r"(?:Kafka|RabbitMQ|RocketMQ)\s*(?:消息堆积|幂等性|延时队列|削峰填谷)",
    # 工程质量与基准压测
    r"(?:JMeter|压测|TPS|QPS|P99|性能瓶颈|并发测试)",
    r"(?:CI/CD|自动化部署|Docker|Kubernetes|K8s|Prometheus|Grafana)"
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
    is_toy = any(re.search(p, text, re.I) for p in TOY_PROJECT_PATTERNS)
    adv_hits = [re.search(p, text, re.I).group(0) for p in ADVANCED_TECH_INDICATORS if re.search(p, text, re.I)]

    has_hardcore_ai = any(k in text for k in ["FGSM", "对抗攻击", "对抗样本", "FAISS", "RAG", "大模型API", "LoRA", "Transformer"])
    has_hardcore_backend = any(k in text for k in ["分布式锁", "分库分表", "Kafka削峰", "JMeter压测", "JVM调优", "QPS"])

    if has_hardcore_ai or has_hardcore_backend or len(adv_hits) >= 3:
        tier = "L3"
        tier_title = "L3 进阶核心工程与算法"
        diff_score = 92
        diff_desc = f"具备显著技术壁垒与前沿探索（深度识别到: {', '.join(adv_hits[:3])}），脱离常规教学课设范畴。"
    elif is_toy:
        tier = "L1"
        tier_title = "L1 入门Demo / 常见课设"
        diff_score = 65
        diff_desc = "属于基础入门或常见初学者爬虫/管理系统模板，同质化程度偏高，在名企技术面试中区分度较弱。"
    elif len(adv_hits) >= 1 or len(text) > 200:
        tier = "L2"
        tier_title = "L2 工业级实际应用"
        diff_score = 80
        diff_desc = f"具备实际工程落地与完整业务闭环（识别到关键技术点: {', '.join(adv_hits[:2]) if adv_hits else '标准工程规范'}）。"
    else:
        tier = "L2"
        tier_title = "L2 基础业务项目"
        diff_score = 75
        diff_desc = "具备基本的开发链路，建议进一步深挖底层原理与性能调优环节。"

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

    # Specific upgrade suggestion
    if is_toy:
        upgrade_tip = "此项目在应届求职中同质化较为严重。建议在描述中强化‘反爬对抗/高并发异步/断点续传/自动化ETL入库’等进阶工程特性，或将其定位为辅助项目。"
    elif not has_github and not has_demo_url:
        upgrade_tip = "建议在项目中附带可公开访问的 GitHub 仓库或在线 Demo 地址，能使项目真实性与工程师自驱形象倍增。"
    elif not has_specific_metrics:
        upgrade_tip = "建议补充具体的量化数字（如：对抗扰动半径下攻击准确率变化、向量检索平均响应毫秒数等），用数据说话。"
    else:
        upgrade_tip = "项目整体工程与算法质量较高，可针对面试准备‘架构设计权衡考量（为什么选此技术栈而非其他）’的深度技术回答。"

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
