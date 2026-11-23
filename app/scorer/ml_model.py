"""
Machine Learning module for resume analysis.
Includes:
1. Job Role Classification (TF-IDF + LogisticRegression)
2. Job Description (JD) Semantic Matching (TF-IDF + Cosine Similarity)
3. Resume Quality Classifier & Benchmark Centroid (Trained on Positive/Negative corpus)
"""
import os
import re
from typing import Dict, List, Any, Optional
import jieba
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from app.scorer.rules import (
    evaluate_contact,
    evaluate_sections,
    evaluate_quantification,
    extract_skills,
    evaluate_readability_and_length
)


# Common Chinese and English stopwords for resumes
STOPWORDS = set([
    "的", "了", "和", "与", "及", "在", "等", "进行", "负责", "参与", "熟练", "熟悉",
    "精通", "掌握", "具有", "能够", "以及", "并", "中", "对", "为", "以", "使用",
    "通过", "有", "良好", "经验", "能力", "相关", "工作", "主要", "基于", "完成",
    "the", "and", "or", "in", "to", "of", "a", "for", "with", "on", "at"
])


def jieba_tokenize(text: str) -> List[str]:
    """Tokenize text using jieba, filtering punctuation and stopwords."""
    words = jieba.cut(text)
    tokens = []
    for w in words:
        w_clean = w.strip().lower()
        if len(w_clean) >= 2 and w_clean not in STOPWORDS and not re.match(r"^[\d\W_]+$", w_clean):
            tokens.append(w_clean)
    return tokens


# Training data seeds for 5 core career roles
CORPUS_SEEDS = {
    "后端开发": [
        "熟练掌握 Java 基础，深入理解 JVM 内存模型、垃圾回收机制、类加载机制及多线程高并发编程",
        "熟练使用 Spring Boot, Spring Cloud, MyBatis 微服务架构，具备微服务拆分与治理经验",
        "精通 MySQL 关系型数据库，具备 SQL 调优、索引优化、分库分表及主从复制实践经验",
        "熟悉 Redis 核心数据结构，精通缓存击穿、缓存穿透、缓存雪崩的解决方案及分布式锁实现",
        "掌握 Kafka, RabbitMQ 等主流消息中间件，具备消息幂等性处理与分布式事务经验",
        "精通 Python FastAPI, Django 框架，熟练构建高吞吐 RESTful API 与异步任务队列",
        "熟练使用 Golang 语言，理解 Goroutine 调度原理、Channel 通信机制，具备高性能网络编程经验",
        "熟练掌握 Linux 常用操作与 Shell 脚本编写，精通 Docker 容器化技术及 Kubernetes 集群部署",
        "具备系统架构设计能力，主导高并发系统重构，优化核心接口响应耗时降低 50%，QPS 提升至 10000+"
    ],
    "前端开发": [
        "精通 HTML5, CSS3, JavaScript ES6+ 语法，深入理解事件循环、原型链及闭包机制",
        "熟练使用 Vue3, Vue2, Pinia, Vue Router，理解响应式原理及虚拟 DOM 渲染过程",
        "熟练掌握 React 18, React Hooks, Redux Toolkit，具备大型单页面应用架构设计与组件库封装经验",
        "精通 TypeScript，熟练编写高类型安全的业务代码与公共工具函数",
        "熟悉 Webpack, Vite 等现代前端工程化构建工具，具备打包体积压缩与首屏加载速度优化经验",
        "熟练使用 Tailwind CSS, Sass, Less 等预处理器，具备自适应与移动端响应式布局设计经验",
        "熟悉微信小程序开发流程，熟练对接微信支付、分享、定位等开放平台接口",
        "掌握前端性能监控、Core Web Vitals 指标优化，主导首屏 FCP 从 2.5s 优化至 0.8s"
    ],
    "数据与算法": [
        "熟练掌握机器学习经典算法：逻辑回归、支持向量机、决策树、随机森林及梯度提升树 XGBoost, LightGBM",
        "熟练使用 PyTorch, TensorFlow 深度学习框架，掌握 CNN, RNN, Transformer 架构及预训练模型微调",
        "熟练使用 Pandas, NumPy, Scikit-Learn 进行大规模特征工程、数据清洗、归一化及异常值检测",
        "熟悉自然语言处理 NLP 技术，包括文本分类、命名实体识别 NER、语义匹配及大语言模型 LLM Prompt Engineering",
        "精通 SQL 数据分析，熟练掌握窗口函数、多表关联，具备 Hive, Spark 大数据离线与实时计算经验",
        "具备推荐系统召回与排序模型构建经验，主导点击率 CTR 预估模型迭代，使核心推荐指标提升 18%",
        "熟悉模型评估指标：AUC, ROC, Precision, Recall, F1-score, NDCG，具备严谨的模型实验与分析能力"
    ],
    "测试与运维": [
        "熟练掌握软件测试理论与测试用例设计方法（等价类划分、边界值分析、因果图法）",
        "熟练使用 Python + Pytest / Unittest 编写自动化接口测试脚本，具备持续集成流水线搭建经验",
        "熟练使用 Selenium, Playwright 进行 Web UI 自动化回归测试，降低手工测试成本 60%",
        "掌握 JMeter, LoadRunner 等性能压测工具，精通性能瓶颈排查、并发线程压测与分析报告撰写",
        "精通 Linux 操作系统，熟练配置 Nginx 反向代理、负载均衡、SSL 证书及日志分析",
        "熟悉 CI/CD 流程，熟练使用 Git, Jenkins, GitHub Actions 搭建自动化打包部署流水线",
        "熟悉 Prometheus + Grafana 监控报警系统搭建，保障生产环境高可用性与故障快速定位"
    ],
    "产品与运营": [
        "主导互联网产品从 0 到 1 的全生命周期规划与落地，输出高质量 PRD 产品需求文档与功能流程图",
        "精通 Axure, Figma, Sketch 高保真原型设计工具，具备出色的用户体验与交互设计嗅觉",
        "负责用户增长策略制定与漏斗转化率分析，精通 A/B 测试设计、核心指标拆解与数据埋点规划",
        "具备出色的跨部门沟通协同与项目管理能力，推进敏捷开发 Scrum 流程，确保产品按期迭代上线",
        "深入洞察用户痛点，主导用户深度访谈与竞品调研，输出行业分析报告并指导产品战略布局",
        "主导核心业务流程改版，提升新用户次日留存率 12%，用户付费转化率提升 25%"
    ]
}


class RoleClassifier:
    """Predicts career track role probabilities for a given resume."""
    def __init__(self):
        self.roles = list(CORPUS_SEEDS.keys())
        self.vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize, token_pattern=None, max_features=1000)
        self.clf = LogisticRegression(max_iter=300, C=1.0)
        self._train()

    def _train(self):
        texts = []
        labels = []
        for role, samples in CORPUS_SEEDS.items():
            for s in samples:
                texts.append(s)
                labels.append(role)
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

    def predict(self, text: str) -> List[Dict[str, Any]]:
        X_test = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X_test)[0]
        classes = self.clf.classes_

        results = []
        for role, prob in zip(classes, probs):
            results.append({
                "role": role,
                "confidence": round(float(prob) * 100, 1)
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results


class QualityEvaluator:
    """
    Evaluates resume quality against the positive/negative training corpus.
    Provides:
    - ML Pass Probability (初筛通过率预测)
    - Benchmark Centroid Similarity (大厂高分基准相似度)
    - Positive & Penalty Signal Extraction
    """
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "models", "quality_model.joblib")
        self.bundle = None
        self._load_or_train()

    def _load_or_train(self):
        if not os.path.exists(self.model_path):
            from train_model import train
            train()
        self.bundle = joblib.load(self.model_path)

    def extract_structural_features(self, text: str) -> np.ndarray:
        contact = evaluate_contact(text)
        sections = evaluate_sections(text)
        quant = evaluate_quantification(text)
        skills = extract_skills(text)
        readability = evaluate_readability_and_length(text)

        features = [
            contact["score"] / 100.0,
            sections["score"] / 100.0,
            min(1.0, quant["quant_count"] / 5.0),
            min(1.0, quant["action_count"] / 6.0),
            min(1.0, quant["star_count"] / 3.0),
            min(1.0, skills["total_count"] / 10.0),
            min(1.0, readability["char_count"] / 1500.0),
            readability["score"] / 100.0
        ]
        return np.array(features, dtype=float)

    def evaluate(self, text: str) -> Dict[str, Any]:
        vectorizer = self.bundle["vectorizer"]
        text_clf = self.bundle["text_clf"]
        struct_clf = self.bundle["struct_clf"]
        pos_centroid = self.bundle["pos_centroid"]
        vocab = np.array(self.bundle["vocab"])
        coefs = np.array(self.bundle["coefs"])

        # 1. Text & Structural ML Predictions
        X_tfidf = vectorizer.transform([text])
        text_prob = float(text_clf.predict_proba(X_tfidf)[0][1])

        struct_feats = self.extract_structural_features(text)
        struct_prob = float(struct_clf.predict_proba([struct_feats])[0][1])

        ensemble_prob = 0.5 * text_prob + 0.5 * struct_prob
        pass_rate = int(round(ensemble_prob * 100))

        # 2. Benchmark Centroid Cosine Similarity
        vec_norm = X_tfidf.toarray() / (np.linalg.norm(X_tfidf.toarray()) + 1e-9)
        raw_sim = float(np.dot(vec_norm, pos_centroid.T)[0][0])
        benchmark_sim = int(min(100, max(25, round(raw_sim * 135))))

        # 3. Signals in candidate resume
        resume_tokens = set(jieba_tokenize(text))
        pos_signals = []
        neg_signals = []

        top_pos_indices = np.argsort(coefs)[::-1]
        for idx in top_pos_indices:
            w = vocab[idx]
            if w in resume_tokens and coefs[idx] > 0.05:
                pos_signals.append(f"{w} (+{coefs[idx]:.2f})")
                if len(pos_signals) >= 6:
                    break

        top_neg_indices = np.argsort(coefs)
        for idx in top_neg_indices:
            w = vocab[idx]
            if w in resume_tokens and coefs[idx] < -0.05:
                neg_signals.append(f"{w} ({coefs[idx]:.2f})")
                if len(neg_signals) >= 5:
                    break

        if pass_rate >= 80:
            quality_level = "大厂通关级 (高通过率)"
        elif pass_rate >= 60:
            quality_level = "良好竞争力 (建议局部优化)"
        else:
            quality_level = "初筛淘汰风险较高 (需重点改进)"

        return {
            "pass_rate": pass_rate,
            "quality_level": quality_level,
            "benchmark_similarity": benchmark_sim,
            "positive_signals": pos_signals,
            "penalty_signals": neg_signals
        }


class JDMatcher:
    """Evaluates semantic matching and keyword overlap between Resume and target JD."""
    @staticmethod
    def match(resume_text: str, jd_text: str) -> Dict[str, Any]:
        if not jd_text or len(jd_text.strip()) < 10:
            return {
                "has_jd": False,
                "similarity_score": 0,
                "matched_keywords": [],
                "missing_keywords": []
            }

        vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize, token_pattern=None)
        try:
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
            sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            score = int(min(100, max(20, round(sim * 150))))
        except Exception:
            score = 60
            sim = 0.4

        feature_names = np.array(vectorizer.get_feature_names_out())
        jd_tfidf = tfidf_matrix[1].toarray()[0]
        resume_tfidf = tfidf_matrix[0].toarray()[0]

        top_jd_indices = np.argsort(jd_tfidf)[::-1][:20]
        matched_kw = []
        missing_kw = []

        for idx in top_jd_indices:
            kw = feature_names[idx]
            if resume_tfidf[idx] > 0.01:
                matched_kw.append(kw)
            else:
                missing_kw.append(kw)

        return {
            "has_jd": True,
            "similarity_score": score,
            "cosine_similarity": round(sim, 3),
            "matched_keywords": matched_kw[:8],
            "missing_keywords": missing_kw[:8]
        }


# Singleton instances
_ROLE_CLASSIFIER: Optional[RoleClassifier] = None
_QUALITY_EVALUATOR: Optional[QualityEvaluator] = None


def get_role_classifier() -> RoleClassifier:
    global _ROLE_CLASSIFIER
    if _ROLE_CLASSIFIER is None:
        _ROLE_CLASSIFIER = RoleClassifier()
    return _ROLE_CLASSIFIER


def get_quality_evaluator() -> QualityEvaluator:
    global _QUALITY_EVALUATOR
    if _QUALITY_EVALUATOR is None:
        _QUALITY_EVALUATOR = QualityEvaluator()
    return _QUALITY_EVALUATOR


def compute_composite_score(
    rule_res: Dict[str, Any],
    role_preds: List[Dict[str, Any]],
    jd_res: Dict[str, Any],
    quality_res: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate overall weighted score and letter rating combining rules and ML."""
    sec_score = rule_res["sections"]["score"]          # 15%
    contact_score = rule_res["contact"]["score"]        # 10%
    quant_score = rule_res["quantification"]["score"]  # 20%
    skills_score = rule_res["skills"]["score"]          # 15%
    read_score = rule_res["readability"]["score"]       # 10%
    bench_score = quality_res["benchmark_similarity"]   # 15% (ML Benchmark Centroid)
    ml_pass_score = quality_res["pass_rate"]           # 15% (ML Classifier)

    total = (
        sec_score * 0.15 +
        contact_score * 0.10 +
        quant_score * 0.20 +
        skills_score * 0.15 +
        read_score * 0.10 +
        bench_score * 0.15 +
        ml_pass_score * 0.15
    )
    overall_score = int(round(total))

    if overall_score >= 90:
        grade = "S"
        level = "卓越 (大厂通过率极高)"
    elif overall_score >= 80:
        grade = "A"
        level = "良好 (具备强竞争力)"
    elif overall_score >= 70:
        grade = "B"
        level = "中等 (存在明确优化空间)"
    elif overall_score >= 60:
        grade = "C"
        level = "及格 (建议针对性调整)"
    else:
        grade = "D"
        level = "待完善 (核心要素缺失)"

    radar_scores = {
        "结构完整性": sec_score,
        "联系信息": contact_score,
        "量化与成果": quant_score,
        "技能树覆盖": skills_score,
        "大厂基准重合(ML)": bench_score,
        "初筛通过预测(ML)": ml_pass_score
    }

    return {
        "overall_score": overall_score,
        "grade": grade,
        "level": level,
        "radar": radar_scores
    }
