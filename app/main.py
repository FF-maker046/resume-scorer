"""
FastAPI application entry point for the Resume Scorer.
"""
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.scorer.parser import extract_text
from app.scorer.rules import run_rule_evaluation
from app.scorer.project_evaluator import evaluate_all_projects
from app.scorer.ml_model import (
    get_role_classifier,
    get_quality_evaluator,
    JDMatcher,
    compute_composite_score
)


app = FastAPI(title="Resume Scorer (简历评分与智能优化)", version="1.0.0")

# Setup static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pre-warm classifier at startup
@app.on_event("startup")
async def startup_event():
    get_role_classifier()
    get_quality_evaluator()


@app.get("/")
async def root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Resume Scorer API is running. Place index.html in app/static."}


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "Resume Scorer", "version": "1.0.0"}


@app.get("/api/sample")
async def get_sample():
    """Return realistic sample resume and JD for one-click testing."""
    sample_resume = """张三 | 求职意向：后端开发工程师
联系电话：13812345678 | 电子邮箱：zhangsan_dev@163.com | 城市：北京

【教育背景】
2020.09 - 2024.06  北京科技大学  软件工程专业  本科 (GPA: 3.8/4.0)

【专业技能】
* 熟练掌握 Java 语言、JVM 原理、多线程并发编程与设计模式
* 熟练使用 Spring Boot, Spring Cloud, MyBatis 进行微服务架构开发
* 精通 MySQL 数据库设计与 SQL 调优，熟练掌握 Redis 缓存机制及分布式锁
* 熟悉 Kafka 消息中间件、Docker 容器化及 Linux 常用运维命令
* 熟悉 Python, Git, CI/CD 持续集成流水线

【工作与实习经历】
2023.07 - 2024.03  某知名互联网科技有限公司  Java 后端研发实习生
* 独立负责订单履约模块重构，运用策略模式消除 20+ 冗余分支判断，代码可维护性显著提升。
* 主导分布式缓存架构优化，引入 Redis 多级缓存与布隆过滤器，彻底解决缓存穿透问题，接口平均响应时间从 180ms 降低至 35ms。
* 协同团队推进核心数据库分库分表改造，承载日均 500 万+ 订单流水，QPS 峰值突破 12000，保障大促期间零故障运行。

【项目经历】
高并发电商秒杀与库存扣减系统  后端核心开发
* 技术栈：Spring Boot + Redis + Kafka + MySQL + Docker
* 针对超卖与高并发瓶颈，设计并落地「Redis预扣减 + Kafka异步落库 + 本地缓存」三层架构，将数据库写入压力降低 85%。
* 编写压测用例，使用 JMeter 进行高并发压力测试，单机 TPS 稳定提升至 8500+，系统可用性达到 99.99%。
"""

    sample_jd = """岗位：Java 后端开发工程师
职责要求：
1. 本科及以上学历，计算机或软件工程相关专业；
2. 熟练掌握 Java 核心基础、多线程高并发编程及常见设计模式；
3. 熟练掌握 Spring Boot, Spring Cloud, MyBatis 等微服务框架；
4. 熟练掌握 MySQL 数据库索引优化及 Redis 高性能缓存方案；
5. 具备高并发系统设计经验，熟悉 Kafka 消息队列与 Docker 容器技术者优先；
6. 具备良好的工程素养、代码规范和团队协作能力。
"""
    return {
        "sample_resume": sample_resume.strip(),
        "sample_jd": sample_jd.strip()
    }


@app.post("/api/score")
async def score_resume(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    jd: Optional[str] = Form(None),
    mode: Optional[str] = Form("auto")
):
    """Core scoring endpoint accepting file or text, and optional mode (auto/campus/social)."""
    resume_text = ""

    if file and file.filename:
        file_bytes = await file.read()
        resume_text = extract_text(file_bytes, file.filename)
    elif text and text.strip():
        resume_text = text.strip()

    if not resume_text or len(resume_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="未检测到有效简历文本，请粘贴内容或上传支持的文件 (PDF / Word / TXT)")

    # 1. Rule Engine Evaluation
    rule_results = run_rule_evaluation(resume_text)

    # 2. ML Role Classifier & Quality Evaluator
    classifier = get_role_classifier()
    role_predictions = classifier.predict(resume_text)

    quality_evaluator = get_quality_evaluator()
    quality_results = quality_evaluator.evaluate(resume_text, mode=mode or "auto")

    # 3. JD Semantic Matcher
    jd_results = JDMatcher.match(resume_text, jd or "")

    # 4. Project Deep 4D Evaluation (Difficulty, Authenticity, Completeness, Relevance)
    target_role = role_predictions[0]["role"] if role_predictions else "数据与算法"
    project_results = evaluate_all_projects(resume_text, target_role=target_role)

    # 5. Composite Scoring
    composite = compute_composite_score(
        rule_results,
        role_predictions,
        jd_results,
        quality_results,
        project_results
    )

    return JSONResponse({
        "success": True,
        "resume_length": len(resume_text),
        "overall": composite,
        "rules": rule_results,
        "projects": project_results,
        "ml": {
            "roles": role_predictions,
            "jd": jd_results,
            "quality": quality_results
        }
    })
