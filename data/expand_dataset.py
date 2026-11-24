"""
Dataset Expansion Script:
Fetches real open-source resume samples from GitHub (geekcompany/ResumeSample),
integrates student/campus AI, Data Science, and algorithmic contest profiles,
and builds an expanded balanced training dataset (60+ samples).
Saved to data/resumes_corpus.json.
"""
import os
import json
import re
import urllib.request

OPEN_SOURCE_GITHUB_FILES = {
    "后端架构师": "https://raw.githubusercontent.com/geekcompany/ResumeSample/master/architect.md",
    "Java后端开发": "https://raw.githubusercontent.com/geekcompany/ResumeSample/master/java.md",
    "前端工程师": "https://raw.githubusercontent.com/geekcompany/ResumeSample/master/web.md",
    "Node全栈开发": "https://raw.githubusercontent.com/geekcompany/ResumeSample/master/node.md",
    "C++系统研发": "https://raw.githubusercontent.com/geekcompany/ResumeSample/master/c.md",
    "Android客户端": "https://raw.githubusercontent.com/geekcompany/ResumeSample/master/android.md",
    "iOS客户端": "https://raw.githubusercontent.com/geekcompany/ResumeSample/master/ios.md",
    "PHP服务端": "https://raw.githubusercontent.com/geekcompany/ResumeSample/master/php.md"
}

# Clean JobDeer template hints and comments
def clean_markdown_resume(text: str) -> str:
    # Remove ``` (notes) ```
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove HTML links markdown like [text](http...)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove extra hashes and lines
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("本简历模板由")]
    return "\n".join(lines)


# Rich Student / Campus Recruitment Samples
STUDENT_CAMPUS_SAMPLES = [
    # 优质校招 1：数据科学与大数据（深度学习/对抗样本/LLM）
    {
        "role": "数据与算法",
        "label": 1,
        "is_student": True,
        "title": "数据科学与大数据技术本科生（AI攻防与大模型应用）",
        "text": """李凯峰 | 求职意向：AI算法研发 / 数据挖掘实习生
联系电话：13660002605 | 电子邮箱：1879043881@qq.com | 城市：广东
【教育背景】
2025.09 - 至今 汕头大学 数据科学与大数据技术（本科）
GPA：4.14 / 5.00（专业排名前 2.67%）
【专业技能】
* 熟练掌握 Python 编程语言，熟悉面向对象设计与常用数据结构
* 深入理解深度学习核心原理，熟练使用 PyTorch 搭建卷积神经网络与分类模型
* 熟悉大模型应用开发，掌握 Streamlit 交互框架、智谱 GLM API 调用与 FAISS 向量检索
* 熟练使用 Pandas, NumPy, Matplotlib 进行数据清洗、探索性数据分析（EDA）与可视化
* 熟练使用 Git 进行代码版本控制，具备良好的开源协作与文档编写习惯（GitHub 配有完整 README）
【项目经历】
1. FGSM 对抗攻击演示系统（核心开发）
* 基于 PyTorch 从零实现 FGSM 快速梯度符号对抗攻击算法，生成对抗扰动样本。
* 编写可视化评估模块，展示攻击前后模型分类准确率变化（攻击后模型错误率提升 65%），并输出图文说明文档。
* 项目完整开源至 GitHub，包含完整环境复现指南与数学原理推导。
2. LLM 智能问答助手（独立全流程开发）
* 技术栈：Python + Streamlit + 智谱GLM API + FAISS
* 独立开发支持多轮对话与本地长文档 RAG 问答的 AI 助手，用户可上传 PDF 针对内容精准提问。
* 基于 FAISS 构建文本嵌入向量库，实现 Top-K 语义分块召回，大幅缓解大模型幻觉现象。
3. 豆瓣电影 TOP250 数据采集与清洗预处理
* 编写高效爬虫脚本采集 250 部经典电影数据，处理空缺值、统一量纲格式，生成结构化数据集。
【竞赛与荣誉】
* 第十七届蓝桥杯软件赛 Python 软件设计省三等奖（2026.05）
【自我评价】
具备强烈的自主学习与动手实践能力，暑假期间从零自学 PyTorch 与 Streamlit，两周内独立交付上线完整 Web 原型。对对抗攻防与大模型技术有浓厚学术热情。"""
    },
    # 优质校招 2：计算机科学与技术本科生（ACM/算法竞赛）
    {
        "role": "数据与算法",
        "label": 1,
        "is_student": True,
        "title": "计算机科学本科生（算法竞赛/ACM银奖/推荐系统）",
        "text": """林浩 | 意向岗位：算法工程师（校招）
电话：13811223344 | 邮箱：linhaocse@zju.edu.cn | 杭州
【教育背景】
2022.09 - 2026.06 浙江大学 计算机科学与技术 本科 GPA: 3.92/4.00 (Top 5%)
【核心技能】
* 算法与数据结构功底扎实，熟练掌握动态规划、图论算法、高并发调度
* 精通 C++, Python，熟练使用 PyTorch, Scikit-Learn
* 熟悉推荐系统协同过滤、矩阵分解与深度召回模型（DSSM, DIN）
【竞赛获奖】
* ICPC 国际大学生程序设计竞赛 区域赛银奖（2024.11）
* 全国大学生数学建模竞赛（全国一等奖，2023.09）
* 连续三年获浙江大学学业一等奖学金
【科研与项目】
基于注意力机制的新闻点击率 CTR 预估系统
* 针对用户历史点击序列，设计两阶段 Transformer 行为聚合网络，在公开新闻数据集上 AUC 达到 0.812，相比基线提升 3.4%。
* 使用 C++ 封装高性能推理算子，将单次召回打分延迟降低至 8ms 以内。"""
    },
    # 优质校招 3：软件工程本科生（全栈与微服务开发）
    {
        "role": "后端开发",
        "label": 1,
        "is_student": True,
        "title": "软件工程本科生（高校选课与分布式校园系统）",
        "text": """王俊杰 | 求职意向：Java后端开发（2026届校招）
联系方式：13599887766 | 邮箱：wangjj_soft@whu.edu.cn | 武汉
【教育背景】
2022.09 - 2026.06 武汉大学 软件工程专业 本科（主修 GPA: 3.86/4.0）
【专业技能】
* 熟练掌握 Java 语言、JVM 垃圾回收机制、类加载流程与 JUC 多线程并发
* 熟练使用 Spring Boot 3, Spring Cloud, MyBatis-Plus 进行微服务开发
* 精通 MySQL 数据库索引原理、Explain 执行计划调优，熟练使用 Redis 缓存
* 掌握 Docker 容器技术与 Linux 常用命令，熟悉 Git 与敏捷协同
【项目经历】
分布式校园抢课与学分结算系统（国家级大学生创新创业训练项目）
* 针对选课高并发瞬间打垮数据库的痛点，设计「Redis分布式锁预扣减 + RabbitMQ异步入库」架构，承载全校 20000+ 人瞬时并发抢课，系统零崩溃零超卖。
* 编写 JMeter 脚本对核心接口进行压测，优化后并发吞吐 TPS 从 800 提升至 5200+，接口耗时下降 70%。
* 获国家计算机软件著作权登记 1 项（第一发明人）。"""
    },
    # 优质校招 4：前端与跨端应用开发
    {
        "role": "前端开发",
        "label": 1,
        "is_student": True,
        "title": "前端开发校招（Vue3/Vite/开源组件库）",
        "text": """刘雨婷 | 求职意向：前端开发工程师（校招/实习）
电话：13788776655 | 邮箱：liuyuting_fe@163.com | 深圳
【教育背景】
2022.09 - 2026.06 华南理工大学 软件工程 本科
【专业技能】
* 熟练掌握 HTML5, CSS3, JavaScript ES6+，熟练使用 TypeScript
* 深入掌握 Vue3 组合式 API (Composition API)、Pinia 状态管理与 Vite 构建
* 熟练运用 Tailwind CSS, ECharts 可视化图表库，掌握响应式布局与移动端适配
* 熟练使用 Git，具备良好的代码审查和开源维护习惯
【项目经历】
高校开源大屏数据可视化与报表平台
* 基于 Vue3 + TypeScript + ECharts 独立搭建数据看板，封装 15 个通用业务图表组件。
* 优化首屏加载性能，引入路由懒加载与 Gzip 压缩，打包体积减小 55%，LCP 优化至 0.9 秒。
* 获得全国大学生计算机设计大赛中南赛区一等奖。"""
    },
    # 优质校招 5：测试开发与质量工程
    {
        "role": "测试与运维",
        "label": 1,
        "is_student": True,
        "title": "测试开发工程师校招（自动化测试/Pytest）",
        "text": """张博文 | 求职：测试开发工程师
手机：13922334455 | 邮箱：zhangbw_test@126.com | 成都
【教育经历】
2022.09 - 2026.06 电子科技大学 自动化专业 本科
【技能特长】
* 熟练使用 Python, Pytest 搭建接口与 Web UI 自动化回归测试框架
* 熟悉 Postman, JMeter 性能并发压力测试，掌握测试用例设计规范（边界值、等价类）
* 熟悉 Linux 基础命令、Git、Docker 与持续集成 CI/CD 流程
【项目经历】
校园一卡通微服务自动化回归测试平台
* 针对校园卡充值、消费、门禁等 30+ 核心微服务接口，设计并实现自动化接口测试用例 400+ 条。
* 结合 GitLab CI 接入每日自动构建与定时回归，缺陷遗漏率降低 80%，节省人工回归测试耗时 75%。"""
    }
]

# Defective / Negative Samples (Real campus and social mistakes)
NEGATIVE_SAMPLES_EXPANDED = [
    {
        "role": "数据与算法",
        "label": 0,
        "is_student": True,
        "title": "学生缺陷样本：无项目细节与数据",
        "text": """李某某，数据科学专业大三学生。
电话：13000000000
想找一份关于人工智能或者大模型相关的实习。
在学校上过 Python 课和机器学习课。平时喜欢看一些技术博客。
做过一些课程作业，会用一点 PyTorch。
希望老师能给一个机会，我一定会努力学习的。"""
    },
    {
        "role": "后端开发",
        "label": 0,
        "is_student": True,
        "title": "学生缺陷样本：大段空话无技术栈",
        "text": """王某 计算机系在读
邮箱：test12345@qq.com
【自我介绍】
本人品行端正，热爱计算机事业。
在校期间积极参加社团活动，担任学生会干部，组织协调能力强。
对待工作认真负责，吃苦耐劳。
希望能在贵公司从基础做起，学到更多东西。"""
    },
    {
        "role": "前端开发",
        "label": 0,
        "is_student": True,
        "title": "学生缺陷样本：缺失联系方式与量化成果",
        "text": """前端实习简历
教育经历：某大学 计算机专业
做过一个商城前端界面，写过 HTML 和 CSS。
也用过 Vue 做了一个简单的留言板。
页面能够正常打开，实现了基本的功能交互。"""
    },
    {
        "role": "测试与运维",
        "label": 0,
        "is_student": True,
        "title": "学生缺陷样本：仅罗列课程名称",
        "text": """张某某 测开意向
手机：13899998888 邮箱：zhang@test.com
【在校课程】
大学物理、高等数学、线性代数、C语言程序设计、操作系统、计算机网络、数据库原理。
每门课程均顺利通过。
熟悉 Office 办公软件使用。"""
    },
    {
        "role": "产品与运营",
        "label": 0,
        "is_student": True,
        "title": "学生缺陷样本：通篇被动描述",
        "text": """赵某 产品助理求职
电话：13600001111 邮箱：zhaopm@163.com
在学校参加过创业比赛。
听从组长安排，负责帮大家写写材料，收集一下问卷。
平时帮老师做做PPT，整理一下班级同学的通讯录表格。
为人随和，服从组织安排。"""
    }
]


def build_expanded_corpus():
    os.makedirs("data", exist_ok=True)
    out_file = os.path.join("data", "resumes_corpus.json")

    positive_list = []
    negative_list = []

    # 1. Fetch real GitHub developer markdown resumes
    print("Fetching open-source developer resumes from GitHub...")
    for title, url in OPEN_SOURCE_GITHUB_FILES.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Resume-Builder/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                raw_md = resp.read().decode("utf-8")
                cleaned = clean_markdown_resume(raw_md)
                if len(cleaned) > 400:
                    role = "后端开发"
                    if "前端" in title or "web" in url or "node" in url:
                        role = "前端开发"
                    elif "c.md" in url:
                        role = "后端开发"
                    positive_list.append({
                        "role": role,
                        "label": 1,
                        "is_student": False,
                        "title": f"GitHub开源优秀简历: {title}",
                        "text": cleaned
                    })
                    print(f"  [+] Added: {title} ({len(cleaned)} chars)")
        except Exception as e:
            print(f"  [-] Failed to fetch {title}: {e}")

    # 2. Add Student & Campus samples
    for sample in STUDENT_CAMPUS_SAMPLES:
        positive_list.append(sample)
    print(f"  [+] Added {len(STUDENT_CAMPUS_SAMPLES)} Campus / Student AI & Data Science samples.")

    # 3. Add existing positive seeds if present
    existing_path = os.path.join("data", "resumes_corpus.json")
    if os.path.exists(existing_path):
        try:
            with open(existing_path, "r", encoding="utf-8") as f:
                old = json.load(f)
                for item in old.get("positive", []):
                    if not any(item["title"] == p["title"] for p in positive_list):
                        positive_list.append(item)
                for item in old.get("negative", []):
                    if not any(item["title"] == n["title"] for n in negative_list):
                        negative_list.append(item)
        except Exception:
            pass

    # 4. Add Defective samples
    for neg in NEGATIVE_SAMPLES_EXPANDED:
        if not any(neg["title"] == n["title"] for n in negative_list):
            negative_list.append(neg)

    full_data = {
        "positive": positive_list,
        "negative": negative_list,
        "total_count": len(positive_list) + len(negative_list),
        "positive_count": len(positive_list),
        "negative_count": len(negative_list)
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Expanded corpus built successfully: {out_file}")
    print(f"   Total samples: {full_data['total_count']} (Positive: {full_data['positive_count']}, Negative: {full_data['negative_count']})")


if __name__ == "__main__":
    build_expanded_corpus()
