# Resume Scorer 简历智能评分与诊断系统

> 🎯 **基于「规则引擎 + 经典机器学习 (scikit-learn)」双驱动的简历质量评估与优化 Web 应用。**
> 
> 零大模型 API 消耗、毫秒级响应、高度可解释，直击招聘初筛与面试痛点。

---

## 🌟 核心特色与功能

1. **多格式文件极速解析**
   - 支持 **PDF**、**Word (.docx)** 与 **纯文本 (.txt)** 格式的拖拽上传与文本提取。
2. **六维规则评分引擎**
   - **结构完整性**：自动检测基本信息、教育背景、专业技能、实习工作、项目经验五大模块。
   - **联系方式规范**：正则精准识别手机号、电子邮箱。
   - **STAR 成果量化**：深度扫描主动行动词（主导/优化/重构等）与数据量化指标（百分比/耗时/QPS/倍数等）。
   - **技能栈覆盖度**：内置后端、前端、数据算法、测试运维、产品等领域的核心词库，提取匹配技能。
   - **篇幅与排版**：评估字符密度、字数区间与分段可读性。
3. **轻量机器学习（ML）模型群**
   - **大厂高分基准向量比对**：从优质通关样本中计算高维 TF-IDF 基准向量中心（Benchmark Centroid），通过余弦相似度计算候选人与名企优质简历的语义重合度。
   - **初筛通过率质量预测模型**：构建正负样本数据集（大厂高分简历 vs 常见硬伤缺陷简历），采用文本 TF-IDF + 结构化多维统计特征，训练集成分类模型，预测初筛通过概率。
   - **特征重要性溯源**：自动高亮简历中显著正向加分的高价值词汇（如：`核心(+0.30)`、`主导(+0.20)`、`架构` 等），以及被动消极警示词（如：`日常维护`、`协助` 等）。
   - **岗位方向自动识别**：基于 Softmax 逻辑回归预测当前简历最契合的职业发展赛道及概率分布。
   - **目标 JD 语义契合度**：基于向量空间模型计算简历与目标岗位 JD（Job Description）的余弦相似度，并输出匹配关键词与缺失项。
4. **现代化响应式 Web 界面**
   - 现代 SaaS 设计风格（Tailwind CSS + Chart.js 雷达图）。
   - 一键载入测试样例，开箱即测。

---

## 🛠️ 技术栈

- **后端开发**：`Python 3.10+` / `FastAPI` / `Uvicorn`
- **算法与 NLP**：`scikit-learn` / `jieba` / `NumPy` / `joblib`
- **文档解析**：`pypdf` / `python-docx`
- **前端页面**：`HTML5` / `Tailwind CSS (CDN)` / `Chart.js` / `FontAwesome` (免复杂前端打包编译，极简部署)

---

## 🚀 快速启动

### 方式 A：本地运行

1. 克隆代码或拉取最新分支：
   ```bash
   git clone https://github.com/FF-maker046/resume-scorer.git
   cd resume-scorer
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. (可选) 重新训练或更新机器学习模型：
   ```bash
   python train_model.py
   ```

4. 启动 Web 服务：
   ```bash
   python run.py
   # 或使用 py run.py
   ```

5. 打开浏览器访问：[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 方式 B：在 GitHub 浏览器中一键运行（GitHub Codespaces）

1. 进入本仓库页面 [FF-maker046/resume-scorer](https://github.com/FF-maker046/resume-scorer)；
2. 点击右上角绿色按钮 **`<> Code`** -> 切换到 **`Codespaces`**；
3. 点击 **`Create codespace on main`**（GitHub 个人用户每月享 60 小时免费额度）；
4. 在网页内打开的终端中输入：
   ```bash
   pip install -r requirements.txt
   python run.py
   ```
5. 终端会自动弹出端口预览链接，点击即可直接在浏览器使用！

---

## 📂 项目结构

```text
resume-scorer/
├── app/
│   ├── main.py              # FastAPI 入口路由与 API
│   ├── models/
│   │   └── quality_model.joblib # 训练生成的质量分类模型与大厂基准向量
│   ├── scorer/
│   │   ├── __init__.py
│   │   ├── parser.py        # PDF / Word / TXT 文本提取解析器
│   │   ├── rules.py         # 规则引擎（完整性、量化指标、技能词匹配）
│   │   └── ml_model.py      # 机器学习模块（质量分类器、岗位分类、JD 相似度）
│   └── static/
│       └── index.html       # 响应式 Web 前端界面与雷达图
├── data/
│   ├── build_dataset.py     # 正负样本数据集生成与扩充脚本
│   └── resumes_corpus.json  # 结构化标注简历语料集
├── train_model.py           # 模型训练、特征权重分析与评估指标脚本
├── requirements.txt         # 项目依赖列表
├── run.py                   # 本地一键启动脚本
├── .gitignore               # Git 忽略文件配置
└── README.md                # 项目详细说明文档
```

---

## 📄 开源许可
MIT License.
