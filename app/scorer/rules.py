"""
Rule-based resume evaluation engine.
Extracts structural features, checks completeness, quantification (STAR), and skills.
"""
import re
from typing import Dict, List, Any


SKILL_CATEGORIES = {
    "后端开发": [
        "Java", "Spring", "SpringBoot", "Spring Boot", "MyBatis", "Python", "FastAPI", "Django", "Flask",
        "Golang", "Go", "C++", "C#", "Linux", "MySQL", "PostgreSQL", "Redis", "Kafka", "RabbitMQ",
        "Docker", "Kubernetes", "K8s", "微服务", "RPC", "gRPC", "Nginx", "Netty", "JVM", "RESTful", "Elasticsearch"
    ],
    "前端开发": [
        "HTML", "CSS", "JavaScript", "TypeScript", "ES6", "React", "Vue", "Vue3", "Angular",
        "Webpack", "Vite", "Tailwind", "Less", "Sass", "Node.js", "Next.js", "Nuxt", "Pinia", "Redux",
        "小程序", "微信小程序", "响应式", "Canvas", "ECharts"
    ],
    "数据与算法": [
        "机器学习", "深度学习", "NLP", "自然语言处理", "计算机视觉", "CV", "PyTorch", "TensorFlow", "scikit-learn",
        "Sklearn", "Pandas", "NumPy", "Spark", "Flink", "Hadoop", "Hive", "SQL", "特征工程",
        "数据挖掘", "数据分析", "推荐系统", "大模型", "LLM", "Prompt", "Transformer", "BERT"
    ],
    "测试与运维": [
        "测试用例", "Selenium", "Pytest", "Postman", "CI/CD", "Jenkins", "Git", "GitHub", "GitLab",
        "自动化测试", "单元测试", "接口测试", "性能测试", "JMeter", "Prometheus", "Grafana", "Shell", "Ansible"
    ],
    "产品与运营": [
        "产品设计", "需求分析", "流程图", "Axure", "Figma", "原型设计", "用户调研", "竞品分析",
        "A/B测试", "敏捷开发", "Scrum", "PRD", "用户增长", "埋点分析"
    ]
}

# Strong action verbs common in high-quality resumes
ACTION_VERBS = [
    "负责", "主导", "核心", "搭建", "设计", "重构", "优化", "研发", "实现", "开发",
    "推进", "落地", "提升", "降低", "突破", "上线", "解决", "统筹", "协同", "主推"
]

# Quantified metric patterns (numbers with metrics)
QUANT_PATTERNS = [
    r"\d+(?:\.\d+)?%",               # 30%, 99.9%, 2.67%
    r"\d+(?:\.\d+)?\s*(?:倍|折)",      # 2倍, 1.5倍
    r"\d+(?:\.\d+)?\s*(?:万|千|亿)",   # 10万, 500千
    r"\d+\s*(?:ms|毫秒|s|秒)",        # 50ms, 200毫秒
    r"\bQPS\s*(?:达|达到|突破)?\s*\d+", # QPS 1000
    r"\d+\s*(?:个|项|人|篇|套|名|行|次|周|天|月|年|轮|门)", # 5项, 20人, 5个, 2周
    r"GPA[：:]\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?", # GPA: 4.14/5.00
    r"(?:前|TOP|Top|top)\s*\d+(?:\.\d+)?%?",        # 前2.67%, TOP250
    r"(?:一|二|三|特)等奖",                         # 三等奖
    r"(?:提升|降低|减少|增长|缩短|节省|突破|承载|交付)\s*\d+", # 提升20
]


def evaluate_contact(text: str) -> Dict[str, Any]:
    """Check for presence of contact info (phone and email)."""
    # Resilient phone pattern supporting optional +86 and separators
    phone_pattern = r"(?:(?:\+|00)86\s*[- ]?)?1[3-9]\d(?:\s*|\-)\d{4}(?:\s*|\-)\d{4}\b|\b1[3-9]\d{9}\b|\b\d{3}[-\s]\d{4}[-\s]\d{4}\b"
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    phones = re.findall(phone_pattern, text)
    emails = re.findall(email_pattern, text)

    score = 0
    items = []
    if phones:
        score += 50
        # clean phone
        clean_phone = re.sub(r"[\s\-]", "", phones[0])
        items.append(f"电话: {clean_phone}")
    if emails:
        score += 50
        items.append(f"邮箱: {emails[0]}")

    return {
        "score": score,
        "has_phone": bool(phones),
        "has_email": bool(emails),
        "details": items
    }


def evaluate_sections(text: str) -> Dict[str, Any]:
    """Check whether core resume sections exist."""
    section_patterns = {
        "基本信息": [r"基本信息", r"个人信息", r"个人资料", r"求职意向", r"籍贯", r"出生年月", r"姓\s*名", r"联系方式"],
        "教育背景": [r"教育背景", r"教育经历", r"学历", r"毕业院校", r"本科", r"硕士", r"博士", r"大学", r"学院"],
        "专业技能": [r"专业技能", r"技能清单", r"技能特长", r"IT技能", r"掌握技术", r"技术栈", r"编程语言"],
        "项目经历": [r"项目经历", r"项目经验", r"项目介绍", r"主要项目", r"核心项目", r"开发经历"],
        "实践与实习": [
            r"工作经历", r"工作经验", r"实习经历", r"实习经验", r"任职经历", r"职业经历",
            r"获奖证书", r"荣誉奖项", r"竞赛获奖", r"自我评价", r"校园经历"
        ]
    }

    found = []
    missing = []
    for sec_name, pats in section_patterns.items():
        matched = any(re.search(pat, text, re.IGNORECASE) for pat in pats)
        if matched:
            found.append(sec_name)
        else:
            missing.append(sec_name)

    # 5 sections, 20 pts each
    score = int((len(found) / len(section_patterns)) * 100)
    return {
        "score": score,
        "found": found,
        "missing": missing,
        "total_sections": len(section_patterns)
    }


def evaluate_quantification(text: str) -> Dict[str, Any]:
    """Check STAR quantification: numbers, metrics, and action verbs."""
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 8]
    if not lines:
        lines = [text]

    quant_hits = []
    action_hits = []
    star_sentences = []

    for line in lines:
        has_quant = any(re.search(p, line) for p in QUANT_PATTERNS)
        has_action = any(v in line for v in ACTION_VERBS)

        if has_quant:
            quant_hits.append(line)
        if has_action:
            action_hits.append(line)
        if has_quant and has_action:
            star_sentences.append(line)

    quant_count = len(quant_hits)
    action_count = len(action_hits)
    star_count = len(star_sentences)

    # Scoring algorithm:
    # Action verbs score up to 40, Quantified metrics score up to 40, STAR combo scores up to 20
    action_score = min(40, action_count * 8)
    quant_score = min(40, quant_count * 10)
    star_score = min(20, star_count * 10)
    total_score = min(100, action_score + quant_score + star_score)

    return {
        "score": total_score,
        "quant_count": quant_count,
        "action_count": action_count,
        "star_count": star_count,
        "sample_star": star_sentences[:3]
    }


def evaluate_readability_and_length(text: str) -> Dict[str, Any]:
    """Evaluate text volume, paragraph structure, and density."""
    char_count = len(re.sub(r"\s+", "", text))
    lines = [l for l in text.split("\n") if l.strip()]
    line_count = len(lines)

    score = 100
    tips = []

    if char_count < 300:
        score -= 40
        tips.append("简历字数过少（不足300字），内容单薄，建议扩充项目与实习细节。")
    elif char_count < 600:
        score -= 15
        tips.append("简历内容偏短（不足600字），建议补充关键成果与专业技能。")
    elif char_count > 3500:
        score -= 20
        tips.append("简历篇幅过长（超过3500字），建议精简，保留1~2页核心亮点。")
    else:
        tips.append("简历篇幅适中，字数在推荐区间内。")

    if line_count < 10 and char_count >= 500:
        score -= 15
        tips.append("段落划分较少，容易造成阅读疲劳，建议使用要点符号（Bullet points）分条陈述。")

    score = max(20, min(100, score))
    return {
        "score": score,
        "char_count": char_count,
        "line_count": line_count,
        "tips": tips
    }


def extract_skills(text: str) -> Dict[str, Any]:
    """Match hard skill keywords by job category."""
    matched_by_cat = {}
    all_matched = set()

    for category, skill_list in SKILL_CATEGORIES.items():
        cat_matches = []
        for skill in skill_list:
            # Word boundary check for short English words, simple substring for Chinese/symbols
            if re.match(r"^[A-Za-z0-9+#.]+$", skill):
                pattern = r"(?i)(?:\b|_)" + re.escape(skill) + r"(?:\b|_)"
            else:
                pattern = re.escape(skill)

            if re.search(pattern, text, re.IGNORECASE):
                cat_matches.append(skill)
                all_matched.add(skill)

        if cat_matches:
            matched_by_cat[category] = cat_matches

    # Skill breadth score
    total_skills = len(all_matched)
    if total_skills >= 10:
        skill_score = 95
    elif total_skills >= 7:
        skill_score = 85
    elif total_skills >= 4:
        skill_score = 75
    elif total_skills >= 1:
        skill_score = 60
    else:
        skill_score = 30

    return {
        "score": skill_score,
        "total_count": total_skills,
        "by_category": matched_by_cat,
        "all_skills": sorted(list(all_matched))
    }


# Competition Grading Database (Based on MOE White List and Industry Hiring Consensus)
COMPETITION_EVALUATIONS = [
    {
        "id": "acm",
        "detect": r"(?:ACM(?:-ICPC)?|ICPC|CCPC)",
        "tiers": [
            (r"(?:金奖|银奖|铜奖|金牌|银牌|铜牌|区域赛|总决赛|出线|全国)", "T1", "ACM/ICPC 区域赛或总决赛荣誉", "算法界硬通货，技术大厂初筛免试直通车，具备压倒性编程算法竞争力。"),
            (r".*", "T3", "ACM/ICPC 校级选拔或参赛", "体现了对高难度算法竞赛的追求，建议继续冲刺区域赛奖项。")
        ]
    },
    {
        "id": "lanqiao",
        "detect": r"蓝桥杯",
        "tiers": [
            (r"(?:国赛|全国总决赛)[^\n]{0,20}(?:一等奖|国一)", "T1", "蓝桥杯全国一等奖", "国家级算法竞赛最高荣誉之一，在各大厂算法岗具备很强认可度。"),
            (r"(?:国赛|全国总决赛)[^\n]{0,20}(?:二等奖|三等奖|国二|国三)", "T2", "蓝桥杯全国二/三等奖", "教育部白名单国家级总决赛荣誉，在校招初筛中具有明确加分效应。"),
            (r"(?:省(?:级)?一等奖|省一)", "T2", "蓝桥杯省一等奖 (晋级全国总决赛)", "省级最高奖并入围国赛，在校招技术岗具备良好的算法基本功证明。"),
            (r"(?:省(?:级)?二等奖|省二)", "T3", "蓝桥杯省二等奖", "白名单良好进阶奖项，具备一定算法思维，但在大厂算法岗竞争中区分度居中。"),
            (r"(?:省(?:级)?三等奖|省三|优秀奖)", "T4", "蓝桥杯省三等奖 / 普及型奖项", "行业公认蓝桥杯省三获奖比例大、门槛较低，各大厂初筛与技术面试官普遍视为普及参与型经历，几乎不产生实质加分。建议在简历中弱化此奖项版面，重点展示高难度工程项目，或冲刺省一及更高梯度赛事。"),
            (r".*", "T4", "蓝桥杯参赛经历", "算法参赛基础经历，建议补充最终奖项或冲刺更高名次。")
        ]
    },
    {
        "id": "math_model",
        "detect": r"(?:数学建模|数模|MCM|ICM)",
        "tiers": [
            (r"(?:O奖|Outstanding|全国特等奖)", "T1", "数学建模顶尖国际奖项 (O奖)", "全球数模最高荣誉，对量化、数据科学与算法岗具有极强背书。"),
            (r"(?:国一|国二|国家一等奖|国家二等奖|M奖|F奖|Finalist|Meritorious)", "T2", "数学建模国家级/美赛一等奖", "国家白名单核心荣誉，体现出色复杂问题数学建模与团队攻坚能力。"),
            (r"(?:省一|省二|省级一等奖|省级二等奖|H奖|Honorable)", "T3", "数学建模省级一/二等奖 / 美赛H奖", "良好的学术与建模实践，但在大厂核心算法岗区分度中等。"),
            (r".*", "T4", "数学建模省三或参赛", "参与型经历，更注重后续项目实际产出。")
        ]
    },
    {
        "id": "csp",
        "detect": r"(?:CCF\s*CSP|CCSP)",
        "tiers": [
            (r"(?:3[0-9]{2}|4[0-9]{2}|500|前\s*[1-5]%|CCSP)", "T1", "CCF CSP 300+高分 / CCSP荣誉", "计算机学会官方权威认证，大厂免笔试硬通货。"),
            (r"(?:2[0-9]{2}|前\s*10%)", "T2", "CCF CSP 200+良好", "具备扎实的数据结构与代码实现底子。"),
            (r".*", "T3", "CCF CSP 基础认证", "建议向 300 分以上冲刺以获取大厂笔试免试。")
        ]
    },
    {
        "id": "design_contest",
        "detect": r"(?:计算机设计大赛|天梯赛|互联网\+|创新大赛|挑战杯)",
        "tiers": [
            (r"(?:国家级?|全国|特等奖|金奖|银奖|国一|国二)", "T2", "国家级白名单竞赛荣誉", "教育部白名单国家级竞赛，体现出色的工程综合素质与项目落地能力。"),
            (r"(?:省(?:级)?金奖|省(?:级)?银奖|省(?:级)?一等奖|省(?:级)?二等奖|铜奖)", "T3", "省级白名单竞赛荣誉", "体现了良好的课外实践能力，建议突出个人核心贡献。"),
            (r".*", "T4", "白名单竞赛参与/省三", "基础参赛荣誉，建议在简历中简短带过，将主要版面留给核心技术项目。")
        ]
    }
]


def evaluate_competitions(text: str) -> List[Dict[str, Any]]:
    """Scan and evaluate competition prestige based on MOE White List."""
    hits = []
    lines = text.split("\n")
    for comp in COMPETITION_EVALUATIONS:
        for line in lines:
            if re.search(comp["detect"], line, re.I):
                for pat, tier, name, comment in comp["tiers"]:
                    if re.search(pat, line, re.I):
                        matched = {
                            "tier": tier,
                            "name": name,
                            "comment": comment,
                            "raw_text": line.strip()
                        }
                        if not any(h["name"] == matched["name"] for h in hits):
                            hits.append(matched)
                        break
    return hits


def run_rule_evaluation(text: str) -> Dict[str, Any]:
    """Run all rule engines and generate structured diagnostics & suggestions."""
    contact = evaluate_contact(text)
    sections = evaluate_sections(text)
    quant = evaluate_quantification(text)
    readability = evaluate_readability_and_length(text)
    skills = extract_skills(text)
    competitions = evaluate_competitions(text)

    # Diagnostic suggestions
    suggestions: List[Dict[str, str]] = []

    # 1. Contact
    if not contact["has_phone"]:
        suggestions.append({
            "type": "warning",
            "title": "缺少联系电话",
            "desc": "简历中未检测到标准手机号码，HR 无法电话联系您，请在醒目位置标注联系电话。"
        })
    if not contact["has_email"]:
        suggestions.append({
            "type": "warning",
            "title": "缺少电子邮箱",
            "desc": "简历中未检测到有效邮箱地址，常用于接收面试邀请与 Offer，建议补充。"
        })
    if contact["has_phone"] and contact["has_email"]:
        suggestions.append({
            "type": "success",
            "title": "联系方式完备",
            "desc": f"已成功识别到候选人有效联系方式（{', '.join(contact['details'])}）。"
        })

    # 2. Sections
    if sections["missing"]:
        for sec in sections["missing"]:
            suggestions.append({
                "type": "warning" if "实践" in sec else "error",
                "title": f"建议补充「{sec}」模块",
                "desc": f"标准简历应具备清晰的板块标题，检测到未包含明显的「{sec}」，补充后可让经历更具说服力。"
            })
    else:
        suggestions.append({
            "type": "success",
            "title": "简历核心架构完整",
            "desc": f"已完整覆盖基本信息、教育背景、专业技能、项目经历与实践模块等核心板块。"
        })

    # 3. Quantification & STAR
    if quant["quant_count"] < 2:
        suggestions.append({
            "type": "error",
            "title": "成果缺乏数字化量化指标",
            "desc": "项目中缺乏具体的数字支撑（如：提升XX%、缩短XXms、承载XX用户等）。建议采用 STAR 法则，多用量化数据突出业务价值。"
        })
    elif quant["quant_count"] < 4:
        suggestions.append({
            "type": "suggestion",
            "title": "建议进一步增加量化成果",
            "desc": "已检测到部分量化数据，但在核心项目经历中可进一步增加比例、耗时、收益等量化说明。"
        })

    if quant["action_count"] < 3:
        suggestions.append({
            "type": "suggestion",
            "title": "建议多使用主动行动词",
            "desc": "尽量避免‘参与了某某’等被动描述，多使用‘主导架构设计’、‘独立负责重构’、‘优化并落地’等强势动词。"
        })

    # 4. Readability
    for tip in readability["tips"]:
        if "单薄" in tip or "过少" in tip or "过长" in tip:
            suggestions.append({"type": "warning", "title": "篇幅与排版优化", "desc": tip})

    # 5. Skills
    if skills["total_count"] < 4:
        suggestions.append({
            "type": "warning",
            "title": "专业技能栈覆盖偏低",
            "desc": "识别出的硬技能关键词较少，建议在技能清单中结构化列出掌握的语言、框架、数据库及工具。"
        })
    else:
        suggestions.append({
            "type": "success",
            "title": "技术栈覆盖清晰",
            "desc": f"成功识别出 {skills['total_count']} 项关键技术技能，覆盖分类: {', '.join(skills['by_category'].keys())}。"
        })

    # 6. Academic, Competition & Open-Source Highlights (Key for Campus & Tech)
    if re.search(r"GPA[：:]\s*(?:3\.[7-9]|4\.[0-9]|[89]\d)/|前\s*\d+(?:\.\d+)?%|Top\s*\d+(?:\.\d+)?%", text, re.I):
        suggestions.append({
            "type": "success",
            "title": "优异学业成绩背书",
            "desc": "检测到优异的绩点（GPA）或突出的专业排名前列，在校招与科研初筛中具备强大学术竞争力。"
        })

    # Nuanced Competition Tier Evaluation
    if competitions:
        for c in competitions:
            tier = c["tier"]
            if tier == "T1":
                suggestions.append({
                    "type": "success",
                    "title": f"竞赛含金量顶尖 (T1): {c['name']}",
                    "desc": c["comment"]
                })
            elif tier == "T2":
                suggestions.append({
                    "type": "success",
                    "title": f"竞赛含金量高 (T2): {c['name']}",
                    "desc": c["comment"]
                })
            elif tier == "T3":
                suggestions.append({
                    "type": "suggestion",
                    "title": f"竞赛进阶建议 (T3): {c['name']}",
                    "desc": c["comment"]
                })
            elif tier == "T4":
                suggestions.append({
                    "type": "warning",
                    "title": f"竞赛区分度提醒 (T4): {c['name']}",
                    "desc": c["comment"]
                })

    if re.search(r"github\.com/|开源|README", text, re.I):
        suggestions.append({
            "type": "success",
            "title": "规范的开源工程素养",
            "desc": "简历附带 GitHub 开源项目或重视 README 文档编写，体现了出色的工程师自驱力与代码规范性。"
        })

    return {
        "contact": contact,
        "sections": sections,
        "quantification": quant,
        "readability": readability,
        "skills": skills,
        "competitions": competitions,
        "suggestions": suggestions
    }
