# Code Risk Agent

基于 LangGraph 的 LLM 驱动代码变更风险治理 Agent。

## 特性

- **混合确定性+概率性架构**：21 条 AST/正则确定性规则提供可解释证据链，LLM 做高层语义判定
- **五节点 LangGraph StateGraph**：Planner → Tool Router → Judge → Reflection → Reporter，条件边实现反思回环（及 Planner 失败直达报告的 fail-fast 边），Reporter 终点节点保证报告永不缺失
- **诚实降级语义**：Planner LLM 不可用 → 立即失败报告（status=failed、退出码 1）；Judge 分批裁决单批降级 → status=degraded 并标注未裁决证据数；合法空计划正常走完全程
- **证据 hunk 归因去重**：同一证据被多个 hunk 命中时合并为一条携带全部 hunk_keys，报告按 hunk 双呈现（风险行内 Hunks + per-hunk 汇总表）
- **Judge 分批裁决**：按文件分组、每批 50 条证据、全局 id 引用跨批合并，避免大证据池超出超时窗口导致整体降级
- **严格 LLM 响应契约**：llm_assisted 证据响应在 chat_json 重试循环内逐字段校验（evidences 键必存、line_no 非负 int、message/rule_id 非空），畸形响应触发修复重试而非被当作“零发现”
- **RAG 知识增强**：安全知识库 + 历史风险检索 + 代码库上下文检索（Embedding + FTS5 混合检索 + RRF 融合）
- **SQLite 双层记忆**：短期 Checkpointer + 长期反馈表
- **可解释证据链**：每个风险都引用至少一条确定性证据

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 分析 diff
python main.py analyze --diff-file path/to/diff.patch

# 直接传入 diff 文本
python main.py analyze --diff-text "diff --git ..."

# 添加反馈（file-pattern 为文件或目录路径，尾缀 / 表示目录前缀，录入时自动归一化）
python main.py feedback --thread-id "xxx" --file-pattern "auth/" --rule-id SEC001 --type false_positive --content "误报"

# 索引代码库（可选，用于 RAG 代码上下文检索）
python main.py index --repo-path /path/to/repo

# 召回通道评测（确定性层离线跑，golden+regression 失败退出码 1）
python evals/recall/run_eval.py

# 追加 embedding 语义层（真实模型，记录 recall/MRR 不做门禁）
python evals/recall/run_eval.py --live --verbose
```

## 架构

```
START → Planner → Tool Router → Judge → Reflection → Reporter → END
          ↑           ↑          ↑            │
        RAG检索     RAG检索     RAG检索     needs_more?
        (历史风险)  (代码上下文) (安全知识)     │
                         ↑                 └─ yes ─┐
                         └─────────────────────────┘
```

### 五个节点

1. **Planner**：LLM 分析变更概况 + RAG 检索历史风险，输出检测计划（选择应执行的规则集）
2. **Tool Router**：根据计划执行确定性规则收集证据 + RAG 检索代码库上下文（调用方/被调用方）
3. **Judge**：LLM 聚合证据做风险判定 + RAG 检索安全知识库（编码规范、漏洞案例）
4. **Reflection**：LLM 判断分析是否充分，不充分则回到 Tool Router 补充检测（最多 3 轮）
5. **Reporter**：纯状态聚合构建最终风险报告（不调 LLM），所有退出路径必经此节点

### RAG 混合检索

- **Embedding 语义检索**：OpenAI 兼容 Embedding API + SQLite 向量存储 + 余弦相似度
- **FTS5 关键词检索**：SQLite 内置全文索引 + BM25 排序
- **RRF 融合**：Reciprocal Rank Fusion 算法合并两路结果

## 规则清单

| 类别 | 数量 | 示例 |
|------|------|------|
| 安全 | 4 | SQL 注入、硬编码密钥、命令注入、不安全反序列化 |
| 复杂度 | 3 | 函数过长、高圈复杂度、深嵌套 |
| Bug 风险 | 4 | 裸 except、可变默认参数、未使用导入、空指针 |
| 风格 | 3 | 命名违规、魔法数字、超长行 |
| 性能 | 3 | 循环内 IO、N+1 查询、循环内字符串拼接 |
| 可维护性 | 3 | 缺少 docstring、重复代码、TODO 标记 |

## 项目结构

```
code-risk-agent/
├── src/
│   ├── models/          # 数据模型 (enums, state)
│   ├── parsers/         # Git Diff 解析器
│   ├── rules/           # 21 条确定性规则 + ToolRegistry
│   ├── llm/             # LLM + Embedding 客户端
│   ├── rag/             # RAG 检索器 + 索引器
│   ├── memory/          # 短期 Checkpointer + 长期反馈表
│   ├── nodes/           # 四个 LangGraph 节点
│   └── graph.py         # StateGraph 构建 + 编译
├── config/              # Pydantic Settings 配置
├── data/                # 安全知识数据
├── tests/               # 测试套件
├── main.py              # CLI 入口
└── requirements.txt
```

## 技术栈

Python 3.11+, LangGraph, OpenAI-compatible LLM, SQLite (FTS5), Pydantic v2, Rich, Click, NumPy

## 设计决策

1. **SQLite 而非 Redis/向量数据库**：单文件零运维，embedding 以 JSON 存储在 SQLite 列中，FTS5 内置
2. **Embedding + FTS5 混合检索**：语义检索找概念相近的知识，关键词检索精确匹配规则 ID/文件路径，RRF 融合
3. **RAG 嵌入节点而非独立节点**：三个检索方向触发时机不同，嵌入对应节点按需检索
4. **AST + 正则混合**：Python 用 ast 模块零依赖，其他语言用正则
5. **五节点而非单节点 LLM**：确定性规则提供可解释证据链，LLM 负责高层聚合判定；Reporter 独立收尾，报告缺失在图结构上不可能
6. **Reflection 条件边**：Agent 自主决策是否回环，LangGraph 核心能力
7. **只分析变更行**：性能优先，变更行是风险高发区
