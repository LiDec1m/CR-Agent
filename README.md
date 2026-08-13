# Code Risk Agent

基于 LangGraph 的 LLM 驱动代码变更风险治理 Agent。

## 特性

- **混合确定性+概率性架构**：20 条 AST/正则确定性规则提供可解释证据链，LLM 做高层语义判定
- **四节点 LangGraph StateGraph**：Planner → Tool Router → Judge → Reflection，条件边实现反思回环
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

# 添加反馈
python main.py feedback --thread-id "xxx" --file-pattern "auth/*" --rule-id SEC001 --type false_positive --content "误报"

# 索引代码库（可选，用于 RAG 代码上下文检索）
python main.py index --repo-path /path/to/repo
```

## 架构

```
START → Planner → Tool Router → Judge → Reflection → END
          ↑           ↑          ↑            │
        RAG检索     RAG检索     RAG检索     needs_more?
        (历史风险)  (代码上下文) (安全知识)     │
                         ↑                 └─ yes ─┐
                         └─────────────────────────┘
```

### 四个节点

1. **Planner**：LLM 分析变更概况 + RAG 检索历史风险，输出检测计划（选择应执行的规则集）
2. **Tool Router**：根据计划执行确定性规则收集证据 + RAG 检索代码库上下文（调用方/被调用方）
3. **Judge**：LLM 聚合证据做风险判定 + RAG 检索安全知识库（编码规范、漏洞案例）
4. **Reflection**：LLM 判断分析是否充分，不充分则回到 Tool Router 补充检测（最多 3 轮）

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
│   ├── rules/           # 20 条确定性规则 + ToolRegistry
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
5. **四节点而非单节点 LLM**：确定性规则提供可解释证据链，LLM 负责高层聚合判定
6. **Reflection 条件边**：Agent 自主决策是否回环，LangGraph 核心能力
7. **只分析变更行**：性能优先，变更行是风险高发区
