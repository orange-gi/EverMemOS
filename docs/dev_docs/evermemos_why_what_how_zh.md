## EverMemOS：Why-What-How 项目介绍

> **不止记忆，更是远见。**  
> EverMemOS 是一个面向对话式 AI 智能体的企业级智能记忆系统：让智能体不仅“记住发生了什么”，还能够“理解这些记忆的意义”，并将其用于当下的决策与行动。

---

## Why：为什么要做 EverMemOS？

- **对话式 AI 的“失忆”是产品体验的硬伤**  
  LLM 天然是短上下文、无状态的：用户的偏好、目标、历史事实无法稳定继承，导致“越聊越像新朋友”，难以形成信任与粘性。

- **传统“记忆 = 数据库检索”不够**  
  只把过去对话切碎存起来、按关键词/向量检索出来，往往会出现：
  - **碎片化**：记忆片段无法形成“故事线”，难以支持复杂推理与多线程对话理解。
  - **相关但不关键**：检索召回的内容缺少“重要性/证据”意识，关键约束容易漏掉。
  - **不可控的幻觉风险**：没有可追溯的“证据链”，模型容易凭空编造“似乎合理”的背景。

- **企业场景需要可扩展、可运维的系统化能力**  
  记忆不是单点能力：它涉及数据摄入、结构化、索引、检索、重排序、在线服务、评估与回归测试。EverMemOS 以工程化方式把这条链路打通。

---

## What：EverMemOS 是什么？

### 1) 系统目标（你能得到什么）

- **长期记忆**：把对话沉淀为结构化、可检索的长期记忆，让后续对话建立在“前序理解”之上。
- **上下文自觉（Contextual Awareness）**：不仅能检索，还能在关键时刻“想得周到”，把与当前任务真正相关且关键的记忆带回推理。
- **可演示、可集成、可评估**：提供 Demo、HTTP API、批处理脚本与统一评估框架，覆盖从体验到落地的全链路。

### 2) 总体框架：两条主线的“认知闭环”

EverMemOS 的系统框架可概括为两条主线：

- **记忆构筑（Memory Construction）**：把原始对话流转为结构化的长期记忆，并建立索引。
- **记忆感知（Memory Perception）**：围绕当前查询进行多策略召回、融合与重排序，把“可用的记忆证据”注入上下文推理。

最终形成闭环：**结构化记忆 → 多策略召回 → 智能检索/融合 → 基于证据的生成与决策**。

### 3) 核心概念：从 MemCell 到多层记忆

EverMemOS 用一套可工程化的“记忆层级”来组织信息（详见 `docs/dev_docs/memory_types_guide_zh.md`）：

- **MemCell（记忆单元）**：对话流经边界检测后得到的最小“主题/事件”容器，是所有下游记忆的源头。
- **Episode（情节记忆）**：对 MemCell 的叙事性总结，提供更高层语义理解（支持群体视角与个人视角）。
- **EventLog（事件日志）**：从 Episode 中抽取的原子化事实（atomic facts），每条事实独立存储与向量化，适合精确问答。
- **Foresight（前瞻记忆）**：基于 Episode 的未来影响/提醒，具备时效窗口，可用于主动提醒与规划。
- **Profile / Preference / Base Memory 等**：用于刻画用户画像、偏好与稳定信息（通过 API 可直接读取或参与检索）。

> 直觉理解：**Episode 负责“讲清楚发生了什么”**，**EventLog 负责“把事实拆到可精确命中”**，**Foresight 负责“面向未来的影响与提醒”**。

### 4) 检索能力：轻量级与 Agentic 两种模式

EverMemOS 支持两类检索路径：

- **轻量级检索（Lightweight）**：`BM25` / `Embedding` / `RRF`（推荐）  
  适合低延迟场景；其中 **RRF（Reciprocal Rank Fusion）** 将关键词与语义检索并行融合，提高召回稳定性。

- **Agentic 检索（Agentic Multi-Round Recall）**：  
  当“单次查询召回不足”时，系统可生成 2-3 个互补查询并行检索、融合与重排序，提高复杂意图覆盖度（代价是更多 LLM 调用与延迟）。

### 5) 运行形态：在线服务 + 多后端存储

EverMemOS 提供基于 FastAPI 的在线服务，并默认依赖以下基础设施（可用 Docker Compose 一键启动）：

- **MongoDB**：主存储（MemCell、Episode、Profile 等）
- **Elasticsearch**：关键词检索（BM25）
- **Milvus**：向量检索（语义召回）
- **Redis**：缓存与分布式能力（如锁、队列等）

### 6) 对外接口：用最简单的方式接入

你可以用“单条消息”的最小格式对接系统（无需预转换），核心接口包括（详见 `docs/api_docs/memory_api_zh.md`）：

- **`POST /api/v1/memories`**：逐条摄入消息，触发累计/边界检测/记忆提取与写入
- **`GET /api/v1/memories/search`**：检索记忆（支持 `rrf` / `embedding` / `bm25` 等模式与范围）
- **`GET /api/v1/memories`**：按用户读取核心记忆（如画像/偏好等）
- **`POST/PATCH /api/v1/memories/conversation-meta`**：对话元数据管理（参与者、标签、场景等）

### 7) 工程化配套：Demo 与评估框架

- **Demo**：`demo/simple_demo.py`（最小可用链路）、`demo/extract_memory.py`（批量导入示例数据）、`demo/chat_with_memory.py`（交互式记忆对话）。  
  详见 `demo/README_zh.md`。

- **Evaluation**：统一评测流水线（add → search → answer → evaluate），支持 LoCoMo、LongMemEval、PersonaMem 等基准。  
  详见 `evaluation/README.md` 与 `evaluation/README_zh.md`。

---

## How：如何使用/集成/扩展 EverMemOS？

### 1) 最快上手：跑一个端到端 Demo

- **启动依赖服务**：`docker-compose up -d`
- **安装依赖**：`uv sync`
- **配置环境**：`cp env.template .env`（按 `docs/usage/CONFIGURATION_GUIDE_zh.md` 填写）
- **启动 API 服务**：`uv run python src/run.py --port 8001`
- **运行快速演示**：`uv run python src/bootstrap.py demo/simple_demo.py`

> 注意：`MEMORY_LANGUAGE` 必须与导入数据语言一致（`zh`/`en`），且需要在启动 API 服务前设置；切换语言需重启服务（详见 `demo/README_zh.md`）。

### 2) 接入你的产品：把“消息流”接到 Memory API

推荐把 EverMemOS 放在你的聊天系统“消息写入链路”上：

- 每条消息到达时，调用 **`POST /api/v1/memories`** 写入
- 在生成回复前，调用 **`GET /api/v1/memories/search`** 检索相关记忆
- 将“检索到的记忆证据”拼接进提示词上下文，让模型基于证据回答（降低幻觉）

适配方式非常轻：对外只需要“单条消息”格式与可选的对话元信息。

### 3) 批量导入历史数据：用标准格式 + 脚本

如果你有历史群聊数据，建议遵循 `data_format/group_chat/group_chat_format.md`，然后使用：

- `src/run_memorize.py`：批量逐条调用 API 导入（支持仅校验 `--validate-only`）  
  详见 `docs/dev_docs/run_memorize_usage.md`。

### 4) 跑基准评测：用 evaluation CLI 做回归与对比

当你要做检索策略/提示词/模型替换等改动时，建议用评估框架做回归：

- 冒烟测试：`uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke`
- 完整评测：`uv run python -m evaluation.cli --dataset locomo --system evermemos`

### 5) 想二次开发：从“分层架构”入手

代码采用分层结构（详见根目录 `README_zh.md` 的 Project Structure）：

- `memory_layer/`：记忆提取（MemCell → Episode/Foresight/EventLog/画像等）
- `retrieval_layer/`：多模态检索与融合/排序
- `agentic_layer/`：统一检索入口与 agentic 编排（如 `MemoryManager`）
- `core/`：依赖注入、生命周期、中间件等工程能力

开发实践与 Mock/DI 规范可参考：`docs/dev_docs/development_guide_zh.md`。

---

## 进一步阅读（建议顺序）

- **快速跑起来**：`docs/dev_docs/getting_started_zh.md`
- **配置说明**：`docs/usage/CONFIGURATION_GUIDE_zh.md`
- **API 参考**：`docs/api_docs/memory_api_zh.md`
- **记忆类型与存储模型**：`docs/dev_docs/memory_types_guide_zh.md`
- **Demo 使用**：`demo/README_zh.md`
- **评估框架**：`evaluation/README_zh.md`

