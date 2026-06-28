# ResearchAgent

ResearchAgent 是一个面向深度学习科研实验的智能辅助系统，支持论文问答、训练日志分析、MMSegmentation 配置检查、复现实验差距分析和 Markdown 报告生成。项目同时提供 Streamlit 可视化界面和 FastAPI 后端服务，更适合用于 GitHub 展示、简历项目和面试演示。

## 项目亮点

- 基于 PyMuPDF 提取论文 PDF 文本，并使用 TF-IDF 构建本地检索索引。
- 支持带页码来源的论文问答，回答可追溯到原文 evidence snippets。
- 支持解析 `.txt`、`.log`、`.jsonl` 训练日志，自动提取 loss、lr、mIoU、mAcc、aAcc 等指标。
- 自动诊断训练趋势，包括 loss 是否下降、mIoU 是否停滞、学习率是否异常、最佳 mIoU 出现位置等。
- 对 MMSegmentation 配置文件进行静态检查，不执行用户上传的 Python 文件，降低安全风险。
- 提供复现实验审计能力：对比论文信息、用户配置和训练日志，输出配置一致性与复现差距分析。
- 新增 FastAPI 服务层，提供类型化接口、文件上传接口、统一异常处理和 OpenAPI 文档。
- 保留 Streamlit 前端，用于快速展示完整科研实验分析流程。
- 可选接入 OpenAI-compatible 或 Anthropic-compatible API；无 API Key 时仍可本地运行。

## 技术栈

- Python 3.10+
- FastAPI、Uvicorn、Pydantic
- Streamlit
- PyMuPDF
- scikit-learn TF-IDF / cosine similarity
- pandas
- matplotlib
- python-dotenv

## 项目结构

```text
ResearchAgent/
|-- app.py                         # Streamlit 展示端
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
|-- src/
|   |-- api/
|   |   |-- main.py                 # FastAPI 入口
|   |   |-- schemas.py              # Pydantic 响应模型
|   |   `-- services.py             # API 业务编排层
|   |-- paper_qa.py                 # PDF 解析与论文问答
|   |-- paper_profile_extractor.py  # 论文复现要素抽取
|   |-- log_analyzer.py             # 训练日志解析与诊断
|   |-- config_checker.py           # MMSeg 配置检查
|   |-- reproduction_audit.py       # 复现实验差距分析
|   |-- report_generator.py         # Markdown 报告生成
|   |-- llm_client.py               # 可选 LLM 调用
|   |-- tool_registry.py
|   `-- utils.py
|-- examples/
|   |-- sample_log.txt
|   |-- sample_config.py
|   `-- sample_report.md
|-- outputs/
`-- assets/
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Streamlit 展示界面：

```bash
streamlit run app.py
```

启动 FastAPI 后端服务：

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

启动后可以访问：

- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

## FastAPI 接口

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/api/v1/papers/qa` | 上传论文 PDF 并进行基于证据的问答 |
| `POST` | `/api/v1/logs/analyze` | 上传训练日志并返回指标表、诊断结论和最佳 mIoU |
| `POST` | `/api/v1/configs/check` | 上传 MMSeg 配置文件并进行静态检查 |
| `POST` | `/api/v1/reproduction/audit` | 基于论文、配置和日志生成复现实验审计报告 |

示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/logs/analyze" ^
  -F "file=@examples/sample_log.txt"
```

## Streamlit 使用流程

1. Paper QA：上传论文 PDF，输入问题，获得带页码来源的回答。
2. Training Log Analyzer：上传训练日志，解析指标并绘制训练曲线。
3. MMSeg Config Checker：上传 `config.py`，检查数据集、类别数、学习率、优化器、pipeline 和预训练权重等配置。
4. Report Generator：整合论文、日志和配置分析结果，生成 Markdown 实验报告。

## 可选 LLM 配置

项目默认不依赖大模型 API。没有 API Key 时，Paper QA 会返回本地检索出的论文片段和页码来源。

如需启用 LLM 生成回答，可以复制 `.env.example` 为 `.env`，并填写：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
LLM_PROVIDER=OpenAI-compatible
```

也可以使用 Anthropic-compatible 代理：

```env
LLM_PROVIDER=Anthropic-compatible
```

请不要把真实 API Key 提交到 GitHub。

## 简历写法参考

- 设计并实现 ResearchAgent 科研实验辅助系统，覆盖论文检索问答、训练日志诊断、MMSeg 配置静态分析和复现实验审计报告生成等流程。
- 使用 FastAPI 封装核心算法能力，设计文件上传接口、Pydantic 响应模型、统一异常处理和 OpenAPI 文档，使项目具备服务化部署能力。
- 基于 PyMuPDF 与 TF-IDF 实现无外部依赖的论文本地检索，并在问答结果中提供页码级证据来源，提升回答可解释性。
- 使用 pandas 和 matplotlib 解析训练日志、提取关键指标、判断训练趋势，并自动生成结构化诊断结论。
- 通过静态解析方式检查 MMSegmentation 配置文件，在不执行用户代码的前提下识别类别数、数据路径、学习率、优化器、pipeline 等潜在问题。

## 后续优化方向

- 接入 embedding 模型、FAISS 或 Chroma，提升论文语义检索效果。
- 增加任务队列和持久化存储，支持更大文件和异步分析。
- 增加 React 前端或 Ant Design 管理界面，形成更完整的前后端分离项目。
- 增加 Dockerfile、CI 测试和在线部署示例，进一步增强工程展示效果。