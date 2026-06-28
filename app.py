import json
import os
import pickle
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config_checker import check_config
from src.llm_client import LLMConfig, generate_grounded_answer
from src.log_analyzer import analyze_log, plot_metrics
from src.paper_qa import answer_question, build_paper_index, extract_pdf_text
from src.paper_profile_extractor import extract_paper_profile
from src.report_generator import build_report
from src.reproduction_audit import analyze_reproduction_gap, compare_config_to_profile
from src.tool_registry import TOOL_REGISTRY
from src.utils import decode_uploaded_text, ensure_outputs_dir


APP_TITLE = "ResearchAgent"
APP_SUBTITLE = "RAG and Tool Calling assistant for deep learning research experiments"
PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
CHAT_HISTORY_PATH = PROJECT_DIR / "outputs" / "paper_chat_history.json"
PAPER_STATE_PATH = PROJECT_DIR / "outputs" / "paper_state.json"
PAPER_INDEX_PATH = PROJECT_DIR / "outputs" / "paper_index.pkl"
LAST_PDF_PATH = PROJECT_DIR / "outputs" / "last_paper.pdf"
LOG_STATE_PATH = PROJECT_DIR / "outputs" / "log_state.json"
LOG_RESULT_PATH = PROJECT_DIR / "outputs" / "log_result.pkl"
LAST_LOG_PATH = PROJECT_DIR / "outputs" / "last_training_log.txt"
CONFIG_STATE_PATH = PROJECT_DIR / "outputs" / "config_state.json"
CONFIG_RESULT_PATH = PROJECT_DIR / "outputs" / "config_result.pkl"
LAST_CONFIG_PATH = PROJECT_DIR / "outputs" / "last_mmseg_config.py"
MAX_SAVED_CHAT_MESSAGES = 30

load_dotenv()


I18N = {
    "English": {
        "app_subtitle": APP_SUBTITLE,
        "language_label": "Language",
        "tools": "Research Tools",
        "tool_caption": "Modular analysis tools for paper reading, experiment diagnosis, and reporting.",
        "paper_title": "Paper QA",
        "paper_caption": "Upload a paper, ask a question, and get retrieval-grounded answers with page citations.",
        "upload_pdf": "Upload paper PDF",
        "build_index": "Build Paper Index",
        "indexed": "Indexed {pages} pages from {name}.",
        "pdf_error": "Failed to process PDF: {error}",
        "question_label": "Your question",
        "question_placeholder": "What is the main contribution of this paper?",
        "top_k": "Top-k retrieved chunks",
        "ask": "Ask",
        "answer": "Answer",
        "sources": "Sources",
        "log_title": "Training Log Analyzer",
        "log_caption": "Parse training logs, plot core metrics, and diagnose convergence signals.",
        "upload_log": "Upload training log",
        "analyze_log": "Analyze Log",
        "log_success": "Training log analyzed.",
        "log_error": "Failed to analyze log: {error}",
        "parsed_metrics": "Parsed Metrics",
        "curves": "Curves",
        "no_plots": "No plottable metric columns were found.",
        "diagnosis": "Diagnosis",
        "best_miou": "Best mIoU",
        "config_title": "MMSeg Config Checker",
        "config_caption": "Inspect MMSegmentation config files safely without executing uploaded Python code.",
        "upload_config": "Upload MMSegmentation config.py",
        "check_config": "Check Config",
        "config_success": "Config checked.",
        "config_error": "Failed to check config: {error}",
        "extracted_fields": "Extracted Fields",
        "findings": "Findings",
        "report_title": "Report Generator",
        "report_caption": "Combine retrieved evidence, log diagnosis, and config findings into a Markdown report.",
        "report_input": "Report title",
        "notes": "Additional notes",
        "notes_placeholder": "Optional experiment context or next-step ideas.",
        "generate_report": "Generate Report",
        "markdown_preview": "Markdown Preview",
        "download_report": "Download Markdown Report",
        "tab_paper": "Paper QA",
        "tab_log": "Training Log Analyzer",
        "tab_config": "MMSeg Config Checker",
        "tab_report": "Report Generator",
        "hero_kicker": "Research workflow agent",
        "hero_body": "A lightweight local MVP for paper retrieval, experiment log diagnosis, config auditing, and report generation.",
        "api_settings": "LLM Settings",
        "api_caption": "Optional. TF-IDF retrieval works without an API; LLM generation uses retrieved sources as context.",
        "enable_llm": "Enable LLM answer generation",
        "provider": "Provider",
        "api_key": "API Key",
        "base_url": "Base URL",
        "model": "Model",
        "api_ready": "LLM is configured.",
        "api_missing": "LLM is enabled, but API key, base URL, or model is missing.",
        "use_env_hint": "Values can be loaded from `.env` or entered here temporarily.",
        "generating": "Generating grounded answer with LLM...",
        "llm_failed": "LLM generation failed, so ResearchAgent returned the retrieval-only answer. Error: {error}",
        "llm_used": "Generated with LLM from retrieved sources.",
    },
    "简体中文": {
        "app_subtitle": "面向深度学习科研实验的 RAG 与 Tool Calling 辅助 Agent",
        "language_label": "语言",
        "tools": "科研工具",
        "tool_caption": "用于论文阅读、实验诊断、配置审计与报告生成的模块化工具。",
        "paper_title": "论文问答",
        "paper_caption": "上传论文 PDF，输入问题，并返回带页码来源的检索结果。",
        "upload_pdf": "上传论文 PDF",
        "build_index": "构建论文索引",
        "indexed": "已从 {name} 索引 {pages} 页文本。",
        "pdf_error": "PDF 处理失败：{error}",
        "question_label": "你的问题",
        "question_placeholder": "这篇论文的主要贡献是什么？",
        "top_k": "检索片段数量",
        "ask": "提问",
        "answer": "回答",
        "sources": "来源片段",
        "log_title": "训练日志分析",
        "log_caption": "解析训练日志，绘制关键指标曲线，并诊断收敛信号。",
        "upload_log": "上传训练日志",
        "analyze_log": "分析日志",
        "log_success": "训练日志分析完成。",
        "log_error": "日志分析失败：{error}",
        "parsed_metrics": "解析指标",
        "curves": "训练曲线",
        "no_plots": "未发现可绘制的指标列。",
        "diagnosis": "诊断结论",
        "best_miou": "最佳 mIoU",
        "config_title": "MMSeg 配置检查",
        "config_caption": "在不执行上传 Python 文件的前提下，安全检查 MMSegmentation 配置。",
        "upload_config": "上传 MMSegmentation config.py",
        "check_config": "检查配置",
        "config_success": "配置检查完成。",
        "config_error": "配置检查失败：{error}",
        "extracted_fields": "提取字段",
        "findings": "检查结果",
        "report_title": "报告生成",
        "report_caption": "整合论文检索、日志诊断和配置检查结果，生成 Markdown 实验报告。",
        "report_input": "报告标题",
        "notes": "补充说明",
        "notes_placeholder": "可填写实验背景或下一步计划。",
        "generate_report": "生成报告",
        "markdown_preview": "Markdown 预览",
        "download_report": "下载 Markdown 报告",
        "tab_paper": "论文问答",
        "tab_log": "训练日志分析",
        "tab_config": "MMSeg 配置检查",
        "tab_report": "报告生成",
        "hero_kicker": "科研实验工作流 Agent",
        "hero_body": "一个轻量级本地 MVP，用于论文检索问答、训练日志诊断、配置审计和实验报告生成。",
        "api_settings": "大模型设置",
        "api_caption": "可选。TF-IDF 检索不需要 API；启用大模型后，会把检索片段作为上下文生成正式回答。",
        "enable_llm": "启用大模型回答生成",
        "provider": "接口类型",
        "api_key": "API Key",
        "base_url": "Base URL",
        "model": "模型名称",
        "api_ready": "大模型配置已就绪。",
        "api_missing": "已启用大模型，但 API Key、Base URL 或模型名称缺失。",
        "use_env_hint": "可以从 `.env` 读取，也可以在这里临时输入。",
        "generating": "正在使用大模型生成带来源的回答...",
        "llm_failed": "大模型生成失败，已返回仅检索版回答。错误：{error}",
        "llm_used": "已基于检索片段调用大模型生成回答。",
    },
}


def init_state() -> None:
    defaults = {
        "paper_result": load_paper_workspace(),
        "paper_answer": load_last_paper_answer(),
        "paper_profile": load_last_paper_profile(),
        "paper_chat_history": load_paper_chat_history(),
        "log_result": load_last_pickle_result(LOG_RESULT_PATH),
        "config_result": load_last_pickle_result(CONFIG_RESULT_PATH),
        "report_markdown": "",
        "language": "English",
        "llm_enabled": os.getenv("LLM_ENABLED", "false").lower() == "true",
        "llm_provider": os.getenv("LLM_PROVIDER", "OpenAI-compatible"),
        "llm_api_key": os.getenv("OPENAI_API_KEY", ""),
        "llm_base_url": os.getenv("OPENAI_BASE_URL", ""),
        "llm_model": os.getenv("OPENAI_MODEL", ""),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def t(key: str) -> str:
    return I18N[st.session_state.language][key]


def current_language_suffix() -> str:
    return "zh" if st.session_state.language == "简体中文" else "en"


def ui_text(english: str, chinese: str) -> str:
    return chinese if current_language_suffix() == "zh" else english


def get_llm_config() -> LLMConfig:
    return LLMConfig(
        enabled=bool(st.session_state.llm_enabled),
        provider=st.session_state.llm_provider,
        api_key=st.session_state.llm_api_key,
        base_url=st.session_state.llm_base_url,
        model=st.session_state.llm_model,
    )


def save_llm_config_to_env(config: LLMConfig) -> None:
    existing_values = read_env_values(ENV_PATH)
    existing_values.update(
        {
            "LLM_ENABLED": "true" if config.enabled else "false",
            "LLM_PROVIDER": config.provider,
            "OPENAI_API_KEY": config.api_key.strip(),
            "OPENAI_BASE_URL": config.base_url.strip(),
            "OPENAI_MODEL": config.model.strip(),
        }
    )
    lines = [
        "# Local ResearchAgent settings.",
        "# Keep this file private. It is ignored by Git.",
    ]
    for key, value in existing_values.items():
        lines.append(f"{key}={format_env_value(value)}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_paper_workspace() -> dict | None:
    if not PAPER_STATE_PATH.exists() or not PAPER_INDEX_PATH.exists():
        return None
    try:
        state = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))
        with PAPER_INDEX_PATH.open("rb") as file:
            index = pickle.load(file)
    except (OSError, json.JSONDecodeError, pickle.PickleError, EOFError):
        return None

    return {
        "file_name": state.get("file_name", "last_paper.pdf"),
        "pages": [],
        "index": index,
        "pdf_path": state.get("pdf_path", str(LAST_PDF_PATH)),
        "restored": True,
    }


def load_last_paper_answer() -> dict | None:
    if not PAPER_STATE_PATH.exists():
        return None
    try:
        state = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    answer = state.get("paper_answer")
    return answer if isinstance(answer, dict) else None


def load_last_paper_profile() -> dict | None:
    if not PAPER_STATE_PATH.exists():
        return None
    try:
        state = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    profile = state.get("paper_profile")
    return profile if isinstance(profile, dict) else None


def save_last_paper_workspace(file_name: str, pdf_bytes: bytes, pages: list[dict], index: object) -> None:
    output_dir = ensure_outputs_dir()
    pdf_path = output_dir / "last_paper.pdf"
    pdf_path.write_bytes(pdf_bytes)
    with PAPER_INDEX_PATH.open("wb") as file:
        pickle.dump(index, file)
    write_paper_state(
        {
            "file_name": file_name,
            "pdf_path": str(pdf_path),
            "page_count": len(pages),
            "paper_answer": None,
            "paper_profile": None,
        }
    )


def save_last_paper_answer(answer: dict) -> None:
    state = read_paper_state()
    state["paper_answer"] = answer
    write_paper_state(state)


def save_last_paper_profile(profile: dict) -> None:
    state = read_paper_state()
    state["paper_profile"] = profile
    write_paper_state(state)


def load_last_pickle_result(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except (OSError, pickle.PickleError, EOFError, AttributeError):
        return None


def save_last_pickle_result(path: Path, result: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(result, file)


def save_last_log_workspace(file_name: str, text: str, result: object) -> None:
    ensure_outputs_dir()
    LAST_LOG_PATH.write_text(text, encoding="utf-8")
    save_last_pickle_result(LOG_RESULT_PATH, result)
    LOG_STATE_PATH.write_text(
        json.dumps(
            {
                "file_name": file_name,
                "log_path": str(LAST_LOG_PATH),
                "result_path": str(LOG_RESULT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_last_config_workspace(file_names: list[str], combined_text: str, result: object) -> None:
    ensure_outputs_dir()
    LAST_CONFIG_PATH.write_text(combined_text, encoding="utf-8")
    save_last_pickle_result(CONFIG_RESULT_PATH, result)
    CONFIG_STATE_PATH.write_text(
        json.dumps(
            {
                "file_names": file_names,
                "config_path": str(LAST_CONFIG_PATH),
                "result_path": str(CONFIG_RESULT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_paper_state() -> dict:
    if not PAPER_STATE_PATH.exists():
        return {}
    try:
        state = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def write_paper_state(state: dict) -> None:
    PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_paper_chat_history() -> list[dict]:
    if not CHAT_HISTORY_PATH.exists():
        return []
    try:
        history = json.loads(CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(history, list):
        return []
    return [message for message in history if isinstance(message, dict)][-MAX_SAVED_CHAT_MESSAGES:]


def save_paper_chat_history() -> None:
    CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = st.session_state.get("paper_chat_history", [])[-MAX_SAVED_CHAT_MESSAGES:]
    CHAT_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_paper_chat_history() -> None:
    st.session_state.paper_chat_history = []
    st.session_state.paper_answer = None
    state = read_paper_state()
    if state:
        state["paper_answer"] = None
        write_paper_state(state)
    try:
        CHAT_HISTORY_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def read_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def format_env_value(value: str) -> str:
    if not value:
        return ""
    if any(char.isspace() for char in value) or "#" in value:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def inject_academic_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ra-ink: #111827;
            --ra-muted: #6b7280;
            --ra-line: #e5e7eb;
            --ra-soft: #f8fafc;
            --ra-panel: rgba(255, 255, 255, 0.92);
            --ra-accent: #374151;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 16%, rgba(226, 232, 240, 0.62), transparent 28%),
                linear-gradient(180deg, #ffffff 0%, #f8fafc 56%, #f3f4f6 100%);
            color: var(--ra-ink);
        }

        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.82);
            border-right: 1px solid var(--ra-line);
            backdrop-filter: blur(16px);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--ra-ink);
            letter-spacing: 0;
        }

        .ra-hero {
            padding: 2.6rem 0 1.65rem;
            border-bottom: 1px solid rgba(229, 231, 235, 0.65);
            margin-bottom: 1.2rem;
        }

        .ra-kicker {
            color: var(--ra-muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }

        .ra-title {
            font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
            font-size: 4.2rem;
            line-height: 1;
            font-weight: 500;
            margin: 0;
            letter-spacing: 0;
        }

        .ra-body {
            max-width: 720px;
            color: var(--ra-muted);
            font-size: 1.02rem;
            margin-top: 0.75rem;
        }

        .ra-section-title {
            font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
            font-size: 3.5rem;
            line-height: 1.04;
            font-weight: 500;
            margin: 1rem 0 0.35rem;
        }

        .ra-section-caption {
            color: var(--ra-muted);
            margin-bottom: 1.2rem;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid var(--ra-line);
            border-radius: 999px;
            padding: 0.25rem;
            gap: 0.2rem;
            width: fit-content;
            margin: 0 auto 1.2rem;
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stTabs"] button[role="tab"] {
            border-radius: 999px;
            min-height: 2.25rem;
            padding: 0 1rem;
            color: #4b5563;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: #eef0f3;
            color: #111827;
            font-weight: 650;
        }

        div[data-testid="stFileUploader"],
        div[data-testid="stDataFrame"],
        div[data-testid="stJson"],
        div[data-testid="stMetric"],
        div[data-testid="stCodeBlock"] {
            border: 1px solid var(--ra-line);
            border-radius: 12px;
            background: var(--ra-panel);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.055);
            padding: 0.3rem;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            border-radius: 10px;
            border-color: #d1d5db;
            background: rgba(255, 255, 255, 0.92);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #d1d5db;
            box-shadow: none;
            font-weight: 650;
        }

        .stButton > button[kind="primary"] {
            background: #1f2937;
            border-color: #1f2937;
            color: white;
        }

        .ra-card {
            border: 1px solid var(--ra-line);
            border-radius: 12px;
            background: var(--ra-panel);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.055);
            padding: 1.15rem 1.25rem;
            margin: 0.75rem 0;
        }

        .ra-source-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }

        .ra-source-card {
            border: 1px solid var(--ra-line);
            border-radius: 10px;
            background: #ffffff;
            padding: 0.85rem;
            min-height: 150px;
        }

        .ra-source-page {
            display: inline-block;
            color: #4b5563;
            background: #f3f4f6;
            border-radius: 999px;
            font-size: 0.75rem;
            padding: 0.18rem 0.55rem;
            margin-top: 0.55rem;
        }

        @media (max-width: 760px) {
            .ra-title { font-size: 2.7rem; }
            .ra-section-title { font-size: 2.45rem; }
            div[data-testid="stTabs"] [role="tablist"] { width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    st.sidebar.title(APP_TITLE)
    st.sidebar.caption(t("app_subtitle"))
    selected_language = st.sidebar.segmented_control(
        t("language_label"),
        options=["English", "简体中文"],
        default=st.session_state.language,
    )
    if selected_language and selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()
    render_api_settings()
    st.sidebar.divider()
    st.sidebar.subheader(t("tools"))
    st.sidebar.caption(t("tool_caption"))
    suffix = current_language_suffix()
    for meta in TOOL_REGISTRY.values():
        st.sidebar.markdown(f"**{meta[f'label_{suffix}']}**")
        st.sidebar.caption(meta[f"description_{suffix}"])


def render_api_settings() -> None:
    with st.sidebar.expander(t("api_settings"), expanded=False):
        st.caption(t("api_caption"))
        st.caption(t("use_env_hint"))
        st.session_state.llm_enabled = st.checkbox(
            t("enable_llm"),
            value=st.session_state.llm_enabled,
        )
        st.session_state.llm_provider = st.selectbox(
            t("provider"),
            ["OpenAI-compatible", "Anthropic-compatible"],
            index=0 if st.session_state.llm_provider != "Anthropic-compatible" else 1,
        )
        st.session_state.llm_api_key = st.text_input(
            t("api_key"),
            value=st.session_state.llm_api_key,
            type="password",
        )
        st.session_state.llm_base_url = st.text_input(
            t("base_url"),
            value=st.session_state.llm_base_url,
            placeholder="https://api.openai.com/v1",
        )
        st.session_state.llm_model = st.text_input(
            t("model"),
            value=st.session_state.llm_model,
            placeholder="gpt-4o-mini",
        )

        config = get_llm_config()
        if st.button(ui_text("Save LLM settings locally", "保存大模型配置到本地"), use_container_width=True):
            try:
                save_llm_config_to_env(config)
                st.success(ui_text("Saved to local .env. It will load automatically after refresh.", "已保存到本地 .env，刷新后会自动加载。"))
            except Exception as exc:
                st.error(ui_text(f"Failed to save settings: {exc}", f"保存配置失败：{exc}"))

        if config.enabled and config.is_ready:
            st.success(t("api_ready"))
        elif config.enabled:
            st.warning(t("api_missing"))


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="ra-hero">
            <div class="ra-kicker">{t("hero_kicker")}</div>
            <h1 class="ra-title">{APP_TITLE}</h1>
            <div class="ra-body">{t("hero_body")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reproduction_notice() -> None:
    st.markdown(
        f"""
        <div class="ra-card" style="border-left: 4px solid #64748b;">
            {ui_text(
                "If the paper does not provide source code or complete experimental settings, ResearchAgent can only perform evidence-based analysis from the paper text, user config, and training logs. It cannot guarantee exact reproduction of the authors' implementation.",
                "如果论文未提供源码或完整实验设置，系统只能基于论文文本、用户 config 和训练日志进行证据化分析，不能保证完全复现作者实现。"
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title_key: str, caption_key: str) -> None:
    st.markdown(
        f"""
        <div>
            <h2 class="ra-section-title">{t(title_key)}</h2>
            <div class="ra-section-caption">{t(caption_key)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_paper_tab() -> None:
    render_section_header("paper_title", "paper_caption")

    uploaded_pdf = st.file_uploader(t("upload_pdf"), type=["pdf"], key="paper_pdf")
    if uploaded_pdf is not None:
        if st.button(t("build_index"), type="primary"):
            try:
                pdf_bytes = uploaded_pdf.getvalue()
                ensure_outputs_dir()
                LAST_PDF_PATH.write_bytes(pdf_bytes)
                pages = extract_pdf_text(LAST_PDF_PATH)
                index = build_paper_index(pages)
                save_last_paper_workspace(uploaded_pdf.name, pdf_bytes, pages, index)
                st.session_state.paper_result = {
                    "file_name": uploaded_pdf.name,
                    "pages": pages,
                    "index": index,
                    "pdf_path": str(LAST_PDF_PATH),
                    "restored": False,
                }
                st.session_state.paper_answer = None
                st.session_state.paper_profile = None
                clear_paper_chat_history()
                st.success(t("indexed").format(pages=len(pages), name=uploaded_pdf.name))
            except Exception as exc:
                st.error(t("pdf_error").format(error=exc))

    paper_result = st.session_state.paper_result
    if paper_result and paper_result.get("restored"):
        st.info(ui_text(
            f"Restored the last submitted paper: {paper_result.get('file_name', 'last_paper.pdf')}.",
            f"已恢复上次提交的论文：{paper_result.get('file_name', 'last_paper.pdf')}。",
        ))

    top_k = st.slider(t("top_k"), min_value=1, max_value=5, value=3, disabled=paper_result is None)

    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button(ui_text("Clear chat", "清空对话"), disabled=not st.session_state.paper_chat_history):
            clear_paper_chat_history()
            st.rerun()
    with col_b:
        if paper_result:
            st.caption(ui_text("Conversation memory is kept in this Streamlit session.", "对话记忆会临时保存在当前 Streamlit 会话中。"))
        else:
            st.info(ui_text("Build a paper index first, then start a multi-turn paper chat.", "请先构建论文索引，然后开始多轮论文对话。"))

    profile_col, _ = st.columns([1, 3])
    with profile_col:
        if st.button(ui_text("Extract paper profile", "提取论文画像"), disabled=paper_result is None):
            profile = extract_paper_profile(paper_result["index"].chunks)
            st.session_state.paper_profile = profile
            save_last_paper_profile(profile)
            st.success(ui_text("Paper profile extracted.", "论文画像提取完成。"))

    render_paper_profile(st.session_state.paper_profile)
    render_paper_chat_history()

    question = st.chat_input(
        ui_text("Ask a follow-up about this paper...", "继续追问这篇论文..."),
        disabled=paper_result is None,
    )
    if question and paper_result:
        handle_paper_chat_turn(question.strip(), top_k)


def render_paper_chat_history() -> None:
    for message in st.session_state.paper_chat_history:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(str(message.get("content", "")))
            if role == "assistant":
                if message.get("llm_used"):
                    st.caption(t("llm_used"))
                if message.get("llm_error"):
                    st.warning(t("llm_failed").format(error=message["llm_error"]))
                render_source_cards(message.get("sources", []))


def render_paper_profile(profile: dict | None) -> None:
    if not profile:
        return
    with st.expander(ui_text("Paper Reference Profile", "论文参考画像"), expanded=True):
        st.caption(ui_text(
            "This profile is extracted from PDF evidence. Missing fields are marked as not specified.",
            "该画像基于 PDF 证据抽取；缺失字段统一标记为 not specified。",
        ))
        col1, col2, col3 = st.columns(3)
        col1.metric("Task", _profile_display_value(profile.get("task")))
        col2.metric("Dataset", _profile_display_value(profile.get("datasets")))
        col3.metric("Metrics", _profile_display_value(profile.get("metrics")))

        st.markdown("#### Training Settings")
        st.json(profile.get("training_settings", {}))

        st.markdown("#### Reported Results")
        st.json(profile.get("reported_results", []))

        st.markdown("#### Missing Reproducibility Details")
        missing = profile.get("missing_details", [])
        if missing:
            st.write(", ".join(missing))
        else:
            st.write(ui_text("No major missing detail was detected.", "未检测到主要缺失细节。"))

        with st.expander(ui_text("Raw structured profile", "原始结构化画像"), expanded=False):
            st.json(profile)


def _profile_display_value(field: dict | None) -> str:
    if not field:
        return "not specified"
    value = field.get("value", "not specified")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:3]) or "not specified"
    return str(value)


def handle_paper_chat_turn(question: str, top_k: int) -> None:
    st.session_state.paper_chat_history.append({"role": "user", "content": question})

    retrieval_query = build_contextual_retrieval_query(question)
    paper_answer = answer_question(
        st.session_state.paper_result["index"],
        retrieval_query,
        top_k=top_k,
    )

    llm_config = get_llm_config()
    if llm_config.enabled and paper_answer.get("sources"):
        try:
            with st.spinner(t("generating")):
                paper_answer["answer"] = generate_grounded_answer(
                    question=question,
                    sources=paper_answer["sources"],
                    config=llm_config,
                    language=st.session_state.language,
                    chat_history=st.session_state.paper_chat_history[:-1],
                )
            paper_answer["llm_used"] = True
        except Exception as exc:
            paper_answer["llm_error"] = str(exc)

    assistant_message = {
        "role": "assistant",
        "content": paper_answer["answer"],
        "sources": paper_answer.get("sources", []),
        "llm_used": paper_answer.get("llm_used", False),
        "llm_error": paper_answer.get("llm_error"),
    }
    st.session_state.paper_chat_history.append(assistant_message)
    save_paper_chat_history()
    st.session_state.paper_answer = paper_answer
    save_last_paper_answer(paper_answer)
    st.rerun()


def build_contextual_retrieval_query(question: str, max_messages: int = 4) -> str:
    recent_messages = st.session_state.get("paper_chat_history", [])[-max_messages:]
    recent_context = " ".join(str(message.get("content", "")) for message in recent_messages)
    return f"{recent_context} {question}".strip()


def render_source_cards(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(ui_text("Source pages", "来源页码"), expanded=False):
        cards = ['<div class="ra-source-grid">']
        for source in sources:
            cards.append(
                f"""
                <div class="ra-source-card">
                    <div>{escape(str(source["text"]))}</div>
                    <span class="ra-source-page">p.{source["page"]}</span>
                </div>
                """
            )
        cards.append("</div>")
        st.markdown("\n".join(cards), unsafe_allow_html=True)


def render_finding_cards(findings: list[dict[str, str]]) -> None:
    palette = {
        "ERROR": ("#fee2e2", "#b91c1c", "ERROR"),
        "WARNING": ("#fef3c7", "#92400e", "WARNING"),
        "INFO": ("#dbeafe", "#1d4ed8", "INFO"),
    }
    cards = []
    for finding in findings:
        level = str(finding.get("level", "INFO")).upper()
        message = escape(str(finding.get("message", "")))
        background, color, label = palette.get(level, palette["INFO"])
        cards.append(
            f"""
            <div style="
                background: {background};
                color: {color};
                border: 1px solid rgba(15, 23, 42, 0.06);
                border-radius: 10px;
                padding: 0.85rem 1rem;
                margin: 0.55rem 0;
                font-weight: 500;
            ">
                <strong>{label}</strong> · {message}
            </div>
            """
        )
    st.markdown("\n".join(cards), unsafe_allow_html=True)


def render_audit_cards(items: list[dict[str, object]]) -> None:
    if not items:
        return
    palette = {
        "MATCH": ("#dcfce7", "#166534", "MATCH"),
        "MISMATCH": ("#fee2e2", "#b91c1c", "MISMATCH"),
        "UNKNOWN": ("#fef3c7", "#92400e", "UNKNOWN"),
    }
    cards = []
    for item in items:
        level = str(item.get("level", "UNKNOWN")).upper()
        background, color, label = palette.get(level, palette["UNKNOWN"])
        title = escape(str(item.get("item", "")))
        message = escape(str(item.get("message", "")))
        paper = escape(str(item.get("paper", "not specified")))
        user = escape(str(item.get("user", "not specified")))
        cards.append(
            f"""
            <div style="
                background: {background};
                color: {color};
                border: 1px solid rgba(15, 23, 42, 0.06);
                border-radius: 10px;
                padding: 0.9rem 1rem;
                margin: 0.55rem 0;
            ">
                <strong>{label} · {title}</strong><br>
                {message}<br>
                <span style="font-size: 0.88rem;">Paper: {paper} | User: {user}</span>
            </div>
            """
        )
    st.markdown("\n".join(cards), unsafe_allow_html=True)


def render_log_tab() -> None:
    render_section_header("log_title", "log_caption")

    uploaded_log = st.file_uploader(t("upload_log"), type=["txt", "log", "jsonl"], key="train_log")
    if uploaded_log is not None:
        if st.button(t("analyze_log"), type="primary"):
            try:
                text = decode_uploaded_text(uploaded_log)
                st.session_state.log_result = analyze_log(text, uploaded_log.name)
                save_last_log_workspace(uploaded_log.name, text, st.session_state.log_result)
                st.success(t("log_success"))
            except Exception as exc:
                st.error(t("log_error").format(error=exc))

    result = st.session_state.log_result
    if result:
        if LAST_LOG_PATH.exists():
            st.caption(ui_text(
                f"Last log is saved locally: {LAST_LOG_PATH.name}",
                f"最后一次日志已保存到本地：{LAST_LOG_PATH.name}",
            ))
        st.markdown(f"### {t('parsed_metrics')}")
        st.dataframe(result.dataframe, use_container_width=True)

        figures = plot_metrics(result.dataframe)
        if figures:
            st.markdown(f"### {t('curves')}")
            for title, fig in figures.items():
                st.pyplot(fig)
        else:
            st.info(t("no_plots"))

        st.markdown(f"### {t('diagnosis')}")
        for item in result.diagnosis:
            st.write(f"- {item}")

        if result.best_miou:
            st.metric(t("best_miou"), f"{result.best_miou['mIoU']:.4f}", help=str(result.best_miou))

        gap_items = analyze_reproduction_gap(st.session_state.paper_profile, result)
        if gap_items:
            st.markdown("### Reproduction Gap Analysis")
            render_audit_cards(gap_items)


def render_config_tab() -> None:
    render_section_header("config_title", "config_caption")

    uploaded_configs = st.file_uploader(
        t("upload_config"),
        type=["py"],
        key="mmseg_config",
        accept_multiple_files=True,
    )
    if uploaded_configs:
        if st.button(t("check_config"), type="primary"):
            try:
                config_parts = []
                file_names = []
                for uploaded_config in uploaded_configs:
                    file_names.append(uploaded_config.name)
                    config_parts.append(
                        f"\n# ---- {uploaded_config.name} ----\n{decode_uploaded_text(uploaded_config)}"
                    )
                text = "\n".join(config_parts)
                st.session_state.config_result = check_config(text, ", ".join(file_names))
                save_last_config_workspace(file_names, text, st.session_state.config_result)
                st.success(t("config_success"))
            except Exception as exc:
                st.error(t("config_error").format(error=exc))

    result = st.session_state.config_result
    if result:
        if LAST_CONFIG_PATH.exists():
            st.caption(ui_text(
                f"Last config is saved locally: {LAST_CONFIG_PATH.name}",
                f"最后一次配置已保存到本地：{LAST_CONFIG_PATH.name}",
            ))
        st.markdown(f"### {t('extracted_fields')}")
        st.json(result.extracted)

        st.markdown(f"### {t('findings')}")
        render_finding_cards(result.findings)

        consistency_items = compare_config_to_profile(st.session_state.paper_profile, result)
        if consistency_items:
            st.markdown("### Config Consistency Check")
            render_audit_cards(consistency_items)


def render_report_tab() -> None:
    render_section_header("report_title", "report_caption")

    title = st.text_input(t("report_input"), value="ResearchAgent Reproduction Audit Report")
    notes = st.text_area(t("notes"), placeholder=t("notes_placeholder"))

    if st.button(t("generate_report"), type="primary"):
        config_consistency = compare_config_to_profile(st.session_state.paper_profile, st.session_state.config_result)
        reproduction_gap = analyze_reproduction_gap(st.session_state.paper_profile, st.session_state.log_result)
        st.session_state.report_markdown = build_report(
            title=title,
            paper_answer=st.session_state.paper_answer,
            paper_profile=st.session_state.paper_profile,
            log_result=st.session_state.log_result,
            config_result=st.session_state.config_result,
            config_consistency=config_consistency,
            reproduction_gap=reproduction_gap,
            notes=notes,
        )

    if st.session_state.report_markdown:
        st.markdown(f"### {t('markdown_preview')}")
        st.code(st.session_state.report_markdown, language="markdown")
        st.download_button(
            t("download_report"),
            st.session_state.report_markdown,
            file_name="researchagent_report.md",
            mime="text/markdown",
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="RA", layout="wide")
    init_state()
    inject_academic_css()
    render_sidebar()
    render_hero()
    render_reproduction_notice()

    tabs = st.tabs([
        t("tab_paper"),
        t("tab_log"),
        t("tab_config"),
        t("tab_report"),
    ])

    with tabs[0]:
        render_paper_tab()
    with tabs[1]:
        render_log_tab()
    with tabs[2]:
        render_config_tab()
    with tabs[3]:
        render_report_tab()


if __name__ == "__main__":
    main()
