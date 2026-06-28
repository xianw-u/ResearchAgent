TOOL_REGISTRY = {
    "paper_qa": {
        "label_en": "Paper Retrieval QA",
        "label_zh": "论文检索问答",
        "description_en": "Extract PDF text, build a TF-IDF retrieval index, and answer with page citations.",
        "description_zh": "解析论文 PDF，构建 TF-IDF 检索索引，并返回带页码来源的回答。",
    },
    "training_log_analyzer": {
        "label_en": "Training Log Diagnosis",
        "label_zh": "训练日志诊断",
        "description_en": "Parse training metrics, plot curves, and diagnose convergence signals.",
        "description_zh": "解析 loss、lr、mIoU 等指标，绘制曲线并诊断训练收敛状态。",
    },
    "mmseg_config_checker": {
        "label_en": "MMSeg Config Audit",
        "label_zh": "MMSeg 配置审计",
        "description_en": "Inspect MMSegmentation configs without executing uploaded Python code.",
        "description_zh": "在不执行上传代码的前提下，检查数据集、类别数、学习率和 pipeline 等配置。",
    },
    "report_generator": {
        "label_en": "Experiment Report Writer",
        "label_zh": "实验报告生成",
        "description_en": "Assemble analysis results into a Markdown experiment report.",
        "description_zh": "整合论文问答、日志分析和配置检查结果，生成 Markdown 实验报告。",
    },
}
