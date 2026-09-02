"""
Medaudit-RAG 全局配置管理
使用 pydantic-settings 统一管理环境变量与系统参数
支持多 LLM 供应商切换 (zhipu / dashscope / deepseek)
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Literal
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """应用配置 — 从 .env 文件或环境变量加载"""

    _ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
    _CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"

    # ── 应用基础 ──
    APP_NAME: str = "Medaudit-RAG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── LLM 供应商切换 ──
    LLM_PROVIDER: Literal["zhipu", "dashscope", "deepseek"] = Field(
        default="zhipu", description="LLM 供应商: zhipu / dashscope / deepseek"
    )

    # ── API Keys (按供应商填写对应的即可) ──
    ZHIPU_API_KEY: str = Field(default="", description="智谱 AI API Key")
    DASHSCOPE_API_KEY: str = Field(default="", description="通义千问 API Key")
    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek API Key")

    # ── 模型配置 ──
    LLM_MODEL_GENERATOR: str = Field(
        default="glm-4.5-air",
        description="生成/推理用模型 (开发阶段推荐 glm-4.5-air 免费额度)"
    )
    LLM_MODEL_JUDGE: str = Field(
        default="glm-4.5-air",
        description="审计 Agent 用模型 (可用更强的 glm-4.7 或 qwen-plus)"
    )
    LLM_MODEL_ROUTER: str = Field(
        default="glm-4.5-air",
        description="意图路由用模型 (轻量任务, 用便宜模型即可)"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.01, description="LLM 温度, 低温保证确定性输出"
    )

    # ── Embedding 模型 ──
    EMBEDDING_PROVIDER: Literal["zhipu", "dashscope", "local"] = Field(
        default="zhipu", description="Embedding 供应商: zhipu / dashscope / local"
    )
    EMBEDDING_MODEL: str = Field(
        default="embedding-3",
        description="向量 Embedding 模型"
    )
    LOCAL_EMBEDDING_MODEL_PATH: str = Field(
        default="",
        description="本地 Embedding 模型目录；为空时使用 EMBEDDING_MODEL",
    )
    LOCAL_EMBEDDING_DEVICE: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="本地 Embedding 推理设备",
    )
    LOCAL_EMBEDDING_BATCH_SIZE: int = Field(
        default=4,
        ge=1,
        description="本地 Embedding 编码批大小",
    )
    LOCAL_EMBEDDING_DIMENSION: int = Field(
        default=0,
        ge=0,
        description="本地 Embedding 预期向量维度；0 表示不做维度门控",
    )

    # ── 本地重排模型（Phase 7 C1c-4；实际重排逻辑由实验管线启用） ──
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="本地证据重排模型标识",
    )
    RERANKER_MODEL_PATH: str = Field(
        default="",
        description="本地重排模型目录；为空时使用 RERANKER_MODEL",
    )
    RERANKER_DEVICE: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="本地重排模型推理设备",
    )
    RERANKER_BATCH_SIZE: int = Field(
        default=4,
        ge=1,
        description="本地重排模型批大小",
    )
    RERANKER_CANDIDATE_K: int = Field(
        default=20,
        ge=1,
        description="进入重排阶段的候选证据数",
    )

    # ── 向量检索 ──
    CHROMA_PERSIST_DIR: str = Field(
        default=str(_CHROMA_DIR), description="ChromaDB 持久化目录"
    )
    RETRIEVAL_TOP_K: int = Field(default=3, description="Top-K 检索窗口")

    # ── 三粒度索引参数 (论文 Table 2) ──
    CHUNK_DETAIL_SIZE: int = 128       # Gdetail: 微观粒度
    CHUNK_DETAIL_OVERLAP: int = 0      # 0% overlap
    CHUNK_CONCEPT_SIZE: int = 512      # Gconcept: 标准粒度
    CHUNK_CONCEPT_OVERLAP: int = 77    # ~15% overlap
    CHUNK_CONTEXT_SIZE: int = 1024     # Gcontext: 宏观粒度
    CHUNK_CONTEXT_OVERLAP: int = 205   # ~20% overlap

    # ── Trust-Score 门控参数 (论文 Eq.4 + Grid Search 最优解) ──
    TRUST_ALPHA: float = Field(default=0.4, description="检索相关性权重 α")
    TRUST_BETA: float = Field(default=0.6, description="忠实度权重 β")
    TRUST_THRESHOLD_HIGH: float = Field(default=7.5, description="TRUSTED 阈值 τ_high")
    TRUST_THRESHOLD_LOW: float = Field(default=5.0, description="REJECTED 阈值 τ_low")

    # ── Margin-Aware Calibration 参数 (论文 Eq.1) ──
    CALIBRATION_TAU_BASE: float = Field(default=0.35, description="余弦相似度基准阈值 τ_base")
    CALIBRATION_GAMMA: float = Field(default=20.0, description="放大因子 γ")

    # ── 医学证据等级权重 ──
    AUTHORITY_WEIGHTS: dict = Field(
        default={
            "national_pharmacopoeia": 1.0,   # 国家药典
            "clinical_guideline": 0.9,        # 临床诊疗指南
            "expert_consensus": 0.7,          # 专家共识
            "textbook": 0.6,                  # 教材
            "case_report": 0.3,               # 个案报道
            "default": 0.5,                   # 未标注来源
        },
        description="文献来源 → 权威度权重映射"
    )

    # ── CORS ──
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

    @field_validator("DEBUG", mode="before")
    @classmethod
    def _parse_debug_value(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value

    def get_active_api_key(self) -> str:
        """根据当前 LLM_PROVIDER 返回对应的 API Key"""
        key_map = {
            "zhipu": self.ZHIPU_API_KEY,
            "dashscope": self.DASHSCOPE_API_KEY,
            "deepseek": self.DEEPSEEK_API_KEY,
        }
        return key_map[self.LLM_PROVIDER]

    def get_base_url(self) -> str | None:
        """返回各供应商的 OpenAI 兼容 Base URL"""
        url_map = {
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "deepseek": "https://api.deepseek.com",
        }
        return url_map[self.LLM_PROVIDER]


@lru_cache()
def get_settings() -> Settings:
    """单例获取配置 (带缓存)"""
    return Settings()
