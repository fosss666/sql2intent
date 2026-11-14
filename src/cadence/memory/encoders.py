"""Memory 的可复现本地语义编码器。"""

from __future__ import annotations

import math
import threading
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from cadence.memory.model import PreferenceMemoryError

E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
E5_EMBEDDING_DIMENSION = 384
E5_MAX_SEQUENCE_LENGTH = 512
E5_ENCODER_ID = (
    f"{E5_MODEL_ID}@{E5_MODEL_REVISION}"
    ":query-prefix:l2-normalized-f32:cpu:max-seq-512"
    ":st-6.2.0:torch-2.15.0:transformers-5.22.1:tokenizers-0.23.3:numpy-2.5.4"
)
E5_RUNTIME_VERSIONS = (
    ("sentence-transformers", "6.2.0"),
    ("torch", "2.15.0"),
    ("transformers", "5.22.1"),
    ("tokenizers", "0.23.3"),
    ("numpy", "2.5.4"),
)

_MODEL_LOAD_LOCK = threading.Lock()
_SHARED_E5_MODEL: Any | None = None


def _check_e5_runtime_versions() -> None:
    """确认当前推理栈与 encoder ID 绑定版本相同，避免混用旧向量缓存。"""

    for distribution, expected in E5_RUNTIME_VERSIONS:
        try:
            actual = version(distribution)
        except PackageNotFoundError as exc:
            raise PreferenceMemoryError("MEMORY_ENCODER_DEPENDENCY_MISSING") from exc
        if actual != expected:
            raise PreferenceMemoryError(
                "MEMORY_ENCODER_RUNTIME_VERSION_MISMATCH", distribution
            )


def _load_e5_model() -> Any:
    """按固定模型、revision 与 CPU 设备加载一次模型并限制最大输入长度。"""

    _check_e5_runtime_versions()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise PreferenceMemoryError("MEMORY_ENCODER_DEPENDENCY_MISSING") from exc

    try:
        model = SentenceTransformer(
            E5_MODEL_ID,
            revision=E5_MODEL_REVISION,
            device="cpu",
        )
        # E5 模型卡规定最多处理 512 tokens；长文本由 tokenizer 按同一上限截断。
        model.max_seq_length = E5_MAX_SEQUENCE_LENGTH
    except Exception as exc:
        raise PreferenceMemoryError("MEMORY_ENCODER_MODEL_LOAD_FAILED") from exc
    return model


def _shared_e5_model() -> Any:
    """跨 adapter 共享进程内模型，避免每个交互轮次重新加载权重。"""

    global _SHARED_E5_MODEL
    if _SHARED_E5_MODEL is None:
        with _MODEL_LOAD_LOCK:
            if _SHARED_E5_MODEL is None:
                _SHARED_E5_MODEL = _load_e5_model()
    return _SHARED_E5_MODEL


class MultilingualE5SmallEncoder:
    """对歧义文本与历史事件文本使用同一 E5 query 前缀生成 L2 归一向量。

    编码器只在首次调用 ``encode`` 时加载；安装 ``memory-encoder`` 可选依赖后，
    SentenceTransformer 会按固定 revision 从本地缓存或模型仓库取得权重。CPU 推理、
    float32 输出和固定长度共同避免部署设备或隐式模型升级改变向量身份。
    """

    encoder_id = E5_ENCODER_ID

    def encode(self, text: str) -> tuple[float, ...]:
        """将文本确定性地映射到 384 维有限 float32 向量。"""

        if not isinstance(text, str) or not text.strip():
            raise PreferenceMemoryError("MEMORY_ENCODER_INPUT_INVALID")
        try:
            output = _shared_e5_model().encode(
                [f"query: {text}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                precision="float32",
            )
            vector = tuple(float(value) for value in output[0])
        except PreferenceMemoryError:
            raise
        except Exception as exc:
            raise PreferenceMemoryError("MEMORY_ENCODER_FAILED") from exc
        if len(vector) != E5_EMBEDDING_DIMENSION or any(
            not math.isfinite(value) for value in vector
        ):
            raise PreferenceMemoryError("MEMORY_ENCODER_OUTPUT_INVALID")
        return vector
