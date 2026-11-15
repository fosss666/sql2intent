"""跨用户混合记忆检索、时间衰减评分和双阈值动作决策。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from cadence.memory.event_log import MemoryEventLog
from cadence.memory.model import (
    CandidateOptionMapping,
    MemoryConfig,
    MemoryDecision,
    MemoryDecisionAction,
    MemoryEvidence,
    MemoryScope,
    PreferenceEventV1,
    PreferenceMemoryError,
    TextEncoder,
)
from cadence.memory.sqlite_store import SQLiteMemoryStore

_SEPARATOR = re.compile(r"[^\w]+", re.UNICODE)
_ENCODER_RUNTIME_FAILURES = frozenset(
    {
        "MEMORY_ENCODER_DEPENDENCY_MISSING",
        "MEMORY_ENCODER_MODEL_LOAD_FAILED",
        "MEMORY_ENCODER_RUNTIME_VERSION_MISMATCH",
        "MEMORY_ENCODER_FAILED",
        "MEMORY_ENCODER_OUTPUT_INVALID",
    }
)


def normalize_key(value: str) -> str:
    """Unicode 兼容规范化、casefold 并统一空白/标点供键与别名精确匹配。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _SEPARATOR.sub(" ", normalized).strip()


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise PreferenceMemoryError("MEMORY_TIME_INVALID", field)
    return value.astimezone(UTC)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """只对同维、有限且非零向量计算 cosine；encoder 故障不得伪装成低相关。"""

    try:
        left_values = tuple(float(item) for item in left)
        right_values = tuple(float(item) for item in right)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PreferenceMemoryError("MEMORY_ENCODER_OUTPUT_INVALID") from exc
    if (
        not left_values
        or len(left_values) != len(right_values)
        or any(not math.isfinite(item) for item in (*left_values, *right_values))
    ):
        raise PreferenceMemoryError("MEMORY_ENCODER_OUTPUT_INVALID")
    left_norm = math.hypot(*left_values)
    right_norm = math.hypot(*right_values)
    if left_norm == 0.0 or right_norm == 0.0:
        raise PreferenceMemoryError("MEMORY_ENCODER_OUTPUT_INVALID")
    # 先归一化再点积，避免对非常大但有限的 encoder 数值发生平方/点积溢出。
    score = sum(
        (a / left_norm) * (b / right_norm)
        for a, b in zip(left_values, right_values, strict=True)
    )
    if not math.isfinite(score):
        raise PreferenceMemoryError("MEMORY_ENCODER_OUTPUT_INVALID")
    return max(-1.0, min(1.0, score))


def _text(event: PreferenceEventV1) -> str:
    """按固定字段拼接 encoder 输入，避免不同调用方拼接顺序漂移。"""

    return " ".join(
        part
        for part in (
            event.ambiguity_text,
            event.context_text or "",
        )
        if part
    )


def _query_text(ambiguity: str, context: str | None) -> str:
    return " ".join(part for part in (ambiguity, context or "") if part)


def _canonical(value: str | None, aliases: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    key = normalize_key(value)
    visited: set[str] = set()
    while key in aliases:
        if key in visited:
            raise PreferenceMemoryError("MEMORY_ALIAS_CYCLE")
        visited.add(key)
        key = aliases[key]
    return key


class MemoryEngine:
    """在 namespace/database/Schema 内跨用户检索，先键/别名再 cosine 回退。"""

    def __init__(
        self,
        event_log: MemoryEventLog | SQLiteMemoryStore,
        *,
        encoder: TextEncoder | None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.event_log = event_log
        self.encoder = encoder
        normalized_aliases: dict[str, str] = {}
        for alias, canonical in (aliases or {}).items():
            alias_key = normalize_key(alias)
            canonical_key = normalize_key(canonical)
            previous = normalized_aliases.get(alias_key)
            if previous is not None and previous != canonical_key:
                raise PreferenceMemoryError("MEMORY_ALIAS_CONFLICT")
            normalized_aliases[alias_key] = canonical_key
        self.aliases = normalized_aliases
        # 在加载配置时预检整张别名图，避免只在特定查询命中时才暴露环路。
        for alias_key in sorted(self.aliases):
            _canonical(alias_key, self.aliases)

    def decide(
        self,
        *,
        scope: MemoryScope,
        query_ambiguity_key: str,
        query_ambiguity_text: str,
        query_context_key: str | None,
        query_context_text: str | None,
        mapping: CandidateOptionMapping,
        config: MemoryConfig,
        as_of: datetime,
        require_cadence_option_set: bool = False,
    ) -> MemoryDecision:
        """返回 ASK/RECOMMEND/AUTO_SELECT；任何证据不足均 fail closed 到 ASK。

        ``require_cadence_option_set`` 会将检索范围限制到相同 semantic root type
        和完整 offered preference-value 集合。
        精确/别名匹配作用于自然语言 ambiguity text；语义检索只在同根类型、同选项集的记录
        内进行。默认 False 使用通用偏好检索范围。
        """

        as_of_utc = _utc(as_of, "as_of")
        if (
            not isinstance(require_cadence_option_set, bool)
            or not query_ambiguity_key.strip()
            or not query_ambiguity_text.strip()
        ):
            raise PreferenceMemoryError("MEMORY_QUERY_INVALID")
        # CADENCE 的 key 是封闭枚举 root type；自然语言 alias 会将
        # PREDICATE 等根类型不能合并，否则会放宽“同一 semantic root type”的检索范围。
        query_key = (
            query_ambiguity_key
            if require_cadence_option_set
            else _canonical(query_ambiguity_key, self.aliases)
        )
        context_key = _canonical(query_context_key, self.aliases)
        if query_key is None:
            raise PreferenceMemoryError("MEMORY_QUERY_INVALID")
        current_option_set = frozenset(
            option.preference_value_id for option in mapping.options
        )
        cadence_evidence_key = _canonical(query_ambiguity_text, self.aliases)

        def matches_cadence_identity(event: PreferenceEventV1) -> bool:
            return (
                event.ambiguity_key == query_ambiguity_key
                and frozenset(
                    option.preference_value_id for option in event.option_mappings
                )
                == current_option_set
            )

        def matches_cadence_evidence(event: PreferenceEventV1) -> bool:
            return _canonical(event.ambiguity_text, self.aliases) == cadence_evidence_key

        if isinstance(self.event_log, SQLiteMemoryStore):
            context_keys: set[str | None] = (
                {None}
                if context_key is None
                else set(self._equivalent_keys(context_key, self.aliases))
            )
            cadence_scope_events = (
                tuple(
                    event
                    for event in self.event_log.read_scope(scope, as_of=as_of_utc)
                    if mapping.option_for_preference(event.preference_value_id) is not None
                    and matches_cadence_identity(event)
                )
                if require_cadence_option_set
                else None
            )
            if require_cadence_option_set:
                exact = [
                    event
                    for event in cadence_scope_events or ()
                    if matches_cadence_evidence(event)
                ]
            else:
                exact = [
                    event
                    for event in self.event_log.find_exact(
                        scope,
                        self._equivalent_keys(query_key, self.aliases),
                        context_keys,
                    )
                    if event.occurred_at <= as_of_utc
                    and mapping.option_for_preference(event.preference_value_id) is not None
                ]
            scoped: list[PreferenceEventV1] | None = None
        else:
            scoped = [
                event
                for event in self.event_log.read_all()
                if event.namespace == scope.namespace
                and event.scope.database_id == scope.database_id
                and event.scope.schema_id == scope.schema_id
                and event.occurred_at <= as_of_utc
                and mapping.option_for_preference(event.preference_value_id) is not None
                and (not require_cadence_option_set or matches_cadence_identity(event))
            ]
            cadence_scope_events = None
            exact = (
                [event for event in scoped if matches_cadence_evidence(event)]
                if require_cadence_option_set
                else [
                    event
                    for event in scoped
                    if _canonical(event.ambiguity_key, self.aliases) == query_key
                    and (
                        (event.context_key is None and context_key is None)
                        or (
                            event.context_key is not None
                            and context_key is not None
                            and _canonical(event.context_key, self.aliases) == context_key
                        )
                    )
                ]
            )
        if exact:
            matched = [(event, 1.0) for event in exact]
            retrieval_mode = "EVIDENCE_ALIAS" if require_cadence_option_set else "KEY_ALIAS"
        elif self.encoder is None and (
            any(
                event.occurred_at <= as_of_utc
                and mapping.option_for_preference(event.preference_value_id) is not None
                for event in (
                    cadence_scope_events
                    if require_cadence_option_set and cadence_scope_events is not None
                    else self.event_log.read_scope(scope)
                    if isinstance(self.event_log, SQLiteMemoryStore)
                    else scoped or ()
                )
            )
        ):
            return self._ask(
                "MEMORY_ENCODER_UNAVAILABLE",
                "NONE",
                scope=scope,
                mapping=mapping,
                config=config,
                as_of=as_of_utc,
                query_ambiguity_key=query_ambiguity_key,
                query_context_key=query_context_key,
            )
        else:
            query_text = _query_text(query_ambiguity_text, query_context_text)
            try:
                if isinstance(self.event_log, SQLiteMemoryStore):
                    if require_cadence_option_set:
                        matched = self._semantic_matches(
                            cadence_scope_events or (),
                            query_text=query_text,
                            config=config,
                        )
                    else:
                        matched = self._semantic_store_matches(
                            self.event_log,
                            scope=scope,
                            query_text=query_text,
                            config=config,
                            as_of=as_of_utc,
                            mapping=mapping,
                        )
                else:
                    matched = self._semantic_matches(
                        scoped or (),
                        query_text=query_text,
                        config=config,
                    )
            except PreferenceMemoryError as exc:
                if exc.reason_code not in _ENCODER_RUNTIME_FAILURES:
                    raise
                # encoder 不可用时不猜测历史偏好；沿用 CADENCE 澄清并在
                # reason code 中保留故障原因，以便运行产物区分无记忆与编码失败。
                return self._ask(
                    exc.reason_code,
                    "NONE",
                    scope=scope,
                    mapping=mapping,
                    config=config,
                    as_of=as_of_utc,
                    query_ambiguity_key=query_ambiguity_key,
                    query_context_key=query_context_key,
                )
            retrieval_mode = "SEMANTIC" if matched else "NONE"
        matched.sort(
            key=lambda item: (
                -item[1],
                -item[0].occurred_at.timestamp(),
                item[0].event_id,
            )
        )
        if not matched:
            return self._ask(
                "MEMORY_NO_RETRIEVABLE_HISTORY",
                retrieval_mode,
                scope=scope,
                mapping=mapping,
                config=config,
                as_of=as_of_utc,
                query_ambiguity_key=query_ambiguity_key,
                query_context_key=query_context_key,
            )
        evidence = self._score(matched, config, as_of_utc, retrieval_mode)
        # 通用配置可设置历史证据门槛；CADENCE 将它留空并由独立策略应用 M_min。
        if (
            config.min_effective_history is not None
            and evidence.effective_history < config.min_effective_history
        ):
            return self._ask(
                "MEMORY_INSUFFICIENT_HISTORY",
                retrieval_mode,
                scope=scope,
                mapping=mapping,
                config=config,
                as_of=as_of_utc,
                query_ambiguity_key=query_ambiguity_key,
                query_context_key=query_context_key,
                evidence=evidence,
            )
        supports = dict(evidence.support_by_value)
        top_values = [
            value_id
            for value_id, value in evidence.support_by_value
            if value == max(supports.values())
        ]
        if len(top_values) != 1:
            return self._ask(
                "MEMORY_PREFERENCE_TIE",
                retrieval_mode,
                scope=scope,
                mapping=mapping,
                config=config,
                as_of=as_of_utc,
                query_ambiguity_key=query_ambiguity_key,
                query_context_key=query_context_key,
                evidence=evidence,
            )
        preferred = top_values[0]
        option = mapping.option_for_preference(preferred)
        if option is None:
            return self._ask(
                "MEMORY_CANDIDATE_MAPPING_INVALID",
                retrieval_mode,
                scope=scope,
                mapping=mapping,
                config=config,
                as_of=as_of_utc,
                query_ambiguity_key=query_ambiguity_key,
                query_context_key=query_context_key,
                evidence=evidence,
            )
        if evidence.top_share >= config.tau_auto:
            action = MemoryDecisionAction.AUTO_SELECT
            reason_code = "MEMORY_AUTO_SELECT_THRESHOLD_MET"
        else:
            action = MemoryDecisionAction.RECOMMEND
            reason_code = "MEMORY_RECOMMEND_BELOW_AUTO"
        decision_id = self._decision_id(
            scope,
            mapping,
            config,
            evidence,
            as_of_utc,
            query_ambiguity_key=query_ambiguity_key,
            query_context_key=query_context_key,
            action=action,
            reason_code=reason_code,
        )
        if action is MemoryDecisionAction.AUTO_SELECT:
            return MemoryDecision(
                action=action,
                reason_code=reason_code,
                decision_id=decision_id,
                preference_value_id=preferred,
                option_id=option.option_id,
                evidence=evidence,
                retrieval_mode=retrieval_mode,
            )
        return MemoryDecision(
            action=action,
            reason_code=reason_code,
            decision_id=decision_id,
            preference_value_id=preferred,
            option_id=option.option_id,
            evidence=evidence,
            retrieval_mode=retrieval_mode,
        )

    def _semantic_matches(
        self,
        events: Sequence[PreferenceEventV1],
        *,
        query_text: str,
        config: MemoryConfig,
    ) -> list[tuple[PreferenceEventV1, float]]:
        if not events or self.encoder is None:
            return []
        if self.encoder.encoder_id != config.encoder_id:
            raise PreferenceMemoryError("MEMORY_ENCODER_CONFIG_MISMATCH")
        try:
            query_vector = self.encoder.encode(query_text)
        except Exception as exc:  # encoder failures must not become false preference matches.
            raise PreferenceMemoryError("MEMORY_ENCODER_FAILED") from exc
        matches: list[tuple[PreferenceEventV1, float]] = []
        for event in events:
            try:
                event_vector = self.encoder.encode(_text(event))
            except Exception as exc:
                raise PreferenceMemoryError("MEMORY_ENCODER_FAILED") from exc
            score = _cosine(query_vector, event_vector)
            if score >= config.tau_retrieval:
                matches.append((event, score))
        return matches

    @staticmethod
    def _equivalent_keys(
        canonical: str, aliases: Mapping[str, str]
    ) -> set[str]:
        """为 SQLite B-tree 查询生成当前规范键及所有指向它的别名键。"""

        return {
            key
            for key in (canonical, *aliases.keys())
            if _canonical(key, aliases) == canonical
        }

    def _semantic_store_matches(
        self,
        store: SQLiteMemoryStore,
        *,
        scope: MemoryScope,
        query_text: str,
        config: MemoryConfig,
        as_of: datetime,
        mapping: CandidateOptionMapping,
    ) -> list[tuple[PreferenceEventV1, float]]:
        """将缺失向量增量补入 SQLite，再在 scope 分区执行 cosine KNN。"""

        if self.encoder is None:
            return []
        if self.encoder.encoder_id != config.encoder_id:
            raise PreferenceMemoryError("MEMORY_ENCODER_CONFIG_MISMATCH")
        try:
            for event in store.events_missing_embeddings(scope, config.encoder_id):
                store.upsert_embedding(
                    event.event_id,
                    event.scope,
                    config.encoder_id,
                    self.encoder.encode(_text(event)),
                )
            query_vector = self.encoder.encode(query_text)
        except PreferenceMemoryError:
            raise
        except Exception as exc:  # encoder failures must not become false preference matches.
            raise PreferenceMemoryError("MEMORY_ENCODER_FAILED") from exc

        matches: list[tuple[PreferenceEventV1, float]] = []
        for event, score in store.search_similar(
            scope,
            config.encoder_id,
            query_vector,
            top_k=config.semantic_top_k,
        ):
            if (
                score >= config.tau_retrieval
                and event.occurred_at <= as_of
                and mapping.option_for_preference(event.preference_value_id) is not None
            ):
                matches.append((event, score))
        return matches

    @staticmethod
    def _score(
        matched: Sequence[tuple[PreferenceEventV1, float]],
        config: MemoryConfig,
        as_of: datetime,
        retrieval_mode: str,
    ) -> MemoryEvidence:
        support: defaultdict[str, float] = defaultdict(float)
        ids: list[str] = []
        for event, _ in matched:
            age_days = max(0, (as_of.date() - event.occurred_at.astimezone(UTC).date()).days)
            weight = 2.0 ** (-age_days / config.half_life_days)
            support[event.preference_value_id] += weight
            ids.append(event.event_id)
        ordered_support = tuple(sorted(support.items(), key=lambda item: item[0]))
        total = sum(value for _, value in ordered_support)
        top = max((value for _, value in ordered_support), default=0.0)
        preferred = min(
            (key for key, value in ordered_support if value == top),
            default="",
        )
        return MemoryEvidence(
            support_by_value=ordered_support,
            effective_history=total,
            top_share=(top / total) if total else 0.0,
            preferred_value_id=preferred,
            retrieved_event_ids=tuple(ids),
            retrieval_mode=retrieval_mode,
        )

    @staticmethod
    def _decision_id(
        scope: MemoryScope,
        mapping: CandidateOptionMapping,
        config: MemoryConfig,
        evidence: MemoryEvidence,
        as_of: datetime,
        *,
        query_ambiguity_key: str,
        query_context_key: str | None,
        action: MemoryDecisionAction,
        reason_code: str,
    ) -> str:
        preimage = {
            "scope": [scope.namespace, scope.database_id, scope.schema_id],
            "candidate_set_key": mapping.candidate_set_key,
            "query_ambiguity_key": normalize_key(query_ambiguity_key),
            "query_context_key": normalize_key(query_context_key or ""),
            "parameter_set_id": config.parameter_set_id,
            "calibration_id": config.calibration_id,
            "validation_fingerprint": config.validation_fingerprint,
            "encoder_id": config.encoder_id,
            "parameters": {
                "tau_retrieval": config.tau_retrieval,
                "semantic_top_k": config.semantic_top_k,
                "half_life_days": config.half_life_days,
                "min_effective_history": config.min_effective_history,
                "tau_auto": config.tau_auto,
            },
            "as_of": as_of.isoformat().replace("+00:00", "Z"),
            "event_ids": list(evidence.retrieved_event_ids),
            "support_by_value": list(evidence.support_by_value),
            "action_evidence": [evidence.effective_history, evidence.top_share],
            "action": action.value,
            "reason_code": reason_code,
        }
        encoded = json.dumps(
            preimage, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _ask(
        cls,
        reason_code: str,
        retrieval_mode: str,
        *,
        scope: MemoryScope,
        mapping: CandidateOptionMapping,
        config: MemoryConfig,
        as_of: datetime,
        query_ambiguity_key: str,
        query_context_key: str | None,
        evidence: MemoryEvidence | None = None,
    ) -> MemoryDecision:
        if evidence is None:
            evidence = MemoryEvidence(
                support_by_value=(),
                effective_history=0.0,
                top_share=0.0,
                preferred_value_id="",
                retrieved_event_ids=(),
                retrieval_mode=retrieval_mode,
            )
        return MemoryDecision(
            action=MemoryDecisionAction.ASK,
            reason_code=reason_code,
            decision_id=cls._decision_id(
                scope,
                mapping,
                config,
                evidence,
                as_of,
                query_ambiguity_key=query_ambiguity_key,
                query_context_key=query_context_key,
                action=MemoryDecisionAction.ASK,
                reason_code=reason_code,
            ),
            evidence=evidence,
            retrieval_mode=retrieval_mode,
        )
