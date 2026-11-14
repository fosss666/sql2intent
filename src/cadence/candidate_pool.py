"""把统一候选池请求送入 CADENCE 的 SQL 因子化入口。"""

from __future__ import annotations

from cadence.datasets.common import StandardRequest
from cadence.factorize_sql import factorize_sql
from cadence.grouping import FactorizedCandidate
from cadence.parse_bind import SchemaCatalog


def factorize_request(request: StandardRequest) -> tuple[FactorizedCandidate, ...]:
    """按请求中的 Schema 方言与显式时间列因子化候选 SQL。"""

    catalog = SchemaCatalog.from_definition(request.catalog)
    dialect = request.catalog.get("dialect_id")
    if not isinstance(dialect, str) or not dialect.strip():
        raise ValueError("CADENCE_CANDIDATE_POOL_DIALECT_INVALID")
    return tuple(
        factorize_sql(
            sql=candidate["sql_text"],
            candidate_id=candidate["candidate_id"],
            catalog=catalog,
            dialect=dialect,
            temporal_columns=request.temporal_columns,
        )
        for candidate in request.candidates
    )
