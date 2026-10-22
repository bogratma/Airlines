
from __future__ import annotations

from collections import defaultdict
from hashlib import sha1

import polars as pl

SOURCE_PRIORITY = {"pdf": 0, "tab": 1, "csv": 2, "boarding": 3, "yaml": 4, "xml": 5, "json": 6}
EVENT_FIELDS = [
    "dep_airport", "arr_airport", "dep_ts", "arr_ts", "name_key", "birth_date",
    "passport", "ticket", "booking_code", "card_program", "card_number",
    "nickname", "travel_class", "fare", "agent",
]

ID_COLS = ["_i_record", "_i_pass", "_i_tick", "_i_nb"]


def _find(parent: dict[str, str], value: str) -> str:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _clean_expr(value: pl.Expr) -> pl.Expr:
    value = value.cast(pl.String).str.strip_chars()
    return pl.when(
        value.is_null()
        | (value == "")
        | value.str.to_uppercase().is_in(["N/A", "NA", "NULL", "NONE", "NOT PRESENTED", "UNKNOWN"])
    ).then(pl.lit(None, dtype=pl.String)).otherwise(value)


def _key_expr(value: pl.Expr, prefix: str) -> pl.Expr:
    clean = _clean_expr(value)
    return pl.when(clean.is_not_null()).then(
        pl.concat_str([pl.lit(prefix + ":"), clean])
    ).otherwise(pl.lit(None, dtype=pl.String))


def _ensure_identity_columns(u: pl.DataFrame) -> pl.DataFrame:
    missing = {
        "source": pl.String,
        "source_record_id": pl.String,
        "passport": pl.String,
        "ticket": pl.String,
        "name_key": pl.String,
        "birth_date": pl.String,
    }
    for column, dtype in missing.items():
        if column not in u.columns:
            u = u.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return u


def _ambiguous_values(u: pl.DataFrame, column: str) -> list[str]:
    candidates = (
        u.filter(pl.col(column).is_not_null() & (pl.col(column).cast(pl.String).str.strip_chars() != ""))
        .group_by(column)
        .agg(
            pl.col("name_key").drop_nulls().n_unique().alias("name_count"),
            pl.col("birth_date").drop_nulls().n_unique().alias("birth_count"),
        )
        .filter((pl.col("name_count") > 1) | (pl.col("birth_count") > 1))
        .select(column)
    )
    return candidates.get_column(column).to_list()


def assign_person_id(u: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(u, pl.DataFrame):
        raise TypeError("assign_person_id ожидает polars.DataFrame")

    u = _ensure_identity_columns(u)
    if "person_id" in u.columns:
        u = u.drop("person_id")
    u = u.with_row_index("_row")

    ambiguous_passports = _ambiguous_values(u, "passport")
    ambiguous_tickets = _ambiguous_values(u, "ticket")
    if ambiguous_passports or ambiguous_tickets:
        print(
            "Entity resolution: конфликтные ключи не используются для склейки — "
            f"паспортов: {len(ambiguous_passports):,}, билетов: {len(ambiguous_tickets):,}."
        )
    repeated_records = (
        u.filter(pl.col("source_record_id").is_not_null())
        .group_by("source_record_id")
        .len()
        .filter(pl.col("len") > 1)
        .select("source_record_id")
        .get_column("source_record_id")
        .to_list()
    )

    passport_expr = _key_expr(pl.col("passport"), "pass")
    if ambiguous_passports:
        passport_expr = pl.when(pl.col("passport").is_in(ambiguous_passports)).then(None).otherwise(passport_expr)

    ticket_expr = _key_expr(pl.col("ticket"), "ticket")
    if ambiguous_tickets:
        ticket_expr = pl.when(pl.col("ticket").is_in(ambiguous_tickets)).then(None).otherwise(ticket_expr)

    record_expr = _key_expr(pl.col("source_record_id"), "record")
    if repeated_records:
        record_expr = pl.when(pl.col("source_record_id").is_in(repeated_records)).then(record_expr).otherwise(None)
    else:
        record_expr = pl.lit(None, dtype=pl.String)

    name_expr = _clean_expr(pl.col("name_key"))
    birth_expr = _clean_expr(pl.col("birth_date"))
    name_birth_expr = pl.when(name_expr.is_not_null() & birth_expr.is_not_null()).then(
        pl.concat_str([pl.lit("name_birth:"), name_expr, pl.lit("|"), birth_expr])
    ).otherwise(pl.lit(None, dtype=pl.String))

    u = u.with_columns(
        record_expr.alias("_i_record"),
        passport_expr.alias("_i_pass"),
        ticket_expr.alias("_i_tick"),
        name_birth_expr.alias("_i_nb"),
    )

    parent: dict[str, str] = {}
    unique_key_rows = u.select(ID_COLS).unique().iter_rows()
    for row in unique_key_rows:
        keys = [value for value in row if value is not None]
        if not keys:
            continue
        for key in keys:
            parent.setdefault(key, key)
        first_key = keys[0]
        for other_key in keys[1:]:
            root_a = _find(parent, first_key)
            root_b = _find(parent, other_key)
            if root_a != root_b:
                if root_a < root_b:
                    parent[root_b] = root_a
                else:
                    parent[root_a] = root_b

    components: dict[str, list[str]] = defaultdict(list)
    for key in sorted(parent):
        components[_find(parent, key)].append(key)

    root_to_pid = {}
    for root, keys in components.items():
        digest = sha1("\n".join(sorted(keys)).encode("utf-8")).hexdigest()[:12].upper()
        root_to_pid[root] = f"P-{digest}"

    idents, person_ids = [], []
    for key in parent:
        idents.append(key)
        person_ids.append(root_to_pid[_find(parent, key)])
    idmap = pl.DataFrame(
        {"ident": idents, "mapped_person_id": person_ids},
        schema={"ident": pl.String, "mapped_person_id": pl.String},
    )

    for column in ID_COLS:
        u = u.join(
            idmap.rename({"ident": column, "mapped_person_id": f"_pid_{column}"}),
            on=column,
            how="left",
        )

    id_columns = [pl.col(f"_pid_{column}") for column in ID_COLS]
    u = u.with_columns(
        person_id=pl.coalesce(id_columns).cast(pl.String)
    ).with_columns(
        person_id=pl.when(pl.col("person_id").is_not_null())
        .then(pl.col("person_id"))
        .otherwise(pl.concat_str([pl.lit("ROW-"), pl.col("_row").cast(pl.String)]))
    )
    return u.drop(["_row", *ID_COLS, *[f"_pid_{column}" for column in ID_COLS]])


def merge_events(u: pl.DataFrame) -> pl.DataFrame:
    required = {"person_id", "flight_code", "date", "source"}
    missing = required - set(u.columns)
    if missing:
        raise ValueError(f"merge_events: отсутствуют колонки {sorted(missing)}")

    events = u.filter(
        pl.col("person_id").is_not_null()
        & pl.col("flight_code").is_not_null()
        & pl.col("date").is_not_null()
    )
    if events.is_empty():
        return pl.DataFrame(schema={
            "person_id": pl.String,
            "flight_code": pl.String,
            "date": pl.String,
            "sources": pl.List(pl.String),
            "n_sources": pl.UInt32,
            "dep_conflict": pl.Boolean,
            "arr_conflict": pl.Boolean,
            "name_conflict": pl.Boolean,
            "passport_conflict": pl.Boolean,
            "ticket_conflict": pl.Boolean,
        })

    priority = pl.col("source").replace_strict(SOURCE_PRIORITY, default=9, return_dtype=pl.Int64)
    events = events.with_columns(_priority=priority).sort("_priority", maintain_order=True)

    expressions = [
        pl.col("source").drop_nulls().unique().sort().alias("sources"),
        (pl.col("dep_airport").drop_nulls().n_unique() > 1).alias("dep_conflict"),
        (pl.col("arr_airport").drop_nulls().n_unique() > 1).alias("arr_conflict"),
    ]
    for column in EVENT_FIELDS:
        if column in events.columns:
            expressions.append(pl.col(column).drop_nulls().first().alias(column))
    for column, alias in (
        ("name_key", "name_conflict"),
        ("passport", "passport_conflict"),
        ("ticket", "ticket_conflict"),
    ):
        if column in events.columns:
            expressions.append((pl.col(column).drop_nulls().n_unique() > 1).alias(alias))
        else:
            expressions.append(pl.lit(False).alias(alias))
    if "source_record_id" in events.columns:
        expressions.append(
            pl.col("source_record_id").drop_nulls().unique().sort().alias("source_record_ids")
        )

    result = events.group_by("person_id", "flight_code", "date", maintain_order=True).agg(expressions)
    return result.with_columns(pl.col("sources").list.len().alias("n_sources"))
