from __future__ import annotations

import re
from typing import Any

import pandas as pd
import polars as pl
from unidecode import unidecode

SCHEMA = {
    "source": pl.String,
    "source_record_id": pl.String,
    "flight_code": pl.String,
    "date": pl.String,
    "dep_airport": pl.String,
    "arr_airport": pl.String,
    "dep_ts": pl.Datetime("us"),
    "arr_ts": pl.Datetime("us"),
    "name_key": pl.String,
    "birth_date": pl.String,
    "passport": pl.String,
    "ticket": pl.String,
    "booking_code": pl.String,
    "card_program": pl.String,
    "card_number": pl.String,
    "nickname": pl.String,
    "travel_class": pl.String,
    "fare": pl.String,
    "agent": pl.String,
}

MISSING_VALUES = {"", "N/A", "NA", "NULL", "NONE", "NOT PRESENTED", "UNKNOWN"}


def _has(df: pl.DataFrame, name: str) -> bool:
    return name in df.columns


def _c(df: pl.DataFrame, name: str) -> pl.Expr:
    if not _has(df, name):
        return pl.lit(None, dtype=pl.String)
    return pl.col(name).cast(pl.String)


def _norm(expr: pl.Expr) -> pl.Expr:
    value = expr.cast(pl.String).str.strip_chars()
    upper = value.str.to_uppercase()
    return (
        pl.when(value.is_null() | upper.is_in(sorted(MISSING_VALUES)))
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(value)
    )


def _nonempty(expr: pl.Expr) -> pl.Expr:
    value = _norm(expr)
    return pl.when(value.is_not_null() & (value.str.len_chars() > 0)).then(value).otherwise(None)


def _id(expr: pl.Expr) -> pl.Expr:
    value = _nonempty(expr).str.to_uppercase().str.replace_all(r"[^0-9A-ZА-ЯЁ]", "")
    return pl.when(value.is_not_null() & (value != "")).then(value).otherwise(None)


def _card_number(expr: pl.Expr) -> pl.Expr:
    value = _nonempty(expr).str.to_uppercase().str.replace(r"\.0+$", "")
    value = value.str.replace_all(r"[^A-Z0-9]", "")
    return pl.when(value.is_not_null() & (value != "")).then(value).otherwise(None)


def _program(expr: pl.Expr) -> pl.Expr:
    value = _nonempty(expr).str.to_uppercase().str.replace_all(r"\s+", " ").str.strip_chars()
    return pl.when(value.is_not_null() & (value != "")).then(value).otherwise(None)


def _ticket(expr: pl.Expr) -> pl.Expr:
    value = _nonempty(expr).str.replace(r"\.0+$", "").str.replace_all(r"\D", "")
    return pl.when(value.is_not_null() & (value != "")).then(value).otherwise(None)


def _flight(expr: pl.Expr) -> pl.Expr:
    value = _norm(expr).str.to_uppercase().str.replace_all(r"[\s\-]", "")
    airline = value.str.extract(r"^([A-Z0-9]{2})\d", 1)
    number = value.str.extract(r"^[A-Z0-9]{2}(\d+)$", 1)
    number = number.str.strip_chars_start("0")
    number = pl.when(number == "").then(pl.lit("0")).otherwise(number)
    result = pl.concat_str([airline, number])
    return pl.when(airline.is_not_null() & number.is_not_null()).then(result).otherwise(None)


def _iata(expr: pl.Expr) -> pl.Expr:
    value = _nonempty(expr).str.to_uppercase()
    return pl.when(value.str.len_chars() == 3).then(value).otherwise(value)


def _date(df: pl.DataFrame, name: str) -> pl.Expr:
    if not _has(df, name):
        return pl.lit(None, dtype=pl.String)

    dtype = df.schema[name]
    col = pl.col(name)
    if dtype == pl.Date or isinstance(dtype, pl.Datetime):
        return col.dt.strftime("%Y-%m-%d")

    value = _norm(col)
    iso = value.str.extract(r"^(\d{4}-\d{2}-\d{2})", 1)
    dmy = value.str.to_date("%d.%m.%Y", strict=False).dt.strftime("%Y-%m-%d")
    slash = value.str.to_date("%Y/%m/%d", strict=False).dt.strftime("%Y-%m-%d")
    return pl.coalesce([iso, dmy, slash])


def _parse_datetime_string(expr: pl.Expr) -> pl.Expr:
    value = _norm(expr)
    return pl.coalesce([
        value.str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        value.str.to_datetime("%Y-%m-%d %H:%M", strict=False),
        value.str.to_datetime("%d.%m.%Y %H:%M:%S", strict=False),
        value.str.to_datetime("%d.%m.%Y %H:%M", strict=False),
        value.str.to_datetime(strict=False),
    ])


def _ts_col(df: pl.DataFrame, name: str) -> pl.Expr:
    if not _has(df, name):
        return pl.lit(None, dtype=pl.Datetime("us"))
    dtype = df.schema[name]
    col = pl.col(name)
    if dtype == pl.Date:
        return col.cast(pl.Datetime("us"))
    if isinstance(dtype, pl.Datetime):
        return col.cast(pl.Datetime("us"), strict=False)
    return _parse_datetime_string(col).cast(pl.Datetime("us"), strict=False)


def _ts_from_parts(df: pl.DataFrame, date_name: str, time_name: str) -> pl.Expr:
    if not (_has(df, date_name) and _has(df, time_name)):
        return pl.lit(None, dtype=pl.Datetime("us"))
    date_value = _date(df, date_name)
    time_value = _norm(pl.col(time_name).cast(pl.String))
    text = pl.concat_str([date_value, time_value], separator=" ")
    return pl.coalesce([
        text.str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        text.str.to_datetime("%Y-%m-%d %H:%M", strict=False),
        text.str.to_datetime(strict=False),
    ]).cast(pl.Datetime("us"), strict=False)


def canon_name_value(raw: Any, max_tokens: int | None = None) -> str | None:
    if raw is None:
        return None
    text = unidecode(str(raw)).upper()
    text = re.sub(r"[’'`\-]", "", text)
    tokens = [token for token in re.split(r"[^A-Z0-9]+", text) if len(token) > 1]
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    canonical = " ".join(sorted(tokens))
    return canonical or None


def canon_name(raw: Any) -> str | None:
    return canon_name_value(raw)


def _ensure_source_row(df: pl.DataFrame) -> pl.DataFrame:
    if "_source_row" in df.columns:
        return df
    return df.with_row_index("_source_row")


def _with_name_key(
    df: pl.DataFrame,
    raw_expr: pl.Expr,
    *,
    max_tokens: int | None = None,
) -> pl.DataFrame:
    df = df.with_columns(_norm(raw_expr).alias("_raw_name"))
    unique_names = df.get_column("_raw_name").drop_nulls().unique().to_list()
    mapping = {}
    for raw in unique_names:
        canon = canon_name_value(raw, max_tokens=max_tokens)
        if canon:
            mapping[raw] = canon
    return df.with_columns(
        pl.col("_raw_name")
        .replace_strict(mapping, default=None, return_dtype=pl.String)
        .alias("_name_key")
    )


def _finish(df: pl.DataFrame, source: str, mapping: dict[str, pl.Expr]) -> pl.DataFrame:
    expressions = []
    for column, dtype in SCHEMA.items():
        if column == "source":
            expr = pl.lit(source, dtype=pl.String)
        elif column == "source_record_id":
            if _has(df, "_source_row"):
                expr = pl.concat_str([
                    pl.lit(f"{source}:"), pl.col("_source_row").cast(pl.String)
                ])
            else:
                expr = pl.lit(None, dtype=pl.String)
        elif column in mapping:
            expr = mapping[column]
        else:
            expr = pl.lit(None, dtype=dtype)
        expressions.append(expr.cast(dtype, strict=False).alias(column))
    return df.select(expressions)


def _first_existing(df: pl.DataFrame, candidates: tuple[str, ...]) -> pl.Expr:
    for candidate in candidates:
        if candidate in df.columns:
            return pl.col(candidate)
    return pl.lit(None, dtype=pl.String)


def unify_tab(df: pl.DataFrame) -> pl.DataFrame:
    df = _ensure_source_row(df)
    df = _with_name_key(df, _c(df, "PaxName"), max_tokens=2)
    return _finish(df, "tab", {
        "flight_code": _flight(_c(df, "FlightCodeSh")),
        "date": _date(df, "DepartDate"),
        "dep_airport": _iata(_c(df, "From")),
        "arr_airport": _iata(_c(df, "Dest")),
        "dep_ts": _ts_from_parts(df, "DepartDate", "DepartTime"),
        "arr_ts": _ts_from_parts(df, "ArrivalDate", "ArrivalTime"),
        "name_key": pl.col("_name_key"),
        "birth_date": _date(df, "PaxBirthDate"),
        "passport": _id(_c(df, "TravelDoc")),
        "ticket": _ticket(_c(df, "e-Ticket")),
        "booking_code": _norm(_c(df, "Code")),
        "card_program": _program(_c(df, "loyalty_program")),
        "card_number": _card_number(_c(df, "loyalty_number")),
        "travel_class": _norm(_c(df, "TrvCls")),
        "fare": _norm(_c(df, "Fare")),
        "agent": _norm(_c(df, "AgentInfo")),
    })


def unify_yaml(df: pl.DataFrame) -> pl.DataFrame:
    df = _ensure_source_row(df)
    return _finish(df, "yaml", {
        "flight_code": _flight(_c(df, "flight")),
        "date": _date(df, "date"),
        "dep_airport": _iata(_c(df, "from")),
        "arr_airport": _iata(_c(df, "to")),
        "card_program": _program(_c(df, "loyalty_program")),
        "card_number": _card_number(_c(df, "loyalty_number")),
        "travel_class": _norm(_c(df, "class")),
        "fare": _norm(_c(df, "fare")),
    })


def unify_xml(df: pl.DataFrame, program_codes: dict[str, str] | None = None) -> pl.DataFrame:
    df = _ensure_source_row(df)
    raw_name = pl.concat_str([
        _c(df, "last_name").fill_null(""),
        pl.lit(" "),
        _c(df, "first_name").fill_null(""),
    ])
    df = _with_name_key(df, raw_name)
    program = _program(_c(df, "bonus_program"))
    if program_codes:
        program = program.replace_strict(program_codes, default=program, return_dtype=pl.String)
    return _finish(df, "xml", {
        "flight_code": _flight(_c(df, "flight_code")),
        "date": _date(df, "flight_date"),
        "dep_airport": _iata(_c(df, "departure")),
        "arr_airport": _iata(_c(df, "arrival")),
        "name_key": pl.col("_name_key"),
        "card_program": program,
        "card_number": _card_number(_c(df, "card_number")),
        "fare": _norm(_c(df, "fare")),
    })


def _iata_dyn(expr: pl.Expr, city_map: dict[str, str] | None) -> pl.Expr:
    value = _nonempty(expr).str.to_uppercase()
    if not city_map:
        return value
    normalized_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in city_map.items()}
    return value.replace_strict(normalized_map, default=value, return_dtype=pl.String)


def unify_csv(df: pl.DataFrame | pd.DataFrame, city_map: dict[str, str] | None = None) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    df = _ensure_source_row(df)
    name = pl.concat_str([
        _c(df, "last_name").fill_null(""),
        pl.lit(" "),
        _c(df, "first_name").fill_null(""),
    ])
    df = _with_name_key(df, name)
    dep_raw = _first_existing(df, (
        "departure", "origin", "origin_airport", "departure_airport", "from", "from_iata"
    ))
    arr_raw = _first_existing(df, (
        "arrival", "destination", "destination_airport", "arrival_airport", "to", "to_iata", "dest"
    ))

    return _finish(df, "csv", {
        "flight_code": _flight(_first_existing(df, ("flight_number", "flight", "flight_no"))),
        "date": _date(df, "flight_date"),
        "dep_airport": _iata_dyn(dep_raw, city_map),
        "arr_airport": _iata_dyn(arr_raw, city_map),
        "dep_ts": _ts_col(df, "flight_datetime"),
        "name_key": pl.col("_name_key"),
        "birth_date": _date(df, "birth_date"),
        "passport": _id(_first_existing(df, ("document", "passport", "passport_number"))),
        "ticket": _ticket(_first_existing(df, ("ticket_number", "ticket", "e_ticket"))),
        "booking_code": _norm(_first_existing(df, ("booking_code", "pnr", "record_locator"))),
    })


def _split_list(expr: pl.Expr, separator: str) -> pl.Expr:
    return _norm(expr).str.split(separator)


def unify_boarding(df: pl.DataFrame) -> pl.DataFrame:
    df = _ensure_source_row(df)
    if "passenger_name_raw" in df.columns:
        raw_name = _c(df, "passenger_name_raw")
    else:
        raw_name = pl.concat_str([
            _c(df, "first_name").fill_null(""),
            pl.lit(" "),
            _c(df, "last_name").fill_null(""),
        ])
    df = _with_name_key(df, raw_name)

    frequent_flyer = _norm(_c(df, "ff_raw")).str.to_uppercase()
    program = _program(frequent_flyer.str.extract(r"^([A-Z0-9]{2})[\s\-]*\d+$", 1))
    number = frequent_flyer.str.extract(r"^[A-Z0-9]{2}[\s\-]*(\d+)$", 1)
    return _finish(df, "boarding", {
        "flight_code": _flight(_first_existing(df, ("flight_no", "flight_number", "flight"))),
        "date": _date(df, "flight_date"),
        "dep_airport": _iata(_first_existing(df, ("from_iata", "departure_airport", "origin"))),
        "arr_airport": _iata(_first_existing(df, ("to_iata", "arrival_airport", "destination"))),
        "dep_ts": _ts_col(df, "departure_dt"),
        "name_key": pl.col("_name_key"),
        "ticket": _ticket(_first_existing(df, ("ticket_no", "ticket_number", "ticket"))),
        "booking_code": _norm(_first_existing(df, ("pnr", "booking_code"))),
        "card_program": program,
        "card_number": _card_number(number),
        "travel_class": _norm(_first_existing(df, ("travel_class", "class"))),
    })


def unify_json(df: pl.DataFrame) -> pl.DataFrame:
    df = _ensure_source_row(df)
    raw_name = pl.concat_str([
        _c(df, "last_name").fill_null(""),
        pl.lit(" "),
        _c(df, "first_name").fill_null(""),
    ])
    df = _with_name_key(df, raw_name)

    cards = _split_list(_c(df, "loyalty_programs"), " | ").alias("_cards")
    passports = _split_list(_c(df, "passports"), ", ").alias("_passports")
    base = df.with_columns(cards, passports).select(
        "_source_row",
        _flight(_c(df, "flight_id")).alias("flight_code"),
        _date(df, "date").alias("date"),
        _iata(_c(df, "departure_airport")).alias("dep_airport"),
        _iata(_c(df, "arrival_airport")).alias("arr_airport"),
        pl.col("_name_key").alias("name_key"),
        _date(df, "birth_date").alias("birth_date"),
        _ticket(_c(df, "ticket_number")).alias("ticket"),
        _norm(_c(df, "booking_code")).alias("booking_code"),
        _norm(_c(df, "nickname")).alias("nickname"),
        pl.col("_cards"),
        pl.col("_passports"),
    ).with_columns(
        pl.col("_cards").list.eval(pl.element().str.strip_chars()).alias("_cards"),
        pl.col("_passports").list.eval(pl.element().str.strip_chars()).alias("_passports"),
    )

    base = base.with_columns(
        pl.col("_cards").list.eval(
            pl.when(pl.element().str.strip_chars() != "")
            .then(pl.element().str.strip_chars())
            .otherwise(pl.lit(None, dtype=pl.String))
        ).list.drop_nulls().alias("_cards"),
        pl.col("_passports").list.eval(
            pl.when(pl.element().str.strip_chars() != "")
            .then(pl.element().str.strip_chars())
            .otherwise(pl.lit(None, dtype=pl.String))
        ).list.drop_nulls().alias("_passports"),
    ).with_columns(
        _has_cards=pl.col("_cards").list.len().fill_null(0) > 0,
        _has_passports=pl.col("_passports").list.len().fill_null(0) > 0,
    )

    common = ["_source_row", "flight_code", "date", "dep_airport", "arr_airport",
              "name_key", "birth_date", "ticket", "booking_code", "nickname"]
    card_rows = base.filter(pl.col("_has_cards")).select(*common, "_cards").explode("_cards").filter(
        pl.col("_cards").is_not_null() & (pl.col("_cards") != "")
    ).with_columns(
        card_program=_program(pl.col("_cards").str.extract(r"^\s*([A-Za-z0-9]+)[:\s]", 1)),
        card_number=_card_number(pl.col("_cards").str.extract(r"(\d+)\s*$", 1)),
    )

    passport_rows = base.filter(pl.col("_has_passports")).select(
        "_source_row", pl.col("_passports").alias("_passport_item")
    ).explode("_passport_item").filter(
        pl.col("_passport_item").is_not_null() & (pl.col("_passport_item") != "")
    ).with_columns(passport=_id(pl.col("_passport_item"))).select("_source_row", "passport")

    both_ids = card_rows.join(
        passport_rows, on="_source_row", how="inner"
    ).select(*common, "passport", "card_program", "card_number")

    card_only = (
        base.filter(pl.col("_has_cards") & ~pl.col("_has_passports"))
        .select(*common, "_cards").explode("_cards")
        .with_columns(
            card_program=_program(pl.col("_cards").str.extract(r"^\s*([A-Za-z0-9]+)[:\s]", 1)),
            card_number=_card_number(pl.col("_cards").str.extract(r"(\d+)\s*$", 1)),
            passport=pl.lit(None, dtype=pl.String),
        )
        .select(*common, "passport", "card_program", "card_number")
    )

    passport_only = (
        base.filter(pl.col("_has_passports") & ~pl.col("_has_cards"))
        .select(*common, "_passports").explode("_passports")
        .with_columns(
            passport=_id(pl.col("_passports")),
            card_program=pl.lit(None, dtype=pl.String),
            card_number=pl.lit(None, dtype=pl.String),
        )
        .select(*common, "passport", "card_program", "card_number")
    )

    no_ids = (
        base.filter(~pl.col("_has_cards") & ~pl.col("_has_passports"))
        .with_columns(
            passport=pl.lit(None, dtype=pl.String),
            card_program=pl.lit(None, dtype=pl.String),
            card_number=pl.lit(None, dtype=pl.String),
        )
        .select(*common, "passport", "card_program", "card_number")
    )

    rows = pl.concat([both_ids, card_only, passport_only, no_ids], how="diagonal_relaxed")
    return _finish(rows, "json", {
        column: pl.col(column)
        for column in [
            "flight_code", "date", "dep_airport", "arr_airport", "name_key", "birth_date",
            "passport", "ticket", "booking_code", "card_program", "card_number", "nickname",
            "travel_class", "fare", "agent",
        ]
        if column in rows.columns
    })


def airports_from_schedule(path: str) -> pl.DataFrame:
    schedule = pl.read_parquet(path)
    needed = {"orig_iata", "orig_city", "orig_country", "dest_iata", "dest_city", "dest_country"}
    missing = needed - set(schedule.columns)
    if missing:
        raise ValueError(f"В расписании {path!r} отсутствуют колонки: {sorted(missing)}")
    return pl.concat([
        schedule.select(
            pl.col("orig_iata").alias("airport"),
            pl.col("orig_city").alias("city"),
            pl.col("orig_country").alias("country"),
        ),
        schedule.select(
            pl.col("dest_iata").alias("airport"),
            pl.col("dest_city").alias("city"),
            pl.col("dest_country").alias("country"),
        ),
    ]).drop_nulls("airport").unique(subset="airport", keep="first")
