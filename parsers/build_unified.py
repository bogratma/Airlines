import os
import re
from collections import defaultdict
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz
from unidecode import unidecode

from .unify import (
    unify_tab, unify_yaml, unify_json, unify_xml, unify_csv, unify_boarding,
    airports_from_schedule, _flight, _iata, _norm, _card_number, _program,
)

_TRANSLIT_TOKEN_ALIASES = {
    "ALEKSEI": "ALEXEY", "ALEKSEY": "ALEXEY", "ALEXEI": "ALEXEY", "ALEXEY": "ALEXEY",
    "SERGEI": "SERGEY", "SERGEY": "SERGEY",
    "EVGENIY": "EVGENY", "YEVGENIY": "EVGENY", "YEVGENY": "EVGENY", "EVGENY": "EVGENY",
    "NIKOLAI": "NIKOLAY", "NIKOLAY": "NIKOLAY", "NIKOLAJ": "NIKOLAY",
    "DMITRI": "DMITRY", "DMITRIY": "DMITRY", "DMITRY": "DMITRY",
    "NATALYA": "NATALIA", "NATALIYA": "NATALIA", "NATALIA": "NATALIA",
    "ANASTASIYA": "ANASTASIA", "ANASTASIA": "ANASTASIA",
    "KSENIIA": "KSENIA", "KSENIA": "KSENIA", "XENIA": "KSENIA",
    "IULIIA": "YULIA", "IULIA": "YULIA", "YULIYA": "YULIA", "YULIA": "YULIA",
    "MARIIA": "MARIA", "MARIYA": "MARIA", "MARIA": "MARIA",
    "SOFIIA": "SOFIA", "SOFIYA": "SOFIA", "SOFIA": "SOFIA",
}


def _card_owner_name_key(value):
    if value is None:
        return None
    text = unidecode(str(value)).upper()
    tokens = [token for token in re.findall(r"[A-Z0-9]+", text) if len(token) > 1]
    if not tokens:
        return None
    normalized = []
    for token in tokens:
        token = _TRANSLIT_TOKEN_ALIASES.get(token, token)
        if token.startswith("IUL"):
            token = "YUL" + token[3:]
        normalized.append(token)
    return " ".join(sorted(normalized))


def _same_card_owner_name(name_a, name_b, birth_a=None, birth_b=None):
    if birth_a and birth_b and str(birth_a) != str(birth_b):
        return False
    a = _card_owner_name_key(name_a)
    b = _card_owner_name_key(name_b)
    if not a or not b:
        return False
    if a == b:
        return True
    tokens_a, tokens_b = a.split(), b.split()
    if len(tokens_a) != len(tokens_b):
        return False
    return fuzz.token_sort_ratio(a, b) >= 88


def _build_card_owner_summary(df_conf):
    rows = (
        df_conf
        .with_columns(
            _card_number(pl.col("card_number")).alias("_card_number_norm"),
            _program(pl.col("card_program")).alias("_card_program_norm"),
        )
        .filter(
            pl.col("_card_number_norm").is_not_null()
            & pl.col("name_key").is_not_null()
            & (pl.col("name_key").str.strip_chars() != "")
        )
        .select(
            pl.col("_card_number_norm").alias("card_number"),
            pl.col("_card_program_norm").alias("card_program"),
            "name_key", "passport", "birth_date",
        )
        .unique()
    )

    by_card = defaultdict(list)
    for row in rows.iter_rows(named=True):
        by_card[row["card_number"]].append(row)

    summaries = []
    cards_with_translit_variants = 0
    for card_number, profiles in by_card.items():
        programs = sorted({p["card_program"] for p in profiles if p["card_program"]})
        clusters = []
        for profile in sorted(profiles, key=lambda x: (x["name_key"] or "", x["birth_date"] or "")):
            placed = False
            for cluster in clusters:
                if any(
                    _same_card_owner_name(
                        profile["name_key"], existing["name_key"],
                        profile["birth_date"], existing["birth_date"],
                    )
                    for existing in cluster
                ):
                    cluster.append(profile)
                    placed = True
                    break
            if not placed:
                clusters.append([profile])

        raw_names = sorted({p["name_key"] for p in profiles if p["name_key"]})
        representative_names = []
        for cluster in clusters:
            names = sorted({p["name_key"] for p in cluster if p["name_key"]})
            representative_names.append(names[0] if names else "UNKNOWN")
        representative_names = sorted(set(representative_names))
        if len(raw_names) > len(representative_names):
            cards_with_translit_variants += 1

        if len(clusters) <= 1:
            continue

        if len(programs) == 1:
            anomaly_type, severity = "loyalty_card_sharing", 2
        elif len(programs) == 0:
            anomaly_type, severity = "loyalty_card_candidate_missing_program", 1
        else:
            anomaly_type, severity = "loyalty_card_number_collision", 1

        summaries.append({
            "card_number": card_number,
            "unique_owners": len(clusters),
            "owners": representative_names,
            "programs": programs,
            "program_count": len(programs),
            "program_summary": ", ".join(programs) if programs else "не указана",
            "anomaly_type": anomaly_type,
            "card_severity": severity,
            "raw_name_variants": raw_names,
        })

    schema = {
        "card_number": pl.String,
        "unique_owners": pl.Int64,
        "owners": pl.List(pl.String),
        "programs": pl.List(pl.String),
        "program_count": pl.Int64,
        "program_summary": pl.String,
        "anomaly_type": pl.String,
        "card_severity": pl.Int64,
        "raw_name_variants": pl.List(pl.String),
    }
    return pl.DataFrame(summaries, schema=schema, strict=False), cards_with_translit_variants


DEFAULT_PATHS = {
    "tab": "clean_tab.parquet", "yaml": "clean_yaml.parquet", "json": "clean_json.parquet",
    "xml": "clean_xml.parquet", "csv": "clean_csv.parquet", "boarding": "clean_zip.parquet",
}


def _resolve_schedule_path(process_dir="process"):
    candidates = [
        os.path.join(process_dir, "clean_pdf.parquet", "skyteam_schedule.parquet"),
        os.path.join(process_dir, "clean_pdf.parquet"),
        os.path.join(process_dir, "skyteam_schedule.parquet"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        if os.path.isdir(candidate):
            preferred = os.path.join(candidate, "skyteam_schedule.parquet")
            if os.path.isfile(preferred):
                return preferred
            parquet_files = sorted(Path(candidate).glob("*.parquet"))
            if len(parquet_files) == 1:
                return str(parquet_files[0])
    return None


def learn_program_codes(xml_df, code_parts):
    readable_parts = [path for path in code_parts if os.path.isfile(path)]
    if not readable_parts:
        return {}
    cards = (
        pl.concat([pl.scan_parquet(path) for path in readable_parts], how="vertical")
        .select("card_program", "card_number")
        .drop_nulls()
        .unique()
        .collect()
    )
    if cards.is_empty() or not {"bonus_program", "card_number"}.issubset(xml_df.columns):
        return {}
    x = (
        xml_df.select(
            _program(pl.col("bonus_program")).alias("name"),
            _card_number(pl.col("card_number")).alias("card_number"),
        )
        .drop_nulls()
        .unique()
    )
    joined = x.join(cards, on="card_number", how="inner")
    if joined.is_empty():
        return {}
    counts = joined.group_by("name", "card_program").len().sort("len", descending=True)
    best = counts.group_by("name", maintain_order=True).first()
    totals = counts.group_by("name").agg(pl.col("len").sum().alias("total"))
    report = best.join(totals, on="name").with_columns(
        share=(pl.col("len") / pl.col("total")).round(2)
    )
    print("Программы xml -> коды (по совпадению номеров карт):")
    print(report.select("name", "card_program", "len", "share"))
    return dict(zip(best["name"].to_list(), best["card_program"].to_list()))


def build_unified_file(process_dir="process", out_name="unified.parquet", paths=None):
    os.makedirs(process_dir, exist_ok=True)
    paths = {**DEFAULT_PATHS, **(paths or {})}
    full = {
        key: (value if os.path.isabs(value) else os.path.join(process_dir, value))
        for key, value in paths.items()
    }
    mandatory = ("tab", "yaml", "json", "xml", "boarding")
    missing = [name for name in mandatory if not os.path.isfile(full[name])]
    if missing:
        details = ", ".join(f"{name}: {full[name]}" for name in missing)
        raise FileNotFoundError(
            "Не найдены обязательные очищенные Parquet-источники: " + details
        )

    parts = {}

    def write_part(name, frame):
        part_path = os.path.join(process_dir, f"_unified_{name}.parquet")
        frame.write_parquet(part_path)
        parts[name] = part_path
        print(f"{name}: {frame.height:,} строк")

    schedule_path = _resolve_schedule_path(process_dir)
    city_map = {}
    if schedule_path:
        print(f"Загружаем справочник аэропортов: {schedule_path}")
        try:
            airports = airports_from_schedule(schedule_path).drop_nulls("city")
            for city, iata_code in airports.select("city", "airport").iter_rows():
                if city and iata_code:
                    city_map[str(city).strip().upper()] = str(iata_code).strip().upper()
            print(f"Загружено городов в справочник: {len(city_map)}")
        except (ValueError, KeyError, pl.exceptions.ColumnNotFoundError) as exc:
            print(f"Справочник городов недоступен ({exc}); сверка маршрутов всё равно будет выполнена.")
    else:
        print("PDF-расписание не найдено: сверка маршрутов будет пропущена.")

    for name, normalizer in (
        ("tab", unify_tab),
        ("yaml", unify_yaml),
        ("json", unify_json),
        ("boarding", unify_boarding),
    ):
        write_part(name, normalizer(pl.read_parquet(full[name])))

    if os.path.isfile(full.get("csv", "")):
        write_part("csv", unify_csv(pl.read_parquet(full["csv"]), city_map))
    else:
        print(f"CSV-источник отсутствует, пропускаем: {full.get('csv')}")

    xml_df = pl.read_parquet(full["xml"])
    program_codes = learn_program_codes(
        xml_df,
        [parts[name] for name in ("tab", "yaml", "json", "boarding") if name in parts],
    )
    write_part("xml", unify_xml(xml_df, program_codes))
    output_path = out_name if os.path.isabs(out_name) else os.path.join(process_dir, out_name)
    lazy_parts = [pl.scan_parquet(part_path) for part_path in parts.values()]
    pl.concat(lazy_parts, how="vertical").sink_parquet(output_path)
    print(f"Готово: {output_path}")
    if schedule_path:
        print("Запуск сверки с официальным расписанием из PDF...")
        unified_df = pl.read_parquet(output_path)
        enriched_df = enrich_and_validate_with_pdf(unified_df, schedule_path)
        tmp_output = output_path + ".tmp.parquet"
        enriched_df.write_parquet(tmp_output)
        os.replace(tmp_output, output_path)
        mismatch_count = enriched_df.filter(pl.col("route_mismatch")).height
        print(f"Сверка завершена. Строк с расхождением маршрута: {mismatch_count:,}")
    for part_path in parts.values():
        try:
            os.remove(part_path)
        except OSError:
            pass
    return output_path


def enrich_and_validate_with_pdf(unified_df, pdf_schedule_path):
    schedule = pl.read_parquet(pdf_schedule_path)
    needed = {"flight", "orig_iata", "dest_iata"}
    missing = needed - set(schedule.columns)
    if missing:
        raise ValueError(f"В PDF-расписании не хватает колонок: {sorted(missing)}")

    routes = schedule.select(
        _flight(pl.col("flight").cast(pl.String)).alias("flight_code"),
        _iata(pl.col("orig_iata").cast(pl.String)).alias("pdf_dep"),
        _iata(pl.col("dest_iata").cast(pl.String)).alias("pdf_arr"),
    ).drop_nulls(["flight_code", "pdf_dep", "pdf_arr"]).unique()
    known_flights = routes.select("flight_code").unique().with_columns(
        pl.lit(True).alias("_flight_known")
    )

    base = unified_df
    if "route_mismatch" in base.columns:
        base = base.drop("route_mismatch")
    base = base.with_row_index("_pdf_row_id")

    possible = base.select(
        "_pdf_row_id", "flight_code", "dep_airport", "arr_airport"
    ).join(routes, on="flight_code", how="left")
    possible = possible.with_columns(
        _route_match=(
            pl.col("dep_airport").is_not_null()
            & pl.col("arr_airport").is_not_null()
            & (pl.col("dep_airport") == pl.col("pdf_dep"))
            & (pl.col("arr_airport") == pl.col("pdf_arr"))
        )
    )
    route_status = possible.group_by("_pdf_row_id").agg(
        pl.col("_route_match").fill_null(False).any().alias("_route_found")
    )
    result = (
        base.join(route_status, on="_pdf_row_id", how="left")
        .join(known_flights, on="flight_code", how="left")
        .with_columns(
            route_mismatch=(
                pl.col("_flight_known").fill_null(False)
                & pl.col("dep_airport").is_not_null()
                & pl.col("arr_airport").is_not_null()
                & ~pl.col("_route_found").fill_null(False)
            )
        )
        .drop("_pdf_row_id", "_route_found", "_flight_known")
    )
    return result

def export_suspicious_passengers(
    df_with_ids,
    output_prefix="process/suspicious_spies",
    max_flights_24h=4,
    max_flights_7d=10,
    max_flights_30d=30,
):
    print("\n=== ЭТАП 5: Анализ аномалий и экспорт отчета по шпионам ===")
    output_parent = os.path.dirname(os.path.abspath(output_prefix))
    os.makedirs(output_parent, exist_ok=True)
    conflicts = (
        df_with_ids.filter(
            pl.col("person_id").is_not_null() &
            pl.col("flight_code").is_not_null() &
            pl.col("date").is_not_null()
        )
        .group_by(["person_id", "flight_code", "date"])
        .agg(
            dep_conflict=(pl.col("dep_airport").drop_nulls().n_unique() > 1),
            arr_conflict=(pl.col("arr_airport").drop_nulls().n_unique() > 1)
        )
    )

    df_conf = df_with_ids.join(conflicts, on=["person_id", "flight_code", "date"], how="left").with_columns(
        _card_number(pl.col("card_number")).alias("_card_number_norm"),
        _program(pl.col("card_program")).alias("_card_program_norm"),
    )
    anomalies_list = []
    if "route_mismatch" in df_conf.columns:
        pdf_mismatches = df_conf.filter(
            pl.col("route_mismatch") == True
        ).select(
            "person_id", "flight_code", "date",
            pl.lit("route_mismatch").alias("anomaly_type"),
            pl.lit("Маршрут расходится с официальным расписанием PDF").alias("reason"),
            pl.lit(3).alias("severity")
        )
        anomalies_list.append(pdf_mismatches)
    source_conflicts = df_conf.filter(
        pl.col("arr_conflict") | pl.col("dep_conflict")
    ).select(
        "person_id", "flight_code", "date",
        pl.lit("source_route_conflict").alias("anomaly_type"),
        pl.lit("Конфликт данных между источниками (разные аэропорты)").alias("reason"),
        pl.lit(3).alias("severity")
    )
    anomalies_list.append(source_conflicts)
    tickets_with_owners = (
        df_conf.filter(
            pl.col("ticket").is_not_null() &
            (pl.col("ticket").str.strip_chars() != "")
        )
        .group_by("ticket")
        .agg(
            pl.col("name_key").drop_nulls().n_unique().alias("ticket_name_count"),
            pl.col("passport").drop_nulls().n_unique().alias("ticket_passport_count"),
            pl.col("birth_date").drop_nulls().n_unique().alias("ticket_birth_count"),
        )
        .filter(
            (pl.col("ticket_name_count") > 1) |
            (pl.col("ticket_passport_count") > 1) |
            (pl.col("ticket_birth_count") > 1)
        )
    )
    ticket_anomalies = (
        df_conf.filter(
            pl.col("ticket").is_not_null() &
            (pl.col("ticket").str.strip_chars() != "") &
            pl.col("flight_code").is_not_null() &
            pl.col("date").is_not_null()
        )
        .join(tickets_with_owners, on="ticket", how="inner")
        .select(
            "person_id", "flight_code", "date",
            pl.lit("ticket_identity_conflict").alias("anomaly_type"),
            pl.concat_str([
                pl.lit("Один номер билета связан с противоречивыми данными пассажиров (имён: "),
                pl.col("ticket_name_count").cast(pl.String),
                pl.lit(", паспортов: "),
                pl.col("ticket_passport_count").cast(pl.String),
                pl.lit(", дат рождения: "),
                pl.col("ticket_birth_count").cast(pl.String),
                pl.lit(")")
            ]).alias("reason"),
            pl.lit(4).alias("severity")
        )
    )
    print(f"Найдено номеров билетов с противоречивыми данными: {len(tickets_with_owners)}")
    anomalies_list.append(ticket_anomalies)
    passports_with_owners = (
        df_conf.filter(
            pl.col("passport").is_not_null()
            & (pl.col("passport").str.strip_chars() != "")
        )
        .group_by("passport")
        .agg(
            pl.col("name_key").drop_nulls().n_unique().alias("passport_name_count"),
            pl.col("birth_date").drop_nulls().n_unique().alias("passport_birth_count"),
        )
        .filter(
            (pl.col("passport_name_count") > 1)
            | (pl.col("passport_birth_count") > 1)
        )
    )
    passport_identity_anomalies = (
        df_conf.filter(
            pl.col("passport").is_not_null()
            & (pl.col("passport").str.strip_chars() != "")
            & pl.col("flight_code").is_not_null()
            & pl.col("date").is_not_null()
        )
        .join(passports_with_owners, on="passport", how="inner")
        .select(
            "person_id", "flight_code", "date",
            pl.lit("passport_identity_conflict").alias("anomaly_type"),
            pl.concat_str([
                pl.lit("Один паспорт связан с противоречивыми данными (имён: "),
                pl.col("passport_name_count").cast(pl.String),
                pl.lit(", дат рождения: "),
                pl.col("passport_birth_count").cast(pl.String),
                pl.lit(")"),
            ]).alias("reason"),
            pl.lit(4).alias("severity"),
        )
        .unique()
    )
    print(f"Найдено паспортов с противоречивыми данными: {len(passports_with_owners)}")
    anomalies_list.append(passport_identity_anomalies)
    cards_with_owners, translit_variant_cards = _build_card_owner_summary(df_conf)
    print(
        "Номеров карт с несколькими различимыми владельцами: "
        f"{len(cards_with_owners)}; карт с вариантами транслитерации, сведёнными в один профиль: "
        f"{translit_variant_cards}"
    )
    if not cards_with_owners.is_empty():
        print("Кандидаты sharing/коллизий карт:")
        with pl.Config(tbl_rows=20, tbl_cols=-1, tbl_width_chars=220):
            print(cards_with_owners.select(
                "card_number", "owners", "programs", "unique_owners", "anomaly_type", "raw_name_variants"
            ).head(20))

    card_sharing = (
        df_conf.join(
            cards_with_owners,
            left_on="_card_number_norm",
            right_on="card_number",
            how="inner",
        )
        .filter(pl.col("name_key").is_not_null())
        .select(
            "person_id", "flight_code", "date",
            "anomaly_type",
            pl.concat_str([
                pl.when(pl.col("anomaly_type") == "loyalty_card_sharing")
                .then(pl.lit("Шеринг карты лояльности (программа: "))
                .when(pl.col("anomaly_type") == "loyalty_card_candidate_missing_program")
                .then(pl.lit("Возможный шеринг карты; программа не указана (номер: "))
                .otherwise(pl.lit("Коллизия номера карты между программами (программы: ")),
                pl.col("program_summary"),
                pl.lit("). Владельцы: ["),
                pl.col("owners").list.join(", "),
                pl.lit("]"),
            ]).alias("reason"),
            pl.col("card_severity").alias("severity"),
        )
        .unique()
    )
    anomalies_list.append(card_sharing)
    multi_passports = (
        df_conf.filter(pl.col("passport").is_not_null())
        .group_by("person_id")
        .agg(pl.col("passport").n_unique().alias("pass_count"))
        .filter(pl.col("pass_count") > 1)
    )
    passport_anomalies = (
        df_conf.join(multi_passports, on="person_id", how="inner")
        .select(
            "person_id", "flight_code", "date",
            pl.lit("multiple_passports").alias("anomaly_type"),
            pl.concat_str([
                pl.lit("Использование нескольких паспортов (всего: "),
                pl.col("pass_count").cast(pl.String),
                pl.lit(")")
            ]).alias("reason"),
            pl.when(pl.col("pass_count") >= 5).then(5)
              .when(pl.col("pass_count") >= 3).then(4)
              .otherwise(3).cast(pl.Int64).alias("severity")
        )
    )
    anomalies_list.append(passport_anomalies)
    timing_coverage = df_conf.group_by("source").agg(
        rows=pl.len(),
        rows_with_dep_ts=pl.col("dep_ts").is_not_null().sum(),
        rows_with_arr_ts=pl.col("arr_ts").is_not_null().sum(),
        complete_flights=(
            pl.col("person_id").is_not_null()
            & pl.col("flight_code").is_not_null()
            & pl.col("date").is_not_null()
            & pl.col("dep_airport").is_not_null()
            & pl.col("arr_airport").is_not_null()
            & pl.col("dep_ts").is_not_null()
            & pl.col("arr_ts").is_not_null()
        ).sum(),
    ).sort("source")
    print("\nЗаполненность данных для проверки перекрывающихся рейсов:")
    with pl.Config(tbl_rows=20, tbl_cols=-1, tbl_width_chars=200):
        print(timing_coverage)
    unique_flights_for_teleport = (
        df_conf.filter(
            pl.col("person_id").is_not_null()
            & pl.col("flight_code").is_not_null()
            & pl.col("date").is_not_null()
            & pl.col("dep_airport").is_not_null()
            & pl.col("arr_airport").is_not_null()
            & pl.col("dep_ts").is_not_null()
            & pl.col("arr_ts").is_not_null()
        )
        .unique(subset=["person_id", "flight_code", "date", "dep_ts", "arr_ts"])
        .sort(["person_id", "dep_ts", "arr_ts"])
    )
    eligible_people = (
        unique_flights_for_teleport.group_by("person_id")
        .agg(eligible_flights=pl.len())
        .filter(pl.col("eligible_flights") > 1)
    )
    print(
        f"Уникальных рейсов с полными временами и аэропортами: {len(unique_flights_for_teleport)}; "
        f"профилей с минимум двумя такими рейсами: {len(eligible_people)}"
    )
    temporal_candidates = (
        unique_flights_for_teleport
        .with_columns(
            _max_arrival_so_far=pl.col("arr_ts").cum_max().over("person_id"),
            prev_arr_ts=pl.col("arr_ts").shift(1).over("person_id"),
            prev_arr_airport=pl.col("arr_airport").shift(1).over("person_id"),
        )
        .with_columns(
            previous_max_arr_ts=pl.col("_max_arrival_so_far").shift(1).over("person_id")
        )
        .with_columns(
            _interval_overlap=(
                pl.col("previous_max_arr_ts").is_not_null()
                & (pl.col("dep_ts") < pl.col("previous_max_arr_ts"))
            ),
            _short_airport_transfer=(
                pl.col("prev_arr_ts").is_not_null()
                & (pl.col("prev_arr_airport") != pl.col("dep_airport"))
                & (pl.col("dep_ts") >= pl.col("prev_arr_ts"))
                & (pl.col("dep_ts") <= pl.col("prev_arr_ts") + pl.duration(minutes=45))
            ),
        )
    )
    teleport_anomalies = (
        temporal_candidates
        .filter(pl.col("_interval_overlap") | pl.col("_short_airport_transfer"))
        .select(
            "person_id", "flight_code", "date",
            pl.lit("impossible_connection").alias("anomaly_type"),
            pl.when(pl.col("_interval_overlap"))
            .then(pl.lit("Пересечение интервалов рейсов: вылет до завершения предыдущего рейса"))
            .otherwise(pl.lit("Слишком короткая пересадка между разными аэропортами (не более 45 минут)"))
            .alias("reason"),
            pl.lit(5).alias("severity"),
        )
        .unique()
    )
    print(f"Найдено перекрытий рейсов/подозрительно коротких пересадок: {len(teleport_anomalies)}")
    if len(teleport_anomalies) == 0:
        print(
            "Если ожидаешь пересечения, проверь выше: времена прибытия могут отсутствовать, "
            "или рейсы одного человека могут оказаться в разных person_id."
        )
    anomalies_list.append(teleport_anomalies)
    frequency_flights = (
        df_conf.filter(
            pl.col("person_id").is_not_null() &
            pl.col("flight_code").is_not_null() &
            pl.col("date").is_not_null() &
            pl.col("dep_ts").is_not_null()
        )
        .group_by(["person_id", "flight_code", "date"])
        .agg(pl.col("dep_ts").drop_nulls().min().alias("dep_ts"))
        .filter(pl.col("dep_ts").is_not_null())
        .sort(["person_id", "dep_ts"])
    )

    if not frequency_flights.is_empty():
        flights_per_timestamp = (
            frequency_flights
            .group_by(["person_id", "dep_ts"])
            .agg(pl.len().alias("flights_at_ts"))
            .sort(["person_id", "dep_ts"])
        )

        frequency_windows = [
            ("24h", "24 часа", max_flights_24h),
            ("7d", "7 дней", max_flights_7d),
            ("30d", "30 дней", max_flights_30d),
        ]

        for suffix, label, threshold in frequency_windows:
            count_col = f"flights_{suffix}"
            rolling_counts = (
                flights_per_timestamp
                .rolling(
                    index_column="dep_ts",
                    period=suffix,
                    group_by="person_id",
                    closed="right",
                )
                .agg(pl.col("flights_at_ts").sum().alias(count_col))
            )
            flights_per_timestamp = flights_per_timestamp.join(
                rolling_counts.select("person_id", "dep_ts", count_col),
                on=["person_id", "dep_ts"],
                how="left",
            )
        frequency_flights = frequency_flights.join(
            flights_per_timestamp.select(
                "person_id", "dep_ts", "flights_24h", "flights_7d", "flights_30d"
            ),
            on=["person_id", "dep_ts"],
            how="left",
        )

        for suffix, label, threshold in frequency_windows:
            count_col = f"flights_{suffix}"
            flagged_people = (
                frequency_flights
                .group_by("person_id")
                .agg(pl.col(count_col).max().alias("max_flights_in_window"))
                .filter(pl.col("max_flights_in_window") > threshold)
            )

            frequency_anomalies = (
                df_conf.filter(
                    pl.col("person_id").is_not_null() &
                    pl.col("flight_code").is_not_null() &
                    pl.col("date").is_not_null()
                )
                .join(flagged_people, on="person_id", how="inner")
                .select(
                    "person_id", "flight_code", "date",
                    pl.lit(f"high_flight_frequency_{suffix}").alias("anomaly_type"),
                    pl.concat_str([
                        pl.lit(f"Необычно высокая частота перелётов ({label}: максимум "),
                        pl.col("max_flights_in_window").cast(pl.String),
                        pl.lit(f" рейсов; порог: {threshold})")
                    ]).alias("reason"),
                    pl.lit(2).alias("severity")
                )
                .unique()
            )
            print(
                f"Пассажиров с частотой выше порога за {label}: "
                f"{len(flagged_people)} (порог: {threshold}); "
                f"отмечено событий: {len(frequency_anomalies)}"
            )
            anomalies_list.append(frequency_anomalies)
    anomalies_list = [
        anomaly.with_columns(pl.col("severity").cast(pl.Int64))
        for anomaly in anomalies_list
    ]
    raw_anomalies = pl.concat(anomalies_list, how="diagonal").unique()
    aggregated_reasons = (
        raw_anomalies
        .group_by(["person_id", "flight_code", "date"])
        .agg(
            pl.col("reason").unique().str.join(" | ").alias("all_suspicion_reasons"),
            pl.col("reason").n_unique().alias("total_reasons_count"),
            pl.col("severity").max().alias("event_max_severity")
        )
    )
    type_scores = (
        raw_anomalies
        .filter(pl.col("person_id").is_not_null())
        .group_by(["person_id", "anomaly_type"])
        .agg(pl.col("severity").max().alias("type_score"))
    )
    person_base_scores = (
        type_scores
        .group_by("person_id")
        .agg(
            pl.col("type_score").sum().alias("base_suspicion_score"),
            pl.col("anomaly_type").n_unique().alias("anomaly_types_count")
        )
    )
    repeatable_types = [
        "route_mismatch",
        "source_route_conflict",
        "impossible_connection",
        "high_flight_frequency_24h",
        "high_flight_frequency_7d",
        "high_flight_frequency_30d",
    ]
    repeated_events = (
        raw_anomalies
        .filter(
            pl.col("person_id").is_not_null() &
            pl.col("anomaly_type").is_in(repeatable_types)
        )
        .unique(subset=["person_id", "anomaly_type", "flight_code", "date"])
        .group_by(["person_id", "anomaly_type"])
        .agg(pl.len().alias("type_event_count"))
        .with_columns(
            repeat_points=pl.when(pl.col("type_event_count") > 1)
            .then(pl.col("type_event_count") - 1)
            .otherwise(0)
        )
        .group_by("person_id")
        .agg(pl.col("repeat_points").sum().alias("repeat_bonus"))
    )

    person_scores = (
        person_base_scores
        .join(repeated_events, on="person_id", how="left")
        .with_columns(pl.col("repeat_bonus").fill_null(0))
        .with_columns(
            pl.when(pl.col("repeat_bonus") > 3)
            .then(3)
            .otherwise(pl.col("repeat_bonus"))
            .cast(pl.Int64)
            .alias("repeat_bonus")
        )
        .with_columns(
            (pl.col("base_suspicion_score") + pl.col("repeat_bonus"))
            .alias("max_severity")
        )
    )

    person_profile = (
        df_conf.filter(pl.col("person_id").is_not_null())
        .group_by("person_id")
        .agg(
            pl.col("name_key").drop_nulls().unique().sort().str.join(" | ").alias("name_key"),
            pl.col("name_key").drop_nulls().n_unique().alias("name_variants_count"),
            pl.col("passport").drop_nulls().unique().sort().str.join(" | ").alias("passport"),
            pl.col("passport").drop_nulls().n_unique().alias("passport_count"),
        )
    )
    person_reasons = (
        raw_anomalies
        .filter(pl.col("person_id").is_not_null())
        .group_by("person_id")
        .agg(pl.col("reason").unique().sort().str.join(" | ").alias("all_suspicion_reasons"))
    )
    person_flights = (
        df_conf.filter(
            pl.col("person_id").is_not_null() &
            pl.col("flight_code").is_not_null() &
            pl.col("date").is_not_null()
        )
        .unique(subset=["person_id", "flight_code", "date"])
        .group_by("person_id")
        .agg(pl.len().alias("unique_flights_count"))
    )

    flights_info = (
        df_conf.select([
            "person_id", "flight_code", "date",
            "dep_ts", "arr_ts",
            "card_program", "card_number", "dep_airport", "arr_airport", "ticket"
        ])
        .unique(subset=["person_id", "flight_code", "date"])
        .with_columns(
            dep_time_readable=pl.col("dep_ts").dt.to_string("%Y-%m-%d %H:%M"),
            arr_time_readable=pl.col("arr_ts").dt.to_string("%Y-%m-%d %H:%M")
        )
    )

    final_report = (
        aggregated_reasons
        .join(person_scores, on="person_id", how="left")
        .join(person_profile, on="person_id", how="left")
        .join(
            flights_info,
            on=["person_id", "flight_code", "date"],
            how="left"
        )
        .sort(
            ["max_severity", "event_max_severity", "total_reasons_count", "person_id"],
            descending=[True, True, True, False],
        )
    )
    parquet_path = f"{output_prefix}.parquet"
    final_report.write_parquet(parquet_path)
    print(f" Успешно сохранено в Parquet: {parquet_path}")
    try:
        final_report.write_excel(f"{output_prefix}.xlsx")
        print(f" Успешно сохранено в Excel: {output_prefix}.xlsx")
    except Exception as exc:
        print(f" Не удалось сохранить Excel-отчёт по рейсам: {exc}")
    suspicious_people = (
        person_scores
        .filter(pl.col("max_severity") > 7)
        .join(person_profile, on="person_id", how="left")
        .join(person_reasons, on="person_id", how="left")
        .join(person_flights, on="person_id", how="left")
        .sort(["max_severity", "anomaly_types_count", "unique_flights_count"],
              descending=[True, True, True])
    )
    high_risk_prefix = f"{output_prefix}_high_risk"
    suspicious_people.write_parquet(f"{high_risk_prefix}.parquet")
    try:
        suspicious_people.write_excel(f"{high_risk_prefix}.xlsx")
    except Exception as exc:
        print(f"Не удалось сохранить Excel high-risk отчёт: {exc}")
    print(
        f"Пассажиров с max_severity > 7: {len(suspicious_people)}. "
        f"Сводные отчёты: {high_risk_prefix}.parquet / {high_risk_prefix}.xlsx"
    )

    return final_report
def bridge_report(u, pid, top=15):
    p = u.filter(pl.col("person_id").cast(pl.String) == str(pid))
    if p.is_empty():
        print(f"person_id={pid!r} не найден. Возможно, отчёт построен до пересборки базы.")
        return

    print(f"\n=== Мосты идентификации для person_id={pid} ===")
    for column in ("ticket", "passport"):
        if column not in u.columns:
            continue
        values = p.get_column(column).drop_nulls().unique().to_list()
        if not values:
            print(f"\n{column}: у профиля нет значений")
            continue
        matches = u.filter(pl.col(column).is_in(values) & pl.col(column).is_not_null())
        view = (
            matches.group_by(column)
            .agg(
                names=pl.col("name_key").drop_nulls().unique().sort(),
                person_ids=pl.col("person_id").drop_nulls().unique().sort(),
                sources=pl.col("source").drop_nulls().unique().sort(),
                rows=pl.len(),
            )
            .with_columns(
                n_names=pl.col("names").list.len(),
                n_person_ids=pl.col("person_ids").list.len(),
            )
            .filter((pl.col("n_names") > 1) | (pl.col("n_person_ids") > 1))
            .sort(["n_names", "n_person_ids", "rows"], descending=True)
            .head(top)
        )
        print(f"\n=== {column}: найденные конфликты по всей таблице ===")
        print(view)
    if {"name_key", "birth_date"}.issubset(u.columns):
        pairs = p.select("name_key", "birth_date").drop_nulls().unique()
        if not pairs.is_empty():
            matches = u.join(pairs, on=["name_key", "birth_date"], how="inner")
            print("\n=== Точные пары name_key + birth_date в других профилях ===")
            print(
                matches.group_by("name_key", "birth_date")
                .agg(
                    person_ids=pl.col("person_id").drop_nulls().unique().sort(),
                    sources=pl.col("source").drop_nulls().unique().sort(),
                    rows=pl.len(),
                )
                .with_columns(n_person_ids=pl.col("person_ids").list.len())
                .filter(pl.col("n_person_ids") > 1)
                .sort("n_person_ids", descending=True)
                .head(top)
            )


def inspect_name(u, name, top=20):
    b = u.filter(pl.col("name_key") == name)
    print(f"Строк: {len(b)}, person_id: {b['person_id'].n_unique()}, "
          f"дат рождения: {b['birth_date'].n_unique()}")
    print(
        b.group_by("person_id")
         .agg(rows=pl.len(),
              births=pl.col("birth_date").drop_nulls().unique().sort(),
              passports=pl.col("passport").drop_nulls().unique().sort(),
              tickets=pl.col("ticket").drop_nulls().n_unique(),
              sources=pl.col("source").unique().sort())
         .sort("rows", descending=True).head(top)
    )
    print(
        b.drop_nulls("passport")
         .group_by("source")
         .agg(n=pl.col("passport").n_unique(),
              min_len=pl.col("passport").str.len_chars().min(),
              max_len=pl.col("passport").str.len_chars().max())
    )

def rank_suspects(report_parquet, out_prefix, top=30):
    rep = pl.read_parquet(report_parquet)
    ty = (rep.select("person_id", pl.col("all_suspicion_reasons").str.split(" | ").alias("r"))
             .explode("r")
             .with_columns(t=pl.col("r").str.extract(r"^([^(.\[]+)", 1).str.strip_chars())
             .select("person_id", "t").drop_nulls().unique())

    n_total = max(rep["person_id"].n_unique(), 1)
    freq = (ty.group_by("t").agg(n=pl.col("person_id").n_unique())
              .with_columns(w=(pl.lit(float(n_total)) / pl.col("n").clip(lower_bound=1)).log())
              .sort("n"))
    print("\nКак часто встречается каждый тип аномалии (n = людей):")
    print(freq)
    prof = rep.group_by("person_id").agg(
        pl.col("max_severity").first(),
        pl.col("name_key").first(),
        pl.col("passport").first(),
        pl.len().alias("events"),
    )
    rank = (ty.join(freq, on="t")
              .group_by("person_id")
              .agg(rare_score=pl.col("w").sum(),
                   types=pl.col("t").sort().str.join(" | "))
              .join(prof, on="person_id")
              .sort(["rare_score", "max_severity"], descending=True))

    rank.write_parquet(f"{out_prefix}_ranked.parquet")
    try:
        rank.write_excel(f"{out_prefix}_ranked.xlsx")
    except Exception as exc:
        print("Excel не сохранён:", exc)
    with pl.Config(tbl_rows=top, tbl_cols=-1, tbl_width_chars=250):
        print(f"\nТоп-{top} по редкости аномалий:")
        print(rank.head(top))
    return rank
