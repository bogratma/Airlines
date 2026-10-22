from __future__ import annotations

import os

import polars as pl

from parsers.build_unified import (
    build_unified_file,
    export_suspicious_passengers,
    bridge_report,
    rank_suspects,
)
from parsers.merge import assign_person_id, merge_events
from parsers.parse_boarding_passes import parse_boardingPass_zip
from parsers.parse_csv import parse_csv
from parsers.parse_json import parse_json
from parsers.parse_pdf import parse_pdf
from parsers.parse_tab import parse_tab
from parsers.parse_xml import parse_xml
from parsers.parse_yaml import parse_yaml
from parsers.reports import plot_suspicious_report

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_DIR = os.path.join(PROJECT_DIR, "process")

JSON_PATH = r"C:\Users\Vladislav\Desktop\Airlines\FrequentFlyerForum-Profiles.json"
TAB_PATH = r"C:\Users\Vladislav\Desktop\Airlines\Sirena-export-fixed.tab"
YAML_PATH = r"C:\Users\Vladislav\Desktop\Airlines\SkyTeam-Exchange.yaml"
XML_PATH = r"C:\Users\Vladislav\Desktop\Airlines\PointzAggregator-AirlinesData.xml"
CSV_PATH = r"C:\Users\Vladislav\Desktop\Airlines\BoardingData.csv"
PDF_PATH = r"C:\Users\Vladislav\Desktop\Airlines\Skyteam_Timetable.pdf"
ZIP_PATH = r"C:\Users\Vladislav\Desktop\Airlines\YourBoardingPassDotAero.zip"

CLEAN_JSON_PATH = os.path.join(PROCESS_DIR, "clean_json.parquet")
CLEAN_TAB_PATH = os.path.join(PROCESS_DIR, "clean_tab.parquet")
CLEAN_YAML_PATH = os.path.join(PROCESS_DIR, "clean_yaml.parquet")
CLEAN_XML_PATH = os.path.join(PROCESS_DIR, "clean_xml.parquet")
CLEAN_CSV_PATH = os.path.join(PROCESS_DIR, "clean_csv.parquet")
CLEAN_PDF_PATH = os.path.join(PROCESS_DIR, "clean_pdf.parquet")
CLEAN_ZIP_PATH = os.path.join(PROCESS_DIR, "clean_zip.parquet")

INSPECT_PERSON_ID = None


def main() -> None:
    os.makedirs(PROCESS_DIR, exist_ok=True)

    parse_json(JSON_PATH, CLEAN_JSON_PATH)
    parse_tab(TAB_PATH, CLEAN_TAB_PATH)
    parse_yaml(YAML_PATH, CLEAN_YAML_PATH)
    parse_xml(XML_PATH, CLEAN_XML_PATH)
    parse_csv(CSV_PATH, CLEAN_CSV_PATH)
    parse_pdf(PDF_PATH, CLEAN_PDF_PATH)
    parse_boardingPass_zip(ZIP_PATH, CLEAN_ZIP_PATH)

    print("\n=== ЭТАП 2: Унификация и сборка общей базы ===")
    unified_raw_path = build_unified_file(
        process_dir=PROCESS_DIR,
        out_name="unified_raw.parquet",
    )
    unified_df = pl.read_parquet(unified_raw_path)

    bad = unified_df.filter(
        pl.col("route_mismatch").fill_null(False)
    )

    print("\n=== ДИАГНОСТИКА РАСХОЖДЕНИЙ С PDF ===")
    print(f"Всего строк в базе: {unified_df.height:,}")
    print(f"Строк с расхождением: {bad.height:,}")

    print("\nРасхождения по источникам:")
    print(
        bad.group_by("source")
        .agg(
            pl.len().alias("mismatch_rows"),
            pl.col("flight_code").n_unique().alias("unique_flights"),
        )
        .sort("mismatch_rows", descending=True)
    )

    print("\nУникальные сочетания рейс + дата + маршрут:")
    print(
        bad.select(
            "flight_code", "date", "dep_airport", "arr_airport"
        ).unique().height
    )

    print("\nПримеры расхождений:")
    with pl.Config(tbl_rows=30, tbl_cols=-1, tbl_width_chars=200):
        print(
            bad.select(
                "source",
                "flight_code",
                "date",
                "dep_airport",
                "arr_airport",
            ).head(30)
        )

    print("\nМаршруты с аэропортами, не похожими на IATA-коды:")
    invalid_airports = bad.filter(
        ~pl.col("dep_airport").str.contains(r"^[A-Z]{3}$")
        | ~pl.col("arr_airport").str.contains(r"^[A-Z]{3}$")
    )

    print(f"Таких строк: {invalid_airports.height:,}")
    print(
        invalid_airports.select(
            "source", "flight_code", "dep_airport", "arr_airport"
        ).head(30)
    )
    print("\n=== ЭТАП 3: Идентификация пассажиров ===")
    df_raw = pl.read_parquet(unified_raw_path)
    df_with_ids = assign_person_id(df_raw)
    id_path = os.path.join(PROCESS_DIR, "unified_with_ids.parquet")
    df_with_ids.write_parquet(id_path)
    print(f"Таблица с person_id сохранена: {id_path}")
    print(f"Строк: {df_with_ids.height:,}; профилей: {df_with_ids['person_id'].n_unique():,}")

    if INSPECT_PERSON_ID is not None:
        print("\n=== Диагностика связей идентификаторов ===")
        bridge_report(df_with_ids, INSPECT_PERSON_ID)

    print("\n=== ЭТАП 4: Слияние событий в мастер-таблицу ===")
    final_master = merge_events(df_with_ids)
    master_path = os.path.join(PROCESS_DIR, "unified_master.parquet")
    final_master.write_parquet(master_path)
    print(f"Мастер-таблица сохранена: {master_path}")
    print(f"Событий в unified_master: {final_master.height:,}")

    if not final_master.is_empty():
        conflict_rows = final_master.filter(
            pl.col("dep_conflict")
            | pl.col("arr_conflict")
            | pl.col("name_conflict")
            | pl.col("passport_conflict")
            | pl.col("ticket_conflict")
        )
        print(f"Событий с конфликтами данных: {conflict_rows.height:,}")
        if not conflict_rows.is_empty():
            print("Примеры конфликтов:")
            with pl.Config(tbl_rows=10, tbl_cols=-1, tbl_width_chars=220):
                print(conflict_rows.head(10))

    print("\n=== ЭТАП 5: Поиск аномалий и формирование отчётов ===")
    report_base = os.path.join(PROCESS_DIR, "suspicious_passengers_report")
    report = export_suspicious_passengers(
        df_with_ids,
        output_prefix=report_base,
        max_flights_24h=4,
        max_flights_7d=10,
        max_flights_30d=30,
    )

    try:
        plot_suspicious_report(
            report_base + ".parquet",
            out_dir=os.path.join(PROCESS_DIR, "charts"),
            top_n=15,
        )
    except Exception as exc:
        print(f"Построение графиков пропущено из-за ошибки: {exc}")

    print("\n=== ЭТАП 6: Дополнительное ранжирование по редкости аномалий ===")
    try:
        rank_suspects(report_base + ".parquet", report_base, top=30)
    except Exception as exc:
        print(f"Дополнительное ранжирование не выполнено: {exc}")
    print("\nГотово. Основные отчёты:")
    print(f"- {report_base}.xlsx")
    print(f"- {report_base}_high_risk.xlsx")
    print(f"- {report_base}_high_risk.parquet")
    print(f"- {master_path}")
if __name__ == "__main__":
    main()
