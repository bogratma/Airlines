import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("first_name", pa.string()),
    ("last_name", pa.string()),
    ("card_number", pa.string()),
    ("bonus_program", pa.string()),
    ("flight_code", pa.string()),
    ("flight_date", pa.date32()),
    ("departure", pa.string()),
    ("arrival", pa.string()),
    ("fare", pa.string()),
])


def _text(elem, tag):
    child = elem.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _parse_date(value):
    try:
        return dt.date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _user_to_rows(user):
    name = user.find("name")
    first = name.get("first") if name is not None else None
    last = name.get("last") if name is not None else None
    uid = user.get("uid")

    for card in user.findall("./cards/card"):
        card_number = card.get("number")
        program = _text(card, "bonusprogramm")

        for act in card.findall("./activities/activity"):
            yield {
                "uid": uid,
                "first_name": first,
                "last_name": last,
                "card_number": card_number,
                "bonus_program": program,
                "flight_code": _text(act, "Code"),
                "flight_date": _parse_date(_text(act, "Date")),
                "departure": _text(act, "Departure"),
                "arrival": _text(act, "Arrival"),
                "fare": _text(act, "Fare"),
            }


def parse_xml(
    input_path: str,
    output_path: str,
    batch_size: int = 100_000,
    compression: str = "snappy",
    verbose: bool = True,):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0

    with pq.ParquetWriter(output_path, SCHEMA, compression=compression) as writer:

        def flush():
            nonlocal rows, total
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                total += len(rows)
                rows = []

        context = ET.iterparse(input_path, events=("start", "end"))
        _, root = next(context)

        for event, elem in context:
            if event == "end" and elem.tag == "user":
                rows.extend(_user_to_rows(elem))
                root.clear()
                if len(rows) >= batch_size:
                    flush()
        flush()

    if verbose:
        print(f"Готово: записано {total} строк в {output_path}")
    return total
