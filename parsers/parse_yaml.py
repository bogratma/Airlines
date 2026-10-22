import re
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

DATE_RE = re.compile(r"^\s*'?\d{4}-\d{2}-\d{2}'?\s*(?::\s*|\s*#.*)?$")

COLS = ["date", "flight", "from", "to", "status",
        "loyalty_program", "loyalty_number", "class", "fare"]
SCHEMA = pa.schema([(c, pa.string()) for c in COLS])


def _s(v):
    return None if v is None else str(v)


def parse_yaml(path, output_parquet_path, encoding="utf-8", batch_size=50_000):
    out = Path(output_parquet_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    batch = []
    total = 0

    with pq.ParquetWriter(out, SCHEMA, compression="snappy") as writer:

        def flush():
            nonlocal total
            if not batch:
                return
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            total += len(batch)
            batch.clear()

        def process(block):
            cleaned_block = [line.replace("\xa0", " ") for line in block]
            block_text = "".join(cleaned_block)

            try:
                data = yaml.load(block_text, Loader=Loader)
            except Exception as e:
                print(f"\n[ОШИБКА YAML БЛОКА] Не удалось распарсить YAML:\n{e}")
                return

            if not isinstance(data, dict):
                return

            for date, flights in data.items():
                if not isinstance(flights, dict):
                    continue

                for flight, info in flights.items():
                    try:
                        info = info if isinstance(info, dict) else {}

                        base = {
                            "date": _s(date),
                            "flight": _s(flight),
                            "from": _s(info.get("FROM")),
                            "to": _s(info.get("TO")),
                            "status": _s(info.get("STATUS")),
                        }

                        ff = info.get("FF")
                        if isinstance(ff, dict) and ff:
                            for card, d in ff.items():
                                parts = str(card).split(" ", 1)
                                d = d if isinstance(d, dict) else {}
                                batch.append({
                                    **base,
                                    "loyalty_program": parts[0],
                                    "loyalty_number": parts[1].strip() if len(parts) > 1 else None,
                                    "class": _s(d.get("CLASS")),
                                    "fare": _s(d.get("FARE")),
                                })
                        else:
                            batch.append({
                                **base,
                                "loyalty_program": None,
                                "loyalty_number": None,
                                "class": None,
                                "fare": None
                            })
                    except Exception as flight_err:
                        print(f"[ПРОПУЩЕН РЕЙС] Дата: {date}, Рейс: {flight} | Причина: {flight_err}")
                        continue

            if len(batch) >= batch_size:
                flush()

        block = []
        with open(path, encoding=encoding) as f:
            for line in f:
                line_clean = line.replace("\xa0", " ")
                if DATE_RE.match(line_clean):
                    if block:
                        process(block)
                    block = [line]
                elif block:
                    block.append(line)
            if block:
                process(block)
        flush()
    df = pl.read_parquet(out)
    print(f"Сохранено: {out} (строк: {len(df)})")
    return df