
import pandas as pd
import polars as pl
COLUMNS = [
    "PaxName", "PaxBirthDate", "DepartDate", "DepartTime",
    "ArrivalDate", "ArrivalTime", "FlightCodeSh", "From", "Dest",
    "Code", "e-Ticket", "TravelDoc", "Seat", "Meal",
    "TrvCls", "Fare", "Baggage", "PaxAdditionalInfo", "AgentInfo",
]
def get_colspecs(header: str):
    starts, pos = [], 0
    for name in COLUMNS:
        idx = header.find(name, pos)
        if idx == -1:
            raise ValueError(f"Не нашёл колонку {name} в заголовке")
        starts.append(idx)
        pos = idx + len(name)
    return list(zip(starts, starts[1:] + [None]))

def parse_tab(tab_path, output_parquet_path, encoding: str = "utf-8"):
    with open(tab_path, encoding=encoding) as f:
        header = f.readline().rstrip("\r\n")
    df_pd = pd.read_fwf(
        tab_path,
        colspecs=get_colspecs(header),
        names=COLUMNS,
        skiprows=1,
        dtype=str,
        keep_default_na=False,
        encoding=encoding,
    )
    df = pl.from_pandas(df_pd)
    df = df.with_columns(
        pl.when(pl.col(c).str.strip_chars() == "")
        .then(None)
        .otherwise(pl.col(c).str.strip_chars())
        .alias(c)
        for c in df.columns
    )
    df = df.with_columns(
        pl.col("PaxAdditionalInfo").str.extract(r"FF#([A-Z0-9]{2})", 1).alias("loyalty_program"),
        pl.col("PaxAdditionalInfo").str.extract(r"FF#[A-Z0-9]{2}\s*(\d+)", 1).alias("loyalty_number"),
    )
    df = df.with_columns(
        pl.col("PaxAdditionalInfo")
        .str.replace(r"FF#[A-Z0-9]{2}\s*\d+", "")
        .str.strip_chars()
        .alias("PaxAdditionalInfo")
    ).with_columns(
        pl.when(pl.col("PaxAdditionalInfo") == "")
        .then(None)
        .otherwise(pl.col("PaxAdditionalInfo"))
        .alias("PaxAdditionalInfo")
    )

    df.write_parquet(output_parquet_path)
    print(f"Строк: {len(df)}, колонок: {len(df.columns)}")
    return df


