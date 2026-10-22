import polars as pl

def drop_empty_columns(df, columns=None, empty_values=("", "N/A","LANDED"), output_path=None):
    if df.height == 0:
        print("Датафрейм пустой, ничего не удаляем.")
        return df
    empty = list(empty_values)
    cols = list(df.columns) if columns is None else list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Колонок нет, пропускаю: {missing}")
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("Нет колонок для проверки, ничего не удаляем.")
        return df
    exprs = []
    for c in cols:
        if df.schema[c] == pl.String:
            s = pl.col(c).str.strip_chars()
            exprs.append((s.is_not_null() & ~s.is_in(empty)).any().alias(c))
        else:
            exprs.append(pl.col(c).is_not_null().any().alias(c))
    has_data = df.select(exprs).row(0, named=True)
    to_drop = [c for c in cols if not has_data[c]]
    for c in cols:
        print(f"'{c}': " + ("данных нет, удаляю" if c in to_drop else "есть данные, оставляю"))
    if to_drop:
        df = df.drop(to_drop)
        if output_path:
            df.write_parquet(output_path)
            print(f"Файл перезаписан: {output_path}")
    return df

def drop_columns(df, columns, output_path):
  cols_to_drop = []
  for col in columns:
    if col in df.columns:
      if df[col].null_count() == len(df):
        print(f"'{col}': данных нет, удаляю")
      else:
        print(f"'{col}': принудительно удаляю по списку")
      cols_to_drop.append(col)
    else:
      print(f"Колонка '{col}' не найдена в датафрейме.")
  if cols_to_drop:
    df = df.drop(cols_to_drop)
  if output_path:
    df.write_parquet(output_path)
    print(f"Файл перезаписан: {output_path}")

  return df