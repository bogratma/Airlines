import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl


def _load(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pl.read_excel(path)
    return pl.read_parquet(path)


def _top(df, col, n):
    return (df.drop_nulls(col).group_by(col).len()
              .sort(["len", col], descending=[True, False]).head(n))


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  сохранено:", path)


def _barh(labels, values, title, xlabel, out_dir, name, color="#4C78A8"):
    if len(labels) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, max(3, 0.38 * len(labels) + 1.5)))
    ax.barh([str(x) for x in labels][::-1], list(values)[::-1], color=color)
    for y, v in enumerate(list(values)[::-1]):
        ax.text(v, y, f" {v}", va="center", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.margins(x=0.1)
    _save(fig, out_dir, name)


def plot_suspicious_report(
    report_path=r"process/suspicious_passengers_report.xlsx",
    out_dir="process/charts",
    top_n=15,
):

    os.makedirs(out_dir, exist_ok=True)
    df = _load(report_path)
    cols = set(df.columns)
    print(f"Загружено строк: {len(df)}, пассажиров: {df['person_id'].n_unique()}")
    agg = [pl.len().alias("events")]
    if "flight_code" in cols and "date" in cols:
        agg.append(pl.struct("flight_code", "date").n_unique().alias("flights"))
    for c in ("max_severity", "name_key", "passport", "anomaly_types_count"):
        if c in cols:
            agg.append(pl.col(c).drop_nulls().first().alias(c))
    persons = df.group_by("person_id").agg(agg)

    if "max_severity" in persons.columns:
        top = persons.sort(["max_severity", "events"], descending=True).head(top_n)
        labels = [
            f"{(r['name_key'] or '?')[:28]} (id {r['person_id']})" for r in top.iter_rows(named=True)
        ]
        _barh(labels, top["max_severity"].to_list(),
              f"Топ-{top_n} пассажиров по баллу подозрительности", "балл",
              out_dir, "01_top_persons_score.png", "#E45756")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        vals = persons["max_severity"].drop_nulls().to_list()
        ax.hist(vals, bins=range(int(min(vals)), int(max(vals)) + 2), color="#4C78A8",
                edgecolor="white", align="left")
        ax.set_title("Распределение пассажиров по баллу подозрительности")
        ax.set_xlabel("балл")
        ax.set_ylabel("пассажиров")
        ax.set_yscale("log")
        _save(fig, out_dir, "02_score_distribution.png")

    if "flights" in persons.columns:
        top = persons.sort("flights", descending=True).head(top_n)
        labels = [
            f"{(r['name_key'] or '?')[:28]} (id {r['person_id']})" for r in top.iter_rows(named=True)
        ]
        _barh(labels, top["flights"].to_list(),
              f"Топ-{top_n} пассажиров по числу рейсов в отчёте", "рейсов",
              out_dir, "03_top_persons_flights.png", "#54A24B")

    if {"flights", "max_severity"} <= set(persons.columns):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(persons["flights"], persons["max_severity"], alpha=0.4, s=18, color="#4C78A8")
        ax.set_title("Число рейсов vs балл подозрительности")
        ax.set_xlabel("рейсов у пассажира")
        ax.set_ylabel("балл")
        ax.grid(alpha=0.3)
        _save(fig, out_dir, "04_flights_vs_score.png")

    if "all_suspicion_reasons" in cols:
        reasons = (
            df.select("person_id", "flight_code", "date", "all_suspicion_reasons")
              .with_columns(reason=pl.col("all_suspicion_reasons").str.split(" | "))
              .explode("reason")
              .with_columns(
                  anomaly=pl.col("reason").str.extract(r"^([^(.\[]+)", 1).str.strip_chars())
              .drop_nulls("anomaly")
        )
        by_type = (reasons.group_by("anomaly")
                          .agg(pl.struct("person_id", "flight_code", "date").n_unique().alias("events"),
                               pl.col("person_id").n_unique().alias("persons"))
                          .sort("events", descending=True))
        _barh(by_type["anomaly"].to_list(), by_type["events"].to_list(),
              "Типы аномалий: число событий (рейсов)", "событий",
              out_dir, "05_anomaly_types_events.png", "#F58518")
        by_p = by_type.sort("persons", descending=True)
        _barh(by_p["anomaly"].to_list(), by_p["persons"].to_list(),
              "Типы аномалий: число пассажиров", "пассажиров",
              out_dir, "06_anomaly_types_persons.png", "#B279A2")

    events = df.unique(subset=[c for c in ("person_id", "flight_code", "date") if c in cols])

    if "flight_code" in cols:
        t = _top(events, "flight_code", top_n)
        _barh(t["flight_code"].to_list(), t["len"].to_list(),
              f"Топ-{top_n} рейсов по числу подозрительных событий", "событий",
              out_dir, "07_top_flights.png")

    for col, title, name in (("dep_airport", "вылета", "08_top_dep_airports.png"),
                             ("arr_airport", "прилёта", "09_top_arr_airports.png")):
        if col in cols:
            t = _top(events, col, top_n)
            _barh(t[col].to_list(), t["len"].to_list(),
                  f"Топ-{top_n} аэропортов {title} (подозрительные события)", "событий",
                  out_dir, name, "#72B7B2")

    if {"dep_airport", "arr_airport"} <= cols:
        routes = (events.drop_nulls(["dep_airport", "arr_airport"])
                        .with_columns(route=pl.col("dep_airport") + " → " + pl.col("arr_airport")))
        t = _top(routes, "route", top_n)
        _barh(t["route"].to_list(), t["len"].to_list(),
              f"Топ-{top_n} маршрутов", "событий", out_dir, "10_top_routes.png", "#9D755D")

    if "card_program" in cols:
        t = _top(events, "card_program", top_n)
        _barh(t["card_program"].to_list(), t["len"].to_list(),
              "Программы лояльности в подозрительных событиях", "событий",
              out_dir, "11_card_programs.png", "#EECA3B")

    if "dep_ts" in cols:
        ts = events["dep_ts"]
        if ts.dtype == pl.String:
            ts = ts.str.to_datetime(strict=False)
        t = pl.DataFrame({"ts": ts}).drop_nulls()
        if len(t):
            m = (t.with_columns(month=pl.col("ts").dt.truncate("1mo"))
                  .group_by("month").len().sort("month"))
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.bar(m["month"].dt.strftime("%Y-%m").to_list(), m["len"].to_list(), color="#4C78A8")
            ax.set_title("Подозрительные события по месяцам вылета")
            ax.set_ylabel("событий")
            ax.tick_params(axis="x", rotation=60)
            _save(fig, out_dir, "12_by_month.png")

            wd = t.group_by(pl.col("ts").dt.weekday().alias("d")).len().sort("d")
            hr = t.group_by(pl.col("ts").dt.hour().alias("h")).len().sort("h")
            names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            axes[0].bar([names[d - 1] for d in wd["d"]], wd["len"], color="#54A24B")
            axes[0].set_title("По дням недели")
            axes[1].bar(hr["h"], hr["len"], color="#F58518")
            axes[1].set_title("По часу вылета")
            axes[1].set_xlabel("час")
            _save(fig, out_dir, "13_weekday_hour.png")

    print(f"Готово, графики в: {os.path.abspath(out_dir)}")