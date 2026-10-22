import pandas as pd


def parse_csv(csv_path, output_path):
    df = pd.read_csv(
        csv_path,
        sep=";",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    df = df.rename(columns={
        "PassengerFirstName": "first_name",
        "PassengerSecondName": "patronymic",
        "PassengerLastName": "last_name",
        "PassengerSex": "sex",
        "PassengerBirthDate": "birth_date",
        "PassengerDocument": "document",
        "BookingCode": "booking_code",
        "TicketNumber": "ticket_number",
        "Baggage": "baggage",
        "FlightDate": "flight_date",
        "FlightTime": "flight_time",
        "FlightNumber": "flight_number",
        "CodeShare": "codeshare",
        "Destination": "destination",
    })

    for c in df.columns:
        df[c] = df[c].str.strip()

    for c in ["booking_code", "ticket_number"]:
        df[f"has_{c}"] = df[c] != "Not presented"
        df[c] = df[c].where(df[c] != "Not presented", other=pd.NA)

    df["birth_date"] = pd.to_datetime(df["birth_date"], format="%m/%d/%Y", errors="coerce")
    df["flight_datetime"] = pd.to_datetime(
        df["flight_date"] + " " + df["flight_time"], format="%Y-%m-%d %H:%M", errors="coerce"
    )
    df["flight_date"] = df["flight_datetime"].dt.normalize()
    df["age_at_flight"] = ((df["flight_datetime"] - df["birth_date"]).dt.days // 365.25).astype("Int16")
    df["patronymic_is_initial"] = df["patronymic"].str.fullmatch(r"[A-Z]\.")
    doc = df["document"].str.extract(r"^(\d{4})\s(\d{6})$")
    df["doc_series"] = doc[0]
    df["doc_number"] = doc[1]
    df["doc_valid_format"] = doc[0].notna()

    df["flight_id"] = df["flight_number"] + "_" + df["flight_datetime"].dt.strftime("%Y%m%d_%H%M")
    df["person_key"] = (
        df["last_name"] + "|" + df["first_name"] + "|" + df["birth_date"].dt.strftime("%Y-%m-%d")
    )

    for c in ["sex", "baggage", "codeshare", "destination", "flight_number", "flight_id"]:
        df[c] = df[c].astype("category")

    cols = [
        "person_key", "last_name", "first_name", "patronymic", "patronymic_is_initial",
        "sex", "birth_date", "age_at_flight",
        "document", "doc_series", "doc_number", "doc_valid_format",
        "booking_code", "has_booking_code", "ticket_number", "has_ticket_number",
        "flight_id", "flight_number", "flight_datetime", "flight_date",
        "codeshare", "destination", "baggage",
    ]
    df = df[cols].sort_values(["flight_datetime", "flight_id"]).reset_index(drop=True)

    print(df.shape)
    print(df.isna().sum())
    print("Некорректных документов:", (~df["doc_valid_format"]).sum())
    print("Полных дублей:", df.duplicated().sum())

    df.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
    return df