import json
import polars as pl

def parse_json(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    flat_rows = []
    for profile in data.get("Forum Profiles", []):
        nickname = profile.get("NickName")
        sex = profile.get("Sex")
        real_name = profile.get("Real Name", {})
        last_name = real_name.get("Last Name")
        first_name = real_name.get("First Name")
        travel_docs = profile.get("Travel Documents", [])
        passports_list = []
        for doc in travel_docs:
            if isinstance(doc, dict):
                p = doc.get("Passports")
                if p is not None:
                    passports_list.append(str(p))
        passports_str = ", ".join(passports_list) if passports_list else None
        loyalty_programs = profile.get("Loyality Programm", [])
        loyalty_items = []
        for lp in loyalty_programs:
            if isinstance(lp, dict):
                prog_code = lp.get("programm")
                num = lp.get("Number")
                loyalty_items.append(f"{prog_code}:{num}")
        loyalty_str = " | ".join(loyalty_items) if loyalty_items else None
        for flight in profile.get("Registered Flights", []):
            arrival = flight.get("Arrival", {})
            departure = flight.get("Departure", {})
            flat_rows.append({
                "nickname": nickname,
                "sex": sex,
                "last_name": last_name,
                "first_name": first_name,
                "passports": passports_str,
                "loyalty_programs": loyalty_str,
                "flight_id": flight.get("Flight"),
                "date": flight.get("Date"),
                "codeshare": flight.get("Codeshare"),
                "departure_airport": departure.get("Airport"),
                "departure_city": departure.get("City"),
                "departure_country": departure.get("Country"),
                "arrival_airport": arrival.get("Airport"),
                "arrival_city": arrival.get("City"),
                "arrival_country": arrival.get("Country"),
            })

    df_json_clean = pl.DataFrame(flat_rows)
    df_json_clean.write_parquet(output_path)
    print(f"JSON успешно распарсен! Строк: {len(df_json_clean)}")
    return df_json_clean

def user_info_json(df,target_nickname):
    user_flights = df.filter(pl.col("nickname") == target_nickname)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(4000)
    print(f"Найдено рейсов для пользователя {target_nickname}: {len(user_flights)}")
    print(user_flights)

