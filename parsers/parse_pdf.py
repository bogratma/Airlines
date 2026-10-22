import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf

SCHEMA = pa.schema([
    ('page', pa.int32()), ('side', pa.string()),
    ('orig_iata', pa.string()), ('orig_city', pa.string()), ('orig_country', pa.string()),
    ('dest_iata', pa.string()), ('dest_city', pa.string()), ('dest_country', pa.string()),
    ('valid_from', pa.date32()), ('valid_to', pa.date32()),
    ('days', pa.string()),
    ('dep_time', pa.string()), ('arr_time', pa.string()),
    ('arr_plus_days', pa.int8()),
    ('flight', pa.string()), ('airline', pa.string()), ('codeshare', pa.bool_()),
    ('aircraft', pa.string()), ('duration_min', pa.int32()),
    ('operated_by', pa.string()),
])

DEDUP_COLS = [c for c in SCHEMA.names if c not in ('page', 'side')]

MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}

def to_date(s: str) -> date:
    d, m = s.split()
    m = MONTHS[m]
    return date(2018 if m >= 11 else 2019, m, int(d))

HEADER = re.compile(r'^(?P<kind>FROM|TO):\s*(?P<body>.+?)\s+(?P<iata>[A-Z]{3})$')
def split_place(body: str):
    if ',' in body:
        city, _, country = body.partition(',')
        return city.strip(), (country.strip() or None)
    return body.strip(), None

ROW = re.compile(
    r'^(?P<vf>\d{2} \w{3})\s*-\s*(?P<vt>\d{2} \w{3})\s+'
    r'(?P<days>[1-7 ]+?)\s+'
    r'(?P<dep>\d{2}:\d{2})\s+(?P<arr>\d{2}:\d{2})(?P<plus>[+-]-?\d)?\s+'
    r'(?P<flight>[A-Z0-9]{2}\d{1,4}\*?)\s+(?P<ac>[A-Z0-9]{3})\s+'
    r'(?P<h>\d+)H(?P<m>\d+)M$')
OPER = re.compile(r'^Operated by:\s*(.+)$')
NOISE = re.compile(r'^(Validity|Time(\s+Time)*$|Dep\b|Arr\b|Consult your travel agent)')


def page_lines(page, clip, tol):
    words = page.get_text("words", clip=clip)
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines, cur, cur_y = [], [], None
    for w in words:
        yc = (w[1] + w[3]) / 2
        if cur_y is None or abs(yc - cur_y) <= tol:
            cur.append(w)
            if cur_y is None:
                cur_y = yc
        else:
            lines.append(cur)
            cur, cur_y = [w], yc
    if cur:
        lines.append(cur)
    return [" ".join(x[4] for x in sorted(l, key=lambda w: w[0])) for l in lines]


def parse_aircraft(doc, page_idx):
    text = doc[page_idx].get_text("text")
    pairs = re.findall(r'^([A-Z0-9]{3})\s*:\s*(.+)$', text, flags=re.M)
    return pl.DataFrame(pairs, schema=['code', 'name'], orient='row').unique(maintain_order=True)


def _parse_pages(doc, out_raw, out_unmatched, first_page, last_page, batch_pages, line_tol, verbose):
    route = {'L': {}, 'R': {}}
    last = {'L': None, 'R': None}
    rows, cur_page = [], [0]
    n_rows = n_unmatched = 0

    def process(side, line, csv_w):
        nonlocal n_unmatched
        line = line.strip()
        if not line:
            return
        if m := HEADER.match(line):
            city, country = split_place(m['body'])
            if m['kind'] == 'FROM':
                route[side] = {'orig_iata': m['iata'], 'orig_city': city, 'orig_country': country}
                last[side] = None
            else:
                route[side] |= {'dest_iata': m['iata'], 'dest_city': city, 'dest_country': country}
        elif m := ROW.match(line):
            fl = m['flight']
            r = {**route[side], 'page': cur_page[0], 'side': side,
                 'valid_from': to_date(m['vf']), 'valid_to': to_date(m['vt']),
                 'days': re.sub(r'\s', '', m['days']),
                 'dep_time': m['dep'], 'arr_time': m['arr'],
                 'arr_plus_days': int(m['plus'].lstrip('+')) if m['plus'] else 0,   # бывает и -1 ("+-1")
                 'flight': fl.rstrip('*'), 'airline': fl[:2],
                 'codeshare': fl.endswith('*'), 'aircraft': m['ac'],
                 'duration_min': int(m['h']) * 60 + int(m['m']),
                 'operated_by': None}
            rows.append(r)
            last[side] = r
        elif m := OPER.match(line):
            if last[side]:
                last[side]['operated_by'] = m[1]
        elif not NOISE.match(line):
            csv_w.writerow([cur_page[0], side, line])
            n_unmatched += 1
            if line.startswith(('FROM:', 'TO:')):
                route[side] = {}
                last[side] = None

    def flush(writer, keep):
        nonlocal n_rows
        ready = [r for r in rows if not any(r is k for k in keep)]
        pending = [r for r in rows if any(r is k for k in keep)]
        if ready:
            writer.write_table(pa.Table.from_pylist(ready, schema=SCHEMA))
            n_rows += len(ready)
        rows[:] = pending

    with open(out_unmatched, 'w', newline='', encoding='utf-8') as f_un, \
            pq.ParquetWriter(out_raw, SCHEMA, compression='zstd') as writer:
        csv_w = csv.writer(f_un)
        csv_w.writerow(['page', 'side', 'line'])
        for i in range(first_page, last_page):
            page = doc[i]
            cur_page[0] = i + 1
            w, h = page.rect.width, page.rect.height
            for side, clip in (('L', pymupdf.Rect(0, 0, w / 2, h)),
                               ('R', pymupdf.Rect(w / 2, 0, w, h))):
                for line in page_lines(page, clip, line_tol):
                    process(side, line, csv_w)
            if (i - first_page + 1) % batch_pages == 0:
                flush(writer, (last['L'], last['R']))
                if verbose:
                    print(f"страница {i + 1}/{last_page}, строк записано: {n_rows}", flush=True)
        flush(writer, ())
    return n_rows, n_unmatched


def dedup(raw, out):
    (pl.scan_parquet(raw)
       .unique(subset=DEDUP_COLS, keep='first', maintain_order=True)
       .sink_parquet(out))


@dataclass(frozen=True)
class ParseResult:
    raw_path: Path
    schedule_path: Path | None
    unmatched_path: Path
    aircraft_path: Path | None
    pages_processed: int
    n_raw: int
    n_schedule: int | None
    n_unmatched: int

    def schedule(self) -> pl.DataFrame:
        return pl.read_parquet(self.schedule_path or self.raw_path)


def parse_pdf(pdf_path, out_dir, first_page=4, last_page=None, batch_pages=500,
              line_tol=3.0, aircraft_page=3, deduplicate=True, build_aircraft=True,
              verbose=True) -> ParseResult:
    pdf_path, out = Path(pdf_path), Path(out_dir)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")
    if batch_pages < 1:
        raise ValueError("batch_pages должен быть >= 1")
    out.mkdir(parents=True, exist_ok=True)
    raw, sched = out / "skyteam_raw.parquet", out / "skyteam_schedule.parquet"
    unmatched, aircraft = out / "skyteam_unmatched.csv", out / "skyteam_aircraft.parquet"

    with pymupdf.open(pdf_path) as doc:
        n_pages = len(doc)
        end = n_pages if last_page is None else min(last_page, n_pages)
        if not 0 <= first_page < end:
            raise ValueError(f"некорректный диапазон страниц: first_page={first_page}, "
                             f"last_page={last_page}, в файле {n_pages} стр.")

        n_raw, n_unmatched = _parse_pages(doc, raw, unmatched, first_page, end,
                                          batch_pages, line_tol, verbose)
        if verbose:
            print(f"готово: {n_raw} строк -> {raw}")

        aircraft_path = None
        if build_aircraft:
            if aircraft_page < n_pages:
                parse_aircraft(doc, aircraft_page).write_parquet(aircraft)
                aircraft_path = aircraft
            elif verbose:
                print(f"справочник самолётов пропущен: в файле нет страницы {aircraft_page}")

    n_sched = None
    if deduplicate:
        dedup(raw, sched)
        n_sched = pl.scan_parquet(sched).select(pl.len()).collect().item()
        if verbose:
            print(f"строк до дедупликации: {n_raw}, после: {n_sched}")

    return ParseResult(raw_path=raw, schedule_path=sched if deduplicate else None,
                       unmatched_path=unmatched, aircraft_path=aircraft_path,
                       pages_processed=end - first_page, n_raw=n_raw,
                       n_schedule=n_sched, n_unmatched=n_unmatched)
