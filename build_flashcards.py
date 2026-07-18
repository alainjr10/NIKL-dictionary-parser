"""
Build a flashcards table from the 6000 common-words HTM, augmented with
NIKL dictionary defs/examples.

Primary level = HTM A/B/C (national common-vocab ranking).
Secondary = NIKL 초급/중급/고급 (kept as dict_level for reference).
"""
import argparse
import ast
import csv
import gzip
import re
import sqlite3
import sys
from pathlib import Path

LEVEL_LABEL = {
    'A': 'Beginner',
    'B': 'Intermediate',
    'C': 'Advanced',
}

# HTM lines look like:
#   24 	 B     	 1   	 대하다        	 Face, confront
HTM_ROW = re.compile(
    r'^\s*(\d+)\s+([ABC])\s+(\S+)\s+(\S+)\s+(.+?)\s*$'
)


def parse_htm(htm_path: Path) -> list[dict]:
    text = htm_path.read_text(encoding='utf-8', errors='replace')
    rows = []
    seen_ids = set()
    for line in text.splitlines():
        m = HTM_ROW.match(line)
        if not m:
            continue
        wordid, level, pos, word, defn = m.groups()
        wordid = int(wordid)
        if wordid in seen_ids:
            continue
        seen_ids.add(wordid)
        rows.append({
            'id': wordid,
            'level': level,
            'level_label': LEVEL_LABEL[level],
            'pos': pos,
            'word': word.strip(),
            'meaning_en': defn.strip(),
        })
    return rows


def load_nikl_index(csv_path: Path) -> dict[str, dict]:
    """Index NIKL CSV by Form. First occurrence wins for multi-homonym rows."""
    csv.field_size_limit(sys.maxsize)
    index = {}
    with csv_path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            form = row['Form']
            if form not in index:
                index[form] = row
    return index


def _safe_list(value) -> list:
    if value is None or value == '' or str(value) == 'nan':
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def first_definitions(nikl_row: dict | None, n: int = 1) -> tuple[str | None, str | None, str | None]:
    """Return (meaning_kr, eng_def, english_form) from the first sense(s)."""
    if not nikl_row:
        return None, None, None

    kor = _safe_list(nikl_row.get('Korean Definition'))
    eng = _safe_list(nikl_row.get('English Definition'))
    eform = _safe_list(nikl_row.get('English Form'))

    meaning_kr = kor[0] if kor else None
    eng_def = eng[0] if eng else None
    english_form = eform[0] if eform else None
    return meaning_kr, eng_def, english_form


def pick_sentence_kr(nikl_row: dict | None, max_sentences: int = 1) -> str | None:
    """Prefer standalone sentence examples; skip dialogue lists."""
    if not nikl_row:
        return None
    usages = _safe_list(nikl_row.get('Usages'))
    sentences = [u.strip() for u in usages if isinstance(u, str) and u.strip()]
    if not sentences:
        return None
    chosen = sentences[:max_sentences]
    return chosen[0] if max_sentences == 1 else ' | '.join(chosen)


def augment(htm_rows: list[dict], nikl_index: dict[str, dict]) -> list[dict]:
    out = []
    for row in htm_rows:
        nikl = nikl_index.get(row['word'])
        meaning_kr, eng_def, english_form = first_definitions(nikl)
        dict_level = None
        dict_pos = None
        if nikl:
            vl = nikl.get('Vocabulary Level')
            if vl and str(vl) not in ('', 'nan', 'None'):
                dict_level = str(vl)
            pos = nikl.get('Part of Speech')
            if pos and str(pos) not in ('', 'nan', 'None'):
                dict_pos = str(pos)

        out.append({
            'id': row['id'],
            'word': row['word'],
            'meaning_en': row['meaning_en'],
            'meaning_kr': meaning_kr,
            'english_form': english_form,
            'eng_def': eng_def,
            'sentence_kr': pick_sentence_kr(nikl, max_sentences=1),
            'sentence_en': None,  # NIKL examples are Korean-only
            'level': row['level'],           # primary: A/B/C
            'level_label': row['level_label'],
            'dict_level': dict_level,        # secondary: 초급/중급/고급
            'pos': row['pos'],
            'dict_pos': dict_pos,
            'has_entry': 1 if nikl else 0,
        })
    return out


FLASHCARD_COLUMNS = [
    'id', 'word', 'meaning_en', 'meaning_kr', 'english_form', 'eng_def',
    'sentence_kr', 'sentence_en', 'level', 'level_label', 'dict_level',
    'pos', 'dict_pos', 'has_entry',
]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FLASHCARD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_sqlite_table(rows: list[dict], conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS flashcards')
    cur.execute('''CREATE TABLE flashcards (
        id INTEGER PRIMARY KEY,
        word TEXT NOT NULL,
        meaning_en TEXT,
        meaning_kr TEXT,
        english_form TEXT,
        eng_def TEXT,
        sentence_kr TEXT,
        sentence_en TEXT,
        level TEXT NOT NULL,
        level_label TEXT NOT NULL,
        dict_level TEXT,
        pos TEXT,
        dict_pos TEXT,
        has_entry INTEGER NOT NULL DEFAULT 0
    )''')
    cur.executemany(
        f"INSERT INTO flashcards ({','.join(FLASHCARD_COLUMNS)}) VALUES ({','.join('?' for _ in FLASHCARD_COLUMNS)})",
        [tuple(r[c] for c in FLASHCARD_COLUMNS) for r in rows],
    )
    cur.execute('CREATE INDEX idx_flashcards_word ON flashcards(word)')
    cur.execute('CREATE INDEX idx_flashcards_level ON flashcards(level)')
    conn.commit()
    return cur.execute('SELECT COUNT(*) FROM flashcards').fetchone()[0]


def write_standalone_sqlite(rows: list[dict], db_path: Path) -> int:
    """Create a flashcards-only SQLite file (replaces if present)."""
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        count = write_sqlite_table(rows, conn)
        conn.execute('VACUUM')
        return count
    finally:
        conn.close()


def compress_gzip(src: Path, dst: Path) -> None:
    with src.open('rb') as f_in, gzip.open(dst, 'wb', compresslevel=9) as f_out:
        f_out.writelines(f_in)


def build_flashcards(
    htm_path: Path,
    nikl_csv: Path,
    db_path: Path | None = None,
    csv_out: Path | None = None,
    standalone_db: Path | None = None,
) -> list[dict]:
    htm_rows = parse_htm(htm_path)
    nikl_index = load_nikl_index(nikl_csv)
    rows = augment(htm_rows, nikl_index)

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, csv_out)

    if db_path is not None:
        if not db_path.exists():
            raise FileNotFoundError(
                f"SQLite DB not found: {db_path}. Run build_db.py first, or pass an existing DB."
            )
        conn = sqlite3.connect(db_path)
        try:
            write_sqlite_table(rows, conn)
        finally:
            conn.close()

    if standalone_db is not None:
        write_standalone_sqlite(rows, standalone_db)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build flashcards from 6000-word HTM, augmented with NIKL data."
    )
    parser.add_argument('--htm', type=str, default='res/6000_korean_words.htm')
    parser.add_argument('--nikl', type=str, default='results_new.csv', help='NIKL parsed CSV')
    parser.add_argument('--db', type=str, default='db/nikl_dict.sqlite', help='SQLite to add flashcards table into')
    parser.add_argument('--csv-out', type=str, default='db/flashcards.csv', help='Also write a CSV copy')
    parser.add_argument('--standalone', type=str, default='db/flashcards.sqlite',
                        help='Also write a flashcards-only SQLite file')
    parser.add_argument('--no-db', action='store_true', help='Skip writing into the main NIKL SQLite')
    parser.add_argument('--no-standalone', action='store_true', help='Skip the flashcards-only SQLite')
    parser.add_argument('--no-gzip', action='store_true', help='Skip gzip of the standalone SQLite')
    args = parser.parse_args()

    htm_path = Path(args.htm)
    nikl_path = Path(args.nikl)
    if not htm_path.exists():
        parser.error(f"HTM not found: {htm_path}")
    if not nikl_path.exists():
        parser.error(f"NIKL CSV not found: {nikl_path}")

    db_path = None if args.no_db else Path(args.db)
    csv_out = Path(args.csv_out) if args.csv_out else None
    standalone_db = None if args.no_standalone else Path(args.standalone)

    print(f"Parsing {htm_path} ...")
    rows = build_flashcards(
        htm_path, nikl_path,
        db_path=db_path,
        csv_out=csv_out,
        standalone_db=standalone_db,
    )

    matched = sum(r['has_entry'] for r in rows)
    with_sent = sum(1 for r in rows if r['sentence_kr'])
    by_level = {}
    for r in rows:
        by_level[r['level']] = by_level.get(r['level'], 0) + 1

    print(f"  flashcards: {len(rows)}")
    print(f"  matched to NIKL entries: {matched}/{len(rows)} ({matched/len(rows):.1%})")
    print(f"  with Korean example sentence: {with_sent}/{len(rows)} ({with_sent/len(rows):.1%})")
    print(f"  levels (primary A/B/C): {by_level}")
    if csv_out:
        print(f"  wrote {csv_out}")
    if db_path:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute('VACUUM')
        finally:
            conn.close()
        print(f"  wrote flashcards table -> {db_path}")
    if standalone_db:
        print(f"  wrote {standalone_db} ({standalone_db.stat().st_size / 1e6:.2f} MB)")
        if not args.no_gzip:
            gz_path = standalone_db.with_suffix('.sqlite.gz')
            compress_gzip(standalone_db, gz_path)
            print(f"  wrote {gz_path} ({gz_path.stat().st_size / 1e6:.2f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
