import argparse
import csv
import gzip
import lzma
import sqlite3
import sys
from pathlib import Path

# Maps CSV headers -> SQLite column names
COLUMN_MAP = {
    'Form': 'form',
    'English Form': 'english_form',
    'Part of Speech': 'pos',
    'Korean Definition': 'kor_def',
    'English Definition': 'eng_def',
    'Usages': 'usages',
    'Vocabulary Level': 'vocab_level',
    'Semantic Category': 'semantic_category',
}


def build_sqlite(csv_path: Path, db_path: Path) -> int:
    """Load the CSV into a fresh SQLite DB. Returns the number of rows inserted."""
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE entries (
            id INTEGER PRIMARY KEY,
            form TEXT,
            english_form TEXT,
            pos TEXT,
            kor_def TEXT,
            eng_def TEXT,
            usages TEXT,
            vocab_level TEXT,
            semantic_category TEXT
        )''')

        # Allow very large fields (usages column can be long)
        csv.field_size_limit(sys.maxsize)

        with csv_path.open(encoding='utf-8') as f:
            reader = csv.DictReader(f)
            missing = [h for h in COLUMN_MAP if h not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV is missing expected columns: {missing}")

            columns = list(COLUMN_MAP.values())
            placeholders = ','.join('?' for _ in columns)
            insert_sql = f"INSERT INTO entries ({','.join(columns)}) VALUES ({placeholders})"

            rows = (
                tuple(row[header] for header in COLUMN_MAP)
                for row in reader
            )
            cur.executemany(insert_sql, rows)
            count = cur.execute('SELECT COUNT(*) FROM entries').fetchone()[0]

        # Indexes for the two search paths (Korean headword + English gloss)
        cur.execute('CREATE INDEX idx_form ON entries(form)')
        cur.execute('CREATE INDEX idx_english_form ON entries(english_form)')

        conn.commit()
        return count
    finally:
        conn.close()


def compress_gzip(src: Path, dst: Path) -> None:
    with src.open('rb') as f_in, gzip.open(dst, 'wb', compresslevel=9) as f_out:
        f_out.writelines(f_in)


def compress_xz(src: Path, dst: Path) -> None:
    with src.open('rb') as f_in, lzma.open(dst, 'wb', preset=9) as f_out:
        f_out.writelines(f_in)


def human_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.2f} MB"


def main():
    parser = argparse.ArgumentParser(description="Convert the parsed dictionary CSV into SQLite (+ compressed).")
    parser.add_argument('--input', '-i', type=str, default='results_new.csv', help='Input CSV path')
    parser.add_argument('--output_dir', '-o', type=str, default='db', help='Output directory for the database files')
    parser.add_argument('--name', '-n', type=str, default='nikl_dict', help='Base name for the database files')
    parser.add_argument('--htm', type=str, default='res/6000_korean_words.htm', help='6000-word HTM for flashcards')
    parser.add_argument('--no-flashcards', action='store_true', help='Skip building the flashcards table')
    parser.add_argument('--no-gzip', action='store_true', help='Skip the gzip (.gz) build')
    parser.add_argument('--xz', action='store_true', help='Also build an xz (.xz) archive (smaller, less convenient on Flutter)')
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        parser.error(f"Input CSV not found: {csv_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / f"{args.name}.sqlite"

    print(f"Building entries from {csv_path} ...")
    count = build_sqlite(csv_path, db_path)
    print(f"  wrote {count} rows -> {db_path} ({human_mb(db_path)})")

    if not args.no_flashcards:
        from build_flashcards import build_flashcards

        htm_path = Path(args.htm)
        if not htm_path.exists():
            parser.error(f"Flashcards HTM not found: {htm_path}")

        flashcards_csv = out_dir / 'flashcards.csv'
        flashcards_db = out_dir / 'flashcards.sqlite'
        print(f"Building flashcards from {htm_path} ...")
        cards = build_flashcards(
            htm_path, csv_path,
            db_path=db_path,
            csv_out=flashcards_csv,
            standalone_db=flashcards_db,
        )
        matched = sum(r['has_entry'] for r in cards)
        with_sent = sum(1 for r in cards if r['sentence_kr'])
        print(f"  wrote {len(cards)} flashcards ({matched} matched, {with_sent} with examples)")
        print(f"  also wrote {flashcards_csv}")
        print(f"  also wrote {flashcards_db} ({human_mb(flashcards_db)})")
        if not args.no_gzip:
            fc_gz = flashcards_db.with_suffix('.sqlite.gz')
            compress_gzip(flashcards_db, fc_gz)
            print(f"  also wrote {fc_gz} ({human_mb(fc_gz)})")

    # Compact after both tables are in place
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('VACUUM')
    finally:
        conn.close()
    print(f"  final DB size: {human_mb(db_path)}")

    if not args.no_gzip:
        gz_path = db_path.with_suffix('.sqlite.gz')
        print("Compressing (gzip) ...")
        compress_gzip(db_path, gz_path)
        print(f"  {gz_path} ({human_mb(gz_path)})")

    if args.xz:
        xz_path = db_path.with_suffix('.sqlite.xz')
        print("Compressing (xz) ...")
        compress_xz(db_path, xz_path)
        print(f"  {xz_path} ({human_mb(xz_path)})")

    print("Done.")


if __name__ == "__main__":
    main()
