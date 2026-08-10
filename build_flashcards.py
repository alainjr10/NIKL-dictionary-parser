"""
Build a flashcards table from the 6000 common-words HTM, augmented with
NIKL dictionary defs/examples.

Primary level = HTM A/B/C (national common-vocab ranking).
Secondary = NIKL 초급/중급/고급 (kept as dict_level for reference).

Enrichment (automatic from NIKL):
- semantic_category, topic_category
- synonyms (shared English gloss tokens)
- related_words (same semantic category peers)
- word_family (NIKL RelatedForm derivations)
Antonyms are not inferred automatically (not in source data).
"""
import argparse
import ast
import csv
import gzip
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

LEVEL_LABEL = {
    'A': 'Beginner',
    'B': 'Intermediate',
    'C': 'Advanced',
}

# NIKL top-level domain → app-friendly topic label
TOPIC_FROM_SEMANTIC = {
    '경제 생활': 'Economy',
    '교육': 'Education',
    '자연': 'Nature & Science',
    '동식물': 'Nature & Science',
    '과학': 'Science',
    '사회 생활': 'Society',
    '정치와 행정': 'Politics & Government',
    '문화': 'Culture',
    '인간': 'People & Emotions',
    '주생활': 'Daily Life',
    '삶': 'Daily Life',
    '식생활': 'Food & Dining',
    '의생활': 'Health & Medicine',
    '종교': 'Religion',
    '개념': 'General',
}

# Skip ultra-generic English tokens when matching synonyms
ENG_STOPWORDS = frozenset({
    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'over',
    'under', 'very', 'more', 'less', 'good', 'bad', 'thing', 'person',
    'people', 'state', 'act', 'action', 'something', 'someone', 'used',
    'being', 'having', 'make', 'take', 'give', 'come', 'go', 'do',
})

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


def load_related_forms(json_dir: Path) -> dict[str, list[str]]:
    """Map headword -> related surface forms from NIKL RelatedForm (derivations)."""
    related: dict[str, list[str]] = defaultdict(list)
    if not json_dir.is_dir():
        return {}

    def feats_dict(obj) -> dict:
        feats = obj.get('feat', [])
        if isinstance(feats, dict):
            feats = [feats]
        return {f.get('att'): f.get('val') for f in feats if isinstance(f, dict)}

    for path in sorted(json_dir.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for entry in data['LexicalResource']['Lexicon']['LexicalEntry']:
            lem = entry.get('Lemma')
            if isinstance(lem, list):
                lem = lem[0]
            head = feats_dict(lem).get('writtenForm')
            if not head:
                continue
            rf = entry.get('RelatedForm')
            if not rf:
                continue
            items = rf if isinstance(rf, list) else [rf]
            for item in items:
                d = feats_dict(item)
                wf = d.get('writtenForm')
                if wf and wf != head and wf not in related[head]:
                    related[head].append(wf)
    return dict(related)


def topic_from_semantic(semantic: str | None) -> str | None:
    if not semantic or str(semantic) in ('', 'nan'):
        return None
    top = str(semantic).split('>')[0].strip()
    return TOPIC_FROM_SEMANTIC.get(top, top)


def english_tokens(nikl_row: dict | None, htm_meaning: str | None = None) -> set[str]:
    tokens: set[str] = set()
    if nikl_row:
        for field in ('English Form', 'English Definition'):
            for part in _safe_list(nikl_row.get(field)):
                for piece in re.split(r'[;,/]', str(part)):
                    t = re.sub(r'[^a-zA-Z\s-]', '', piece).strip().lower()
                    if len(t) >= 3 and t not in ENG_STOPWORDS:
                        tokens.add(t)
    if htm_meaning:
        for piece in re.split(r'[;,/]', htm_meaning):
            t = re.sub(r'[^a-zA-Z\s-]', '', piece).strip().lower()
            if len(t) >= 3 and t not in ENG_STOPWORDS:
                tokens.add(t)
    return tokens


def build_relation_indexes(
    flashcard_words: set[str],
    nikl_index: dict[str, dict],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Precompute synonym candidates (eng token) and semantic peers."""
    by_token: dict[str, list[str]] = defaultdict(list)
    by_semantic: dict[str, list[str]] = defaultdict(list)

    for form in flashcard_words:
        row = nikl_index.get(form)
        if not row:
            continue
        sc = row.get('Semantic Category')
        if sc and str(sc) not in ('', 'nan'):
            by_semantic[str(sc)].append(form)
        for t in english_tokens(row):
            by_token[t].append(form)

    return dict(by_token), dict(by_semantic)


def pick_synonyms(
    word: str,
    nikl_row: dict | None,
    htm_meaning: str,
    by_token: dict[str, list[str]],
    limit: int = 8,
) -> list[str]:
    tokens = english_tokens(nikl_row, htm_meaning)
    scored: dict[str, int] = defaultdict(int)
    for t in tokens:
        for other in by_token.get(t, []):
            if other != word:
                scored[other] += 1
    ranked = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:limit]]


def pick_related(
    word: str,
    nikl_row: dict | None,
    synonyms: list[str],
    by_semantic: dict[str, list[str]],
    limit: int = 8,
) -> list[str]:
    if not nikl_row:
        return []
    sc = nikl_row.get('Semantic Category')
    if not sc or str(sc) in ('', 'nan'):
        return []
    peers = by_semantic.get(str(sc), [])
    skip = {word, *synonyms}
    out = [p for p in peers if p not in skip]
    return out[:limit]


def augment(
    htm_rows: list[dict],
    nikl_index: dict[str, dict],
    related_forms: dict[str, list[str]],
    by_token: dict[str, list[str]],
    by_semantic: dict[str, list[str]],
) -> list[dict]:
    out = []
    for row in htm_rows:
        nikl = nikl_index.get(row['word'])
        meaning_kr, eng_def, english_form = first_definitions(nikl)
        dict_level = None
        dict_pos = None
        semantic_category = None
        topic_category = None
        if nikl:
            vl = nikl.get('Vocabulary Level')
            if vl and str(vl) not in ('', 'nan', 'None'):
                dict_level = str(vl)
            pos = nikl.get('Part of Speech')
            if pos and str(pos) not in ('', 'nan', 'None'):
                dict_pos = str(pos)
            sc = nikl.get('Semantic Category')
            if sc and str(sc) not in ('', 'nan', 'None'):
                semantic_category = str(sc)
                topic_category = topic_from_semantic(semantic_category)

        synonyms = pick_synonyms(row['word'], nikl, row['meaning_en'], by_token)
        related = pick_related(row['word'], nikl, synonyms, by_semantic)
        family = related_forms.get(row['word'], [])[:8]
        # antonyms: not in NIKL export — leave empty for manual/LLM later
        antonyms: list[str] = []

        out.append({
            'id': row['id'],
            'word': row['word'],
            'meaning_en': row['meaning_en'],
            'meaning_kr': meaning_kr,
            'english_form': english_form,
            'eng_def': eng_def,
            'sentence_kr': pick_sentence_kr(nikl, max_sentences=1),
            'sentence_en': None,
            'level': row['level'],
            'level_label': row['level_label'],
            'dict_level': dict_level,
            'pos': row['pos'],
            'dict_pos': dict_pos,
            'has_entry': 1 if nikl else 0,
            'semantic_category': semantic_category,
            'topic_category': topic_category,
            'synonyms': json.dumps(synonyms, ensure_ascii=False),
            'related_words': json.dumps(related, ensure_ascii=False),
            'word_family': json.dumps(family, ensure_ascii=False),
            'antonyms': json.dumps(antonyms, ensure_ascii=False),
        })
    return out


def enrich_nikl_row(
    form: str,
    nikl_row: dict,
    by_token: dict[str, list[str]],
    by_semantic: dict[str, list[str]],
    related_forms: dict[str, list[str]],
) -> dict[str, str | None]:
    """Compute relation JSON fields for a dictionary entry row."""
    sc = nikl_row.get('Semantic Category')
    semantic = str(sc) if sc and str(sc) not in ('', 'nan', 'None') else None
    topic = topic_from_semantic(semantic) if semantic else None
    synonyms = pick_synonyms(form, nikl_row, '', by_token)
    related = pick_related(form, nikl_row, synonyms, by_semantic)
    family = related_forms.get(form, [])[:8]
    return {
        'topic_category': topic,
        'synonyms': json.dumps(synonyms, ensure_ascii=False),
        'related_words': json.dumps(related, ensure_ascii=False),
        'word_family': json.dumps(family, ensure_ascii=False),
        'antonyms': json.dumps([], ensure_ascii=False),
    }


def build_entry_relations(
    forms: set[str],
    nikl_index: dict[str, dict],
    json_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Precompute indexes and related forms for dictionary enrichment."""
    related_forms = load_related_forms(json_dir)
    by_token, by_semantic = build_relation_indexes(forms, nikl_index)
    return by_token, by_semantic, related_forms


FLASHCARD_COLUMNS = [
    'id', 'word', 'meaning_en', 'meaning_kr', 'english_form', 'eng_def',
    'sentence_kr', 'sentence_en', 'level', 'level_label', 'dict_level',
    'pos', 'dict_pos', 'has_entry',
    'semantic_category', 'topic_category',
    'synonyms', 'related_words', 'word_family', 'antonyms',
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
        has_entry INTEGER NOT NULL DEFAULT 0,
        semantic_category TEXT,
        topic_category TEXT,
        synonyms TEXT,
        related_words TEXT,
        word_family TEXT,
        antonyms TEXT
    )''')
    cur.executemany(
        f"INSERT INTO flashcards ({','.join(FLASHCARD_COLUMNS)}) VALUES ({','.join('?' for _ in FLASHCARD_COLUMNS)})",
        [tuple(r[c] for c in FLASHCARD_COLUMNS) for r in rows],
    )
    cur.execute('CREATE INDEX idx_flashcards_word ON flashcards(word)')
    cur.execute('CREATE INDEX idx_flashcards_level ON flashcards(level)')
    cur.execute('CREATE INDEX idx_flashcards_topic ON flashcards(topic_category)')
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
    json_dir: Path | None = None,
    db_path: Path | None = None,
    csv_out: Path | None = None,
    standalone_db: Path | None = None,
) -> list[dict]:
    htm_rows = parse_htm(htm_path)
    nikl_index = load_nikl_index(nikl_csv)
    flashcard_words = {r['word'] for r in htm_rows}
    related_forms = load_related_forms(json_dir or Path('2024_01'))
    by_token, by_semantic = build_relation_indexes(flashcard_words, nikl_index)
    rows = augment(htm_rows, nikl_index, related_forms, by_token, by_semantic)

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
    parser.add_argument('--json-dir', type=str, default='2024_01', help='NIKL JSON dir for RelatedForm')
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
        json_dir=Path(args.json_dir),
        db_path=db_path,
        csv_out=csv_out,
        standalone_db=standalone_db,
    )

    matched = sum(r['has_entry'] for r in rows)
    with_sent = sum(1 for r in rows if r['sentence_kr'])
    with_topic = sum(1 for r in rows if r.get('topic_category'))
    with_syn = sum(1 for r in rows if json.loads(r['synonyms']))
    with_rel = sum(1 for r in rows if json.loads(r['related_words']))
    with_fam = sum(1 for r in rows if json.loads(r['word_family']))
    by_level = {}
    for r in rows:
        by_level[r['level']] = by_level.get(r['level'], 0) + 1

    print(f"  flashcards: {len(rows)}")
    print(f"  matched to NIKL entries: {matched}/{len(rows)} ({matched/len(rows):.1%})")
    print(f"  with Korean example sentence: {with_sent}/{len(rows)} ({with_sent/len(rows):.1%})")
    print(f"  with topic_category: {with_topic}/{len(rows)}")
    print(f"  with synonyms: {with_syn}/{len(rows)}")
    print(f"  with related_words: {with_rel}/{len(rows)}")
    print(f"  with word_family: {with_fam}/{len(rows)}")
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
