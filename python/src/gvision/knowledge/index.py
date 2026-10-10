"""One game's wiki on disk: SQLite full-text search over its sections.

FTS5 ranks with BM25, weighting page titles and section headings above the
body, so "what does the mason want" finds the "Trading > Mason" section. It
runs on the CPU in a few milliseconds and needs nothing installed. A section
used to answer is stored again as rendered text (``set_rendered``), which
replaces the rough wikitext version until the page changes.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from gvision.knowledge.text import Section, chunks

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further get gets getting give had has have having he
her here hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or
other our out over own same she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who whom why will with would you your
yours tell know want wants need needs main best good get got make makes one ones thing things way much many really
please game guy guys like say says said go going kind sort type types use used using worth
""".split())

TITLE_WEIGHT = 10.0
HEADING_WEIGHT = 5.0


def terms(question: str, game: str = "") -> list[str]:
    """The words worth searching for: no filler, no game name, no duplicates."""
    skip = STOPWORDS | {w for w in re.findall(r"[a-z0-9]+", game.lower())}
    out: list[str] = []
    for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", question.lower()):
        w = w.split("'")[0]
        if (len(w) >= 3 or w.isdigit()) and w not in skip and w not in out:
            out.append(w)
    return out


def _stem(word: str) -> str:
    for suffix in ("ies", "es", "s", "ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


@dataclass
class Hit:
    page: str
    section: int
    heading: str
    text: str
    rendered: bool
    named: bool
    """A searched word is in the page title or section heading: the hit is about what was asked."""

    @property
    def where(self) -> str:
        return f"{self.page} > {self.heading.split(' > ')[-1]}" if self.heading else self.page


class WikiIndex:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        try:
            self._db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS pages (title TEXT PRIMARY KEY, revid INTEGER);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                    title, heading, body, section UNINDEXED, rendered UNINDEXED,
                    tokenize='porter unicode61');
            """)
        except sqlite3.OperationalError as e:
            self._db.close()
            raise RuntimeError(f"this Python's SQLite can't search text (no FTS5): {e}") from e

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- meta ------------------------------------------------------------------

    def get(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set(self, **values: object) -> None:
        with self._lock, self._db:
            self._db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                                 [(k, None if v is None else str(v)) for k, v in values.items()])

    # --- pages -----------------------------------------------------------------

    def revid(self, title: str) -> int | None:
        with self._lock:
            row = self._db.execute("SELECT revid FROM pages WHERE title = ?", (title,)).fetchone()
        return row[0] if row else None

    def replace_page(self, title: str, revid: int, sections: Iterable[Section]) -> None:
        rows = [(title, s.heading, piece, s.index) for s in sections for piece in chunks(s.text)]
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks WHERE title = ?", (title,))
            self._db.executemany("INSERT INTO chunks (title, heading, body, section, rendered) VALUES (?, ?, ?, ?, 0)",
                                 rows)
            self._db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?)", (title, revid))

    def set_rendered(self, title: str, section: int, text: str) -> None:
        with self._lock, self._db:
            row = self._db.execute("SELECT heading FROM chunks WHERE title = ? AND section = ? LIMIT 1",
                                   (title, section)).fetchone()
            if row is None:
                return
            self._db.execute("DELETE FROM chunks WHERE title = ? AND section = ?", (title, section))
            self._db.executemany("INSERT INTO chunks (title, heading, body, section, rendered) VALUES (?, ?, ?, ?, 1)",
                                 [(title, row[0], piece, section) for piece in chunks(text)])

    def prune(self, keep: set[str]) -> int:
        """Drop pages that are no longer on the wiki (after a full rebuild)."""
        with self._lock, self._db:
            gone = [t for (t,) in self._db.execute("SELECT title FROM pages") if t not in keep]
            for t in gone:
                self._db.execute("DELETE FROM chunks WHERE title = ?", (t,))
                self._db.execute("DELETE FROM pages WHERE title = ?", (t,))
        return len(gone)

    def page_count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT count(*) FROM pages").fetchone()[0]

    # --- search ----------------------------------------------------------------

    def search(self, words: list[str], k: int = 4, pool: int = 16) -> list[Hit]:
        """The best ``k`` chunks: BM25 picks ``pool``, then the ones whose title or
        heading names more of the words go first ("Trading > Mason" before a
        page that only mentions masons)."""
        if not words:
            return []
        query = " OR ".join(f'"{w}"' for w in words)
        with self._lock:
            rows = self._db.execute(
                "SELECT title, section, heading, body, rendered FROM chunks WHERE chunks MATCH ? "
                f"ORDER BY bm25(chunks, {TITLE_WEIGHT}, {HEADING_WEIGHT}, 1.0) LIMIT ?",
                (query, pool),
            ).fetchall()
        stems = {_stem(w) for w in words}
        scored = []
        for title, section, heading, body, rendered in rows:
            named = len(stems & {_stem(w) for w in re.findall(r"[a-z0-9]+", f"{title} {heading}".lower())})
            scored.append((named, Hit(title, int(section), heading, body, bool(int(rendered)), bool(named))))
        scored.sort(key=lambda x: -x[0])  # stable: BM25 order among equals
        return [h for _, h in scored[:k]]
