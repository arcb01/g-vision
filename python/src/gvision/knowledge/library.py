"""The session game's wiki: kept on disk, caught up in the background, searched per question.

When the panel starts a session for a game, its wiki is downloaded once into
``data/wiki/<game>.sqlite`` (every article, about 50 a request) and kept
there. Questions search that copy, so answers don't wait on the network. A
section used to answer is fetched rendered once, when online, which is cleaner
than the bulk wikitext, and kept too. The index catches up with the wiki's
edits when a session starts and it is more than a week old, or when the panel
asks; ``<game>.json`` next to it tells the panel how fresh it is.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from gvision.knowledge.index import WikiIndex, terms
from gvision.knowledge.text import html_to_text, wikitext_sections
from gvision.knowledge.wiki import WikiClient, WikiError
from gvision.protocol import Game, WikiStatusMsg

log = logging.getLogger(__name__)

STALE_S = 7 * 86400
"""Catch up with the wiki's edits when a session starts and the index is older than this."""
RECENT_CHANGES_S = 25 * 86400
"""Wikis list about 30 days of edits; an older index is downloaded again in full."""
LIVE_TIMEOUT_S = 2.0
"""How long a question waits for a section to come back rendered before using the stored text."""
MAX_SNIPPETS = 2
MAX_SNIPPET_CHARS = 1400


@dataclass
class Snippet:
    where: str
    """"Trading > Mason"."""
    text: str

    @property
    def page(self) -> str:
        return self.where.split(" > ")[0]


class Knowledge:
    def __init__(self, data_dir: Path | str, send: Callable[[WikiStatusMsg], Any],
                 client: Callable[[Game], Any] | None = None) -> None:
        self.dir = Path(data_dir)
        self._send = send
        self._client_for = client or (lambda game: WikiClient(game.wiki))
        self.game: Game | None = None
        self.index: WikiIndex | None = None
        self.wiki: Any = None
        self._task: asyncio.Task | None = None
        self._state = "off"
        self._error: str | None = None
        self._total: int | None = None

    # --- status ----------------------------------------------------------------

    def status(self) -> WikiStatusMsg:
        idx = self.index
        updated = idx.get("updated_ts") if idx else None
        return WikiStatusMsg(
            game_id=self.game.id if self.game else None, state=self._state,
            pages=idx.page_count() if idx else 0, total=self._total,
            updated_ts=float(updated) if updated else None, source=idx.get("source") if idx else None,
            error=self._error,
        )

    def publish(self) -> None:
        self._send(self.status())

    def _set_state(self, state: str, error: str | None = None) -> None:
        self._state, self._error = state, error
        self.publish()

    def _sidecar(self) -> None:
        """What the panel's game picker shows: pages and when they were last caught up."""
        s = self.status()
        path = self.dir / f"{self.game.id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"game": self.game.id, "name": self.game.name, "wiki": self.game.wiki,
                                   "source": s.source, "pages": s.pages, "updated_ts": s.updated_ts}),
                       encoding="utf-8")
        tmp.replace(path)

    # --- session game ----------------------------------------------------------

    async def set_game(self, game: Game | None) -> None:
        if game and self.game and (game.id, game.wiki) == (self.game.id, self.game.wiki):
            self.publish()
            return
        await self._stop()
        self.game = game
        self._total = None
        if game is None:
            self._set_state("off")
            return
        try:
            self.index = await asyncio.to_thread(WikiIndex, self.dir / f"{game.id}.sqlite")
        except RuntimeError as e:
            log.error("wiki: %s", e)
            self._set_state("error", str(e))
            return
        self.wiki = self._client_for(game)
        if self.index.get("wiki") not in (None, game.wiki):  # another wiki for this game: start over
            self.index.set(complete=0, resume="", wiki=game.wiki)
        age = time.time() - float(self.index.get("updated_ts") or 0)
        if self.index.get("complete") != "1":
            self._start(self._build())
        elif age > STALE_S:
            self._start(self._update(full=age > RECENT_CHANGES_S))
        else:
            self._set_state("ready")

    async def update(self, full: bool = False) -> None:
        """Catch up with the wiki now (the panel's Update button)."""
        if not self.game or not self.index or (self._task and not self._task.done()):
            return
        age = time.time() - float(self.index.get("updated_ts") or 0)
        if self.index.get("complete") != "1":
            self._start(self._build())
        else:
            self._start(self._update(full=full or age > RECENT_CHANGES_S))

    async def wait(self) -> None:
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def close(self) -> None:
        await self._stop()

    def _start(self, job) -> None:
        self._task = asyncio.create_task(self._run(job))

    async def _run(self, job) -> None:
        try:
            await job
        except asyncio.CancelledError:
            raise
        except Exception as e:  # network, a wiki that answers oddly, the disk: say so in the panel
            log.warning("wiki: %s failed: %s", self.game.id if self.game else "?", e, exc_info=not isinstance(
                e, (httpx.HTTPError, WikiError)))
            self._set_state("error", f"{type(e).__name__}: {e}" if not isinstance(e, WikiError) else str(e))

    async def _stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._task = None
        if self.wiki is not None:
            with contextlib.suppress(Exception):
                await self.wiki.close()
            self.wiki = None
        if self.index is not None:
            await asyncio.to_thread(self.index.close)
            self.index = None

    # --- keeping the index -----------------------------------------------------

    def _store(self, pages: list) -> None:
        for p in pages:
            self.index.replace_page(p.title, p.revid, wikitext_sections(p.wikitext))

    async def _build(self) -> None:
        """Download every article; picks up where an interrupted download stopped."""
        self._set_state("indexing")
        info = await self.wiki.site_info()
        self._total = info.get("articles")
        self.index.set(source=info.get("sitename") or self.game.name, wiki=self.game.wiki)
        resume = self.index.get("resume") or None
        started = time.time()
        seen: set[str] = set()
        async for pages, nxt in self.wiki.all_pages(resume):
            await asyncio.to_thread(self._store, pages)
            seen.update(p.title for p in pages)
            self.index.set(resume=nxt or "")
            self.publish()
        if resume is None:  # saw the whole wiki: drop deleted pages
            await asyncio.to_thread(self.index.prune, seen)
        self.index.set(complete=1, resume="", updated_ts=started)
        self._total = None
        self._sidecar()
        log.info("wiki: %s indexed, %d pages", self.game.id, self.index.page_count())
        self._set_state("ready")

    async def _update(self, full: bool) -> None:
        if full:
            self.index.set(complete=0, resume="")
            await self._build()
            return
        self._set_state("updating")
        started = time.time()
        titles = await self.wiki.changed_since(float(self.index.get("updated_ts") or 0))
        pages = await self.wiki.pages_named(sorted(titles)) if titles else []
        await asyncio.to_thread(self._store, pages)
        self.index.set(updated_ts=started)
        self._sidecar()
        log.info("wiki: %s caught up, %d pages changed", self.game.id, len(pages))
        self._set_state("ready")

    # --- questions -------------------------------------------------------------

    async def search(self, question: str, k: int = MAX_SNIPPETS) -> list[Snippet]:
        """Sections about what the question names, rendered when the wiki answers in time."""
        if not self.index or not self.game:
            return []
        index, wiki = self.index, self.wiki
        hits = await asyncio.to_thread(index.search, terms(question, self.game.name), 6)
        picked, seen = [], set()
        for h in hits:
            if h.named and (h.page, h.section) not in seen:
                seen.add((h.page, h.section))
                picked.append(h)
        out = []
        for h in picked[:k]:
            text = h.text
            if not h.rendered and wiki is not None:
                try:
                    html = await asyncio.wait_for(wiki.section_html(h.page, h.section), LIVE_TIMEOUT_S)
                    rendered = html_to_text(html)
                    if rendered:
                        await asyncio.to_thread(index.set_rendered, h.page, h.section, rendered)
                        text = rendered
                except (httpx.HTTPError, WikiError, asyncio.TimeoutError, ValueError) as e:
                    log.info("wiki: using the stored text of %s: %s", h.where, e)
            out.append(Snippet(h.where, text[:MAX_SNIPPET_CHARS]))
        return out
