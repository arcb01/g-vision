"""MediaWiki API client: the whole wiki in bulk, what changed, one section rendered.

Most game wikis run MediaWiki (Fandom, wiki.gg, minecraft.wiki, bg3.wiki...),
so one client covers them. The API's address is found from any page's
``EditURI`` link, so a plain wiki URL is enough.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "G-VISION/0.1 (local game assistant; https://github.com/arcb01/g-vision)"
BATCH = 50
"""Pages per request with their wikitext: the API's limit for normal users."""
PAUSE_S = 0.5
"""Between bulk requests, to be gentle with volunteer-run wikis."""
API_PATHS = ("/api.php", "/w/api.php", "/mediawiki/api.php")
_EDIT_URI = re.compile(r'<link[^>]+rel="EditURI"[^>]+href="([^"]+?)\?action=rsd', re.I)


@dataclass
class Page:
    title: str
    revid: int
    wikitext: str


class WikiError(Exception):
    pass


class WikiClient:
    def __init__(self, wiki_url: str, client: httpx.AsyncClient | None = None, timeout: float = 20.0) -> None:
        self.wiki_url = wiki_url
        self.api: str | None = wiki_url if wiki_url.rstrip("/").endswith("api.php") else None
        self._client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                                                   headers={"User-Agent": USER_AGENT})

    async def close(self) -> None:
        await self._client.aclose()

    async def discover(self) -> str:
        """The api.php address for the wiki URL."""
        if self.api:
            return self.api
        url = self.wiki_url if "://" in self.wiki_url else f"https://{self.wiki_url}"
        try:
            r = await self._client.get(url)
            m = _EDIT_URI.search(r.text)
            if m:
                self.api = urljoin(str(r.url), m.group(1))
                return self.api
            base = f"{r.url.scheme}://{r.url.host}"
        except httpx.HTTPError:
            p = urlparse(url)
            base = f"{p.scheme}://{p.netloc}"
        for path in API_PATHS:
            try:
                r = await self._client.get(base + path, params={"action": "query", "meta": "siteinfo",
                                                                "format": "json"})
                if r.status_code == 200 and "query" in r.json():
                    self.api = base + path
                    return self.api
            except (httpx.HTTPError, ValueError):
                continue
        raise WikiError(f"no MediaWiki API found at {self.wiki_url}")

    async def _get(self, **params: object) -> dict:
        api = await self.discover()
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        for attempt in range(5):
            r = await self._client.get(api, params=params)
            if r.status_code in (429, 503) or (r.status_code == 200 and '"maxlag"' in r.text[:200]):
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                log.info("wiki: busy, waiting %.0f s", wait)
                await asyncio.sleep(min(wait, 30))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise WikiError(f"{data['error'].get('code')}: {data['error'].get('info')}")
            return data
        raise WikiError("the wiki kept saying it is busy")

    async def site_info(self) -> dict:
        data = await self._get(action="query", meta="siteinfo", siprop="general|statistics")
        q = data["query"]
        return {"sitename": q["general"].get("sitename", ""), "articles": q.get("statistics", {}).get("articles")}

    @staticmethod
    def _pages(data: dict) -> list[Page]:
        out = []
        for p in data.get("query", {}).get("pages", []):
            revs = p.get("revisions") or []
            if p.get("missing") or not revs:
                continue
            slot = revs[0].get("slots", {}).get("main", {})
            out.append(Page(p["title"], int(revs[0].get("revid", 0)), slot.get("content", "")))
        return out

    async def all_pages(self, start: str | None = None) -> AsyncIterator[tuple[list[Page], str | None]]:
        """Every article with its wikitext, about ``BATCH`` at a time, with where to resume after each."""
        batch_start = start
        cont: dict[str, str] = {"gapcontinue": start} if start else {}
        while True:
            data = await self._get(action="query", generator="allpages", gapnamespace=0,
                                   gapfilterredir="nonredirects", gaplimit=BATCH, prop="revisions",
                                   rvprop="ids|content", rvslots="main", **cont)
            nxt = data.get("continue") or {}
            # Content for one batch can come over a few responses (rvcontinue);
            # resuming then repeats the batch, which only rewrites the same pages.
            if "rvcontinue" not in nxt:
                batch_start = nxt.get("gapcontinue")
            yield self._pages(data), batch_start
            if not nxt:
                return
            cont = dict(nxt)
            await asyncio.sleep(PAUSE_S)

    async def pages_named(self, titles: list[str]) -> list[Page]:
        out: list[Page] = []
        for i in range(0, len(titles), BATCH):
            data = await self._get(action="query", titles="|".join(titles[i:i + BATCH]), prop="revisions",
                                   rvprop="ids|content", rvslots="main", redirects=1)
            out += self._pages(data)
            await asyncio.sleep(PAUSE_S)
        return out

    async def changed_since(self, ts: float) -> set[str]:
        """Articles edited or created since ``ts`` (wikis keep about 30 days of this)."""
        start = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        titles: set[str] = set()
        cont: dict[str, str] = {}
        while True:
            data = await self._get(action="query", list="recentchanges", rcnamespace=0, rctype="edit|new",
                                   rcdir="newer", rcstart=start, rcprop="title", rclimit=500, **cont)
            titles.update(c["title"] for c in data.get("query", {}).get("recentchanges", []))
            cont = data.get("continue", {})
            if "rccontinue" not in cont:
                return titles
            cont = {"rccontinue": cont["rccontinue"]}

    async def section_html(self, title: str, section: int) -> str:
        data = await self._get(action="parse", page=title, prop="text", section=section, redirects=1,
                               disableeditsection=1, disablelimitreport=1)
        return data.get("parse", {}).get("text", "")
