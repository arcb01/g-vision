"""Game wiki knowledge: text extraction, the local index, lookup routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gvision.knowledge.index import WikiIndex, terms
from gvision.knowledge.library import Game, Knowledge
from gvision.knowledge.lookup import LookupTool, about_screen
from gvision.knowledge.text import chunks, html_to_text, wikitext_sections

MASON = """{{Infobox entity|image=Mason.png|health=20|behavior=Passive}}
A '''mason''' is a [[villager]] who trades [[stone]].<ref>x</ref>
== Trade offers ==
Intro to trades.
=== Mason ===
{| class="wikitable"
! Level !! Wants !! Gives
|-
| Novice || 10 [[Clay Ball]] || {{Item|Emerald}}
|-
| style="x" | Apprentice || 20 Stone || 1 Emerald
|}
[[File:Mason.png|thumb|A [[mason]]]]
[[Category:Mobs]]
== Farmer ==
Farmers want wheat.
"""


def test_wikitext_sections():
    sections = wikitext_sections(MASON)
    assert [s.index for s in sections] == [0, 1, 2, 3]
    assert [s.heading for s in sections] == ["", "Trade offers", "Trade offers > Mason", "Farmer"]
    lead = sections[0].text
    assert "health: 20" in lead and "Mason.png" not in lead
    assert "A mason is a villager who trades stone." in lead and "<ref>" not in lead
    table = sections[2].text
    assert "Novice | 10 Clay Ball | Emerald" in table
    assert "Apprentice | 20 Stone | 1 Emerald" in table
    assert "Category" not in table and "thumb" not in table


def test_html_to_text_keeps_table_rows():
    html = ('<p>Masons trade <a>stone</a>.<sup class="reference">[1]</sup></p>'
            '<table><tr><th>Level</th><th>Wants</th></tr>'
            '<tr><td>Novice</td><td>10 ×\n<a>Clay Ball</a></td></tr></table>'
            '<h3>Next<span class="mw-editsection">[edit]</span></h3>')
    text = html_to_text(html)
    assert "Masons trade stone." in text and "[1]" not in text and "[edit]" not in text
    assert "Level | Wants" in text
    assert "Novice | 10 × Clay Ball" in text


def test_chunks_split_long_text():
    parts = chunks("a" * 1000 + "\n" + "b" * 800 + "\n" + "c" * 3000, size=1200)
    assert all(len(p) <= 1200 for p in parts)
    assert "".join(parts).replace("\n", "") == "a" * 1000 + "b" * 800 + "c" * 3000


def test_terms_drop_filler_and_game_name():
    assert terms("in Minecraft the mason villager what are the main items that they want", "Minecraft") == [
        "mason", "villager", "items"]


@pytest.fixture
def index(tmp_path):
    idx = WikiIndex(tmp_path / "mc.sqlite")
    idx.replace_page("Trading", 1, wikitext_sections(MASON))
    idx.replace_page("Stone", 2, wikitext_sections("Stone is a block.\n== Uses ==\nMasons buy stone."))
    yield idx
    idx.close()


def test_search_prefers_heading_match(index):
    hits = index.search(terms("what does the mason villager want", "Minecraft"))
    assert hits[0].page == "Trading" and hits[0].heading == "Trade offers > Mason"
    assert hits[0].named
    assert index.page_count() == 2


def test_search_named_needs_a_title_word(index):
    hits = index.search(terms("which ones are worth it", ""))
    assert all(not h.named for h in hits)


def test_rendered_section_replaces_rough_text(index):
    index.set_rendered("Trading", 2, "Level | Wants | Gives\nNovice | 10 × Clay Ball | Emerald")
    hit = index.search(["mason"])[0]
    assert hit.rendered and hit.text.startswith("Level | Wants")
    # a new revision brings back the wikitext version until it is fetched again
    index.replace_page("Trading", 3, wikitext_sections(MASON))
    assert not index.search(["mason"])[0].rendered


def test_replace_page_and_prune(index):
    index.replace_page("Trading", 2, wikitext_sections("== Mason ==\nNothing now."))
    assert "Nothing now." in index.search(["mason"])[0].text
    index.prune(keep={"Trading"})
    assert index.page_count() == 1


def test_about_screen():
    assert about_screen("what does this sign say")
    assert about_screen("what's the shape inside the button on the center of the screen?")
    assert not about_screen("in Minecraft the mason villager what are the main items that they want")
    assert not about_screen("how do I craft a beacon")


class FakeWiki:
    """MediaWiki stand-in: two pages, one section rendered."""

    def __init__(self) -> None:
        self.rendered: list[tuple[str, int]] = []
        self.pages = {"Trading": MASON, "Stone": "Stone is a block."}

    async def site_info(self):
        return {"sitename": "Minecraft Wiki", "articles": 2}

    async def all_pages(self, start=None):
        for title, text in self.pages.items():
            yield [SimpleNamespace(title=title, revid=1, wikitext=text)], None

    async def pages_named(self, titles):
        return [SimpleNamespace(title=t, revid=2, wikitext=self.pages[t]) for t in titles if t in self.pages]

    async def changed_since(self, ts):
        return {"Stone"}

    async def section_html(self, title, section):
        self.rendered.append((title, section))
        return "<table><tr><td>Novice</td><td>10 Clay Ball</td><td>Emerald</td></tr></table>"

    async def close(self):
        pass


def test_knowledge_builds_and_searches(tmp_path):
    async def run():
        sent = []
        wiki = FakeWiki()
        k = Knowledge(tmp_path, sent.append, client=lambda game: wiki)
        await k.set_game(Game(id="minecraft", name="Minecraft", wiki="https://minecraft.wiki"))
        await k.wait()
        assert sent[-1].state == "ready" and sent[-1].pages == 2 and sent[-1].updated_ts
        found = await k.search("what does the mason villager want")
        assert found and found[0].page == "Trading" and found[0].text.startswith("Novice | 10 Clay Ball")
        assert wiki.rendered == [("Trading", 2)]
        await k.search("what does the mason villager want")  # cached now
        assert wiki.rendered == [("Trading", 2)]
        assert (tmp_path / "minecraft.json").exists()
        await k.update(full=False)
        await k.wait()
        assert sent[-1].state == "ready"
        await k.set_game(None)
        assert sent[-1].state == "off"
        assert await k.search("mason") == []

    asyncio.run(run())


class FakeQwen:
    def __init__(self) -> None:
        self.messages = None

    async def chat(self, messages, **kwargs):
        self.messages = messages
        return SimpleNamespace(content="Masons want clay balls and stone for emeralds.", raw={})


def test_lookup_answers_from_the_wiki(tmp_path):
    async def run():
        wiki = FakeWiki()
        k = Knowledge(tmp_path, lambda m: None, client=lambda game: wiki)
        qwen = FakeQwen()
        tool = LookupTool(qwen, k)
        assert "error" in (await tool("what does the mason want")).content  # no game yet
        await k.set_game(Game(id="minecraft", name="Minecraft", wiki="https://minecraft.wiki"))
        await k.wait()
        result = await tool("what does the mason villager want")
        assert result.speak.startswith("Masons want")
        assert result.content["pages"] == ["Trading > Mason"]
        prompt = qwen.messages[0]["content"]
        assert "Minecraft" in prompt and "Novice | 10 Clay Ball" in prompt
        assert "error" in (await tool("is it worth it")).content  # nothing the wiki names

    asyncio.run(run())


def test_no_game_means_no_wiki(tmp_path):
    k = Knowledge(tmp_path, lambda m: None)
    assert k.status().state == "off"


def fake_mediawiki():
    """httpx transport answering like a MediaWiki with three articles, two per batch."""
    import httpx

    articles = {"Apple": "An apple.", "Mason": "== Trades ==\nClay.", "Stone": "Rock."}
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        p = dict(request.url.params)
        calls.append(p)
        if request.url.path == "/wiki/Main_Page":
            return httpx.Response(200, text='<link rel="EditURI" type="application/rsd+xml" '
                                             'href="//game.example/w/api.php?action=rsd"/>')
        assert request.url.path == "/w/api.php" and p["format"] == "json"
        if p.get("meta") == "siteinfo":
            return httpx.Response(200, json={"query": {"general": {"sitename": "Game Wiki"},
                                                       "statistics": {"articles": 3}}})
        if p.get("generator") == "allpages":
            titles = sorted(articles)
            start = titles.index(p["gapcontinue"]) if "gapcontinue" in p else 0
            batch = titles[start:start + 2]
            data = {"query": {"pages": [{"title": t, "revisions": [{"revid": 7, "slots": {"main": {
                "content": articles[t]}}}]} for t in batch]}}
            if start + 2 < len(titles):
                data["continue"] = {"gapcontinue": titles[start + 2], "continue": "gapcontinue||"}
            return httpx.Response(200, json=data)
        if p.get("list") == "recentchanges":
            return httpx.Response(200, json={"query": {"recentchanges": [{"title": "Mason"}, {"title": "Mason"}]}})
        if p.get("titles"):
            return httpx.Response(200, json={"query": {"pages": [{"title": t, "revisions": [{"revid": 8, "slots": {
                "main": {"content": articles[t]}}}]} for t in p["titles"].split("|")]}})
        if p.get("action") == "parse":
            return httpx.Response(200, json={"parse": {"text": "<p>Masons want clay.</p>"}})
        return httpx.Response(404)

    return httpx.MockTransport(handle), calls


def test_wiki_client_against_a_mediawiki(monkeypatch):
    import httpx

    from gvision.knowledge import wiki as wiki_mod
    from gvision.knowledge.wiki import WikiClient

    monkeypatch.setattr(wiki_mod, "PAUSE_S", 0)
    transport, calls = fake_mediawiki()

    async def run():
        c = WikiClient("https://game.example/wiki/Main_Page", client=httpx.AsyncClient(transport=transport))
        assert await c.discover() == "https://game.example/w/api.php"
        assert await c.site_info() == {"sitename": "Game Wiki", "articles": 3}
        batches = [(sorted(p.title for p in pages), nxt) async for pages, nxt in c.all_pages()]
        assert batches == [(["Apple", "Mason"], "Stone"), (["Stone"], None)]
        assert await c.changed_since(0) == {"Mason"}
        assert [p.revid for p in await c.pages_named(["Mason"])] == [8]
        assert "Masons want clay" in await c.section_html("Mason", 1)
        await c.close()

    asyncio.run(run())
