"""Wiki pages as plain text: wikitext for the bulk index, HTML for live sections.

The bulk index reads wikitext, 50 pages a request, so a whole wiki fits in
minutes. Wikitext keeps tables and infoboxes as templates, so templates are
flattened to their values ("{{Trade|10|Clay Ball|1|Emerald}}" reads
"10 Clay Ball 1 Emerald"): rough, but the words are there to be found. When a
section is used to answer, its rendered HTML is fetched once and replaces the
rough text (``html_to_text``), tables as one row per line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

CHUNK_CHARS = 1200
"""Long sections are split into chunks of about this size, at paragraph breaks."""

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_DROP_TAGS = re.compile(r"<(gallery|math|score|syntaxhighlight|templatedata|nowiki)[^>]*>.*?</\1>", re.S | re.I)
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_LINK = re.compile(r"\[\[([^\[\]]*)\]\]")
_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")
_EXTERNAL = re.compile(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]")
_BEHAVIOR = re.compile(r"__[A-Z]+__")
_HEADING = re.compile(r"^(={1,6})\s*(.*?)\s*\1\s*$")
_FILE_VALUE = re.compile(r"\.(png|gif|jpe?g|svg|webp|ogg|mp3|mp4|webm)\b", re.I)
_SKIP_NS = ("file:", "image:", "category:", "media:")
_SKIP_PARAMS = {"image", "image2", "imagesize", "caption", "file", "icon", "size", "width", "height", "style",
                "class", "align", "link", "alt", "upright", "float"}


@dataclass
class Section:
    index: int
    """MediaWiki's section number (0 is the lead), for fetching it rendered."""
    heading: str
    """"Trade offers > Mason"; empty for the lead."""
    text: str


def _link(m: re.Match) -> str:
    inner = m.group(1)
    target, _, label = inner.partition("|")
    if target.strip().lower().startswith(_SKIP_NS) or target.startswith(":category:"):
        return ""
    return (label or target).strip()


def _template(m: re.Match) -> str:
    parts = m.group(1).split("|")
    values = []
    for p in parts[1:]:
        key, eq, value = p.partition("=")
        if eq and re.fullmatch(r"\s*[\w ]{1,30}\s*", key):
            name, value = key.strip(), value.strip()
            if not value or name.lower() in _SKIP_PARAMS or _FILE_VALUE.search(value):
                continue
            values.append(f"{name.replace('_', ' ')}: {value}" if not name.isdigit() else value)
        else:
            value = p.strip()
            if value and not _FILE_VALUE.search(value):
                values.append(value)
    return " ".join(values)


def _flatten(text: str) -> str:
    text = _COMMENT.sub("", text)
    text = _REF.sub("", text)
    text = _DROP_TAGS.sub("", text)
    for _ in range(10):  # innermost links and templates first
        text, n = _LINK.subn(_link, text)
        if not n:
            break
    for _ in range(20):
        text, n = _TEMPLATE.subn(_template, text)
        if not n:
            break
    text = _EXTERNAL.sub(r"\1", text)
    text = _BEHAVIOR.sub("", text)
    text = _TAG.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    return _tables(text)


def _tables(text: str) -> str:
    """Wikitable markup to one row per line, cells separated by " | "."""
    out: list[str] = []
    row: list[str] = []

    def end_row() -> None:
        if row:
            out.append(" | ".join(c for c in row if c))
            row.clear()

    in_table = 0
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("{|"):
            in_table += 1
            continue
        if in_table and s.startswith("|}"):
            end_row()
            in_table -= 1
            continue
        if in_table:
            if s.startswith("|-") or s.startswith("|+"):
                end_row()
                if s.startswith("|+"):
                    out.append(s[2:].strip())
                continue
            if s[:1] in ("|", "!"):
                for cell in re.split(r"\|\||!!", s[1:]):
                    # "style=... | value": keep the value
                    value = cell.split("|")[-1] if "=" in cell.split("|")[0] and "|" in cell else cell
                    row.append(value.strip())
                continue
            if row:
                row[-1] = f"{row[-1]} {s}".strip()
                continue
        out.append(line)
    end_row()
    return "\n".join(out)


def tidy(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [re.sub(r"^[*#:;]+\s*", "- ", line) for line in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def wikitext_sections(wikitext: str) -> list[Section]:
    """A page's sections as plain text, numbered as MediaWiki numbers them."""
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    index, heading, body = 0, "", []

    def flush() -> None:
        text = tidy(_flatten("\n".join(body)))
        if text:
            sections.append(Section(index, heading, text))

    for line in wikitext.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), tidy(_flatten(m.group(2)))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            index += 1
            heading, body = " > ".join(t for _, t in path), []
        else:
            body.append(line)
    flush()
    return sections


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split at paragraph (then line) breaks into pieces of about ``size`` characters."""
    if len(text) <= size:
        return [text]
    out: list[str] = []
    current = ""
    for para in re.split(r"(?<=\n)", text):
        while len(para) > size:  # one huge table or paragraph
            cut = para.rfind("\n", 0, size)
            cut = cut if cut > size // 2 else size
            if current:
                out.append(current)
                current = ""
            out.append(para[:cut])
            para = para[cut:]
        if len(current) + len(para) > size and current:
            out.append(current)
            current = ""
        current += para
    if current.strip():
        out.append(current)
    return [c.strip() for c in out if c.strip()]


class _HtmlText(HTMLParser):
    SKIP = {"script", "style", "noscript"}
    SKIP_CLASSES = ("mw-editsection", "reference", "navbox", "toc", "noprint", "mw-references-wrap", "metadata")
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "caption", "ul", "ol", "dl"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip: list[str] = []
        self._cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if self._skip:
            if tag not in ("br", "img", "hr", "wbr", "input", "meta", "link"):
                self._skip.append(tag)
            return
        classes = a.get("class") or ""
        if tag in self.SKIP or any(c in classes.split() for c in self.SKIP_CLASSES) or tag == "sup" and "reference" in classes:
            self._skip.append(tag)
            return
        if tag == "tr":
            self.out.append("\n")
            self._cell = False
        elif tag in ("td", "th"):
            if self._cell:
                self.out.append(" | ")
            self._cell = True
        elif tag == "br":
            self.out.append(" " if self._cell else "\n")
        elif tag in self.BLOCK:
            self.out.append("\n")
            if tag == "li":
                self.out.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if self._skip:
            if self._skip[-1] == tag:
                self._skip.pop()
            return
        if tag == "table":
            self.out.append("\n")
            self._cell = False
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.out.append(data.replace("\n", " ") if self._cell else data)


def html_to_text(html: str) -> str:
    p = _HtmlText()
    p.feed(html)
    p.close()
    text = "".join(p.out).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip(" |") for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
