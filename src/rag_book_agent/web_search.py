"""Search, read and rank public web pages for the research Agent."""

import hashlib
import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from typing import Dict, List
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urlparse, urlunparse

import httpx

from rag_book_agent.config import Settings


class WebSearch:
    """A bounded web-research pipeline with independent search/read fallbacks."""

    def __init__(self, settings: Settings, reranker=None):
        self.settings = settings
        self.reranker = reranker
        self.last_provider = "off"
        self.last_trace = {}

    def search(self, question: str, limit: int = 5, force: bool = False) -> List[dict]:
        if not self.settings.web_search_enabled and not force:
            self.last_provider = "off"
            self.last_trace = {"status": "disabled"}
            return []
        candidate_limit = max(limit * self.settings.web_search_candidate_multiplier, limit)
        candidates = self._normalize_and_diversify(
            self._discover(question, candidate_limit), candidate_limit
        )
        if not candidates:
            self.last_trace = {"status": "empty", "provider": self.last_provider}
            return []
        pages = self._read_pages(candidates)
        pages_read = sum(item.get("fetch_status") == "success" for item in pages)
        selected = self._rank(question, pages)[:limit]
        self.last_trace = {
            "status": "success" if selected else "no-readable-pages",
            "provider": self.last_provider,
            "candidates": len(candidates),
            "pages_read": pages_read,
            "selected": len(selected),
        }
        return selected

    def _discover(self, question: str, limit: int) -> List[dict]:
        if self.settings.web_search_url:
            response = httpx.post(
                self.settings.web_search_url,
                json={"query": question, "limit": limit},
                timeout=self.settings.web_search_discovery_timeout_seconds,
            )
            response.raise_for_status()
            self.last_provider = "custom"
            return response.json().get("results", [])
        try:
            from ddgs import DDGS

            rows = DDGS(timeout=self.settings.web_search_discovery_timeout_seconds).text(
                question, max_results=limit
            )
            self.last_provider = "ddgs"
            return [{"title": row.get("title", ""), "url": row.get("href", ""),
                     "snippet": row.get("body", ""), "text": ""} for row in rows]
        except Exception as error:
            self.last_provider = "duckduckgo-html"
            try:
                return self._fetch_duckduckgo(question, limit)
            except Exception:
                self.last_trace = {"discovery_error": str(error)[:160]}
                return []

    def _fetch_duckduckgo(self, question: str, limit: int) -> List[dict]:
        try:
            response = httpx.get(
                "https://html.duckduckgo.com/html/", params={"q": question},
                headers={"User-Agent": self.settings.web_search_user_agent},
                timeout=self.settings.web_search_discovery_timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            html = response.text
        except (httpx.HTTPError, OSError):
            html = self._playwright_fetch(
                "https://html.duckduckgo.com/html/?q=" + quote_plus(question)
            )
            self.last_provider = "playwright-search"
        pattern = re.compile(
            r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>[\s\S]*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>', re.I
        )
        results = []
        for url, title, snippet in pattern.findall(html):
            results.append({"title": self._strip_html(title), "url": url,
                            "snippet": self._strip_html(snippet), "text": ""})
            if len(results) >= limit:
                break
        return results

    def _read_pages(self, candidates: List[dict]) -> List[dict]:
        results = []
        workers = min(self.settings.web_search_max_workers, len(candidates))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self._read_one, item): item for item in candidates}
            for future in as_completed(futures):
                item = dict(futures[future])
                try:
                    item["text"] = future.result()
                    item["fetch_status"] = "success" if item["text"] else "empty"
                except Exception as error:
                    item["text"] = ""
                    item["fetch_status"] = "failed: " + str(error)[:80]
                if not item["text"]:
                    snippet = item.get("snippet", "")
                    if self._is_usable_text(snippet, minimum=80):
                        item["text"] = snippet
                results.append(item)
        return results

    def _read_one(self, item: dict) -> str:
        url = item.get("url", "")
        self._ensure_public_url(url)
        try:
            response = httpx.get(
                url, headers={"User-Agent": self.settings.web_search_user_agent},
                timeout=self.settings.web_search_page_timeout_seconds, follow_redirects=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "html" not in content_type and "text/plain" not in content_type:
                return ""
            text = self._extract_main_text(response.text, str(response.url))
        except (httpx.HTTPError, OSError):
            html = self._playwright_fetch(url)
            text = self._extract_main_text(html, url)
        return text[: self.settings.web_search_max_page_chars].strip()

    @staticmethod
    def _extract_main_text(html: str, url: str) -> str:
        try:
            import trafilatura

            return trafilatura.extract(
                html, url=url, include_comments=False, include_tables=True,
                favor_precision=True, deduplicate=True,
            ) or ""
        except (ImportError, ValueError, TypeError):
            clean = re.sub(
                r"<(script|style|nav|footer|header)[^>]*>[\s\S]*?</\1>",
                " ", html, flags=re.I,
            )
            clean = re.sub(r"<[^>]+>", " ", clean)
            return re.sub(r"\s+", " ", unescape(clean)).strip()

    def _rank(self, question: str, items: List[dict]) -> List[dict]:
        seen_text = set()
        unique = []
        for item in items:
            text = item.get("text", "").strip()
            if not self._is_usable_text(text, minimum=80):
                continue
            digest = hashlib.sha1(
                re.sub(r"\s+", "", text[:1500]).encode("utf-8")
            ).hexdigest()
            if digest in seen_text:
                continue
            seen_text.add(digest)
            item["score"] = self._score(question, item)
            unique.append(item)
        unique.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return unique

    def _score(self, question: str, item: dict) -> float:
        text = "%s\n%s\n%s" % (
            item.get("title", ""), item.get("snippet", ""), item.get("text", "")[:1800]
        )
        if self.reranker is not None:
            try:
                return float(self.reranker.score(question, text, item.get("title", "")))
            except Exception:
                pass
        terms = set(re.findall(r"[\w\u4e00-\u9fff]+", question.lower()))
        haystack = text.lower()
        return sum(1.0 for term in terms if term in haystack) / max(1, len(terms))

    def _normalize_and_diversify(self, items: List[dict], limit: int) -> List[dict]:
        selected = []
        seen = set()
        domain_counts: Dict[str, int] = {}
        for raw in items:
            item = dict(raw)
            url = self._canonical_url(item.get("url") or item.get("href", ""))
            if not url or url in seen:
                continue
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            if domain_counts.get(domain, 0) >= self.settings.web_search_per_domain_limit:
                continue
            seen.add(url)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            item["url"] = url
            item.setdefault("snippet", item.get("body", ""))
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _canonical_url(url: str) -> str:
        url = unescape(url).strip()
        parsed = urlparse(url)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            url = unquote(parse_qs(parsed.query).get("uddg", [""])[0])
            parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ""
        query = [(key, value) for key, values in parse_qs(parsed.query).items()
                 for value in values if not key.lower().startswith(("utm_", "spm"))]
        return urlunparse((
            parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/") or "/",
            "", urlencode(query), "",
        ))

    @staticmethod
    def _ensure_public_url(url: str) -> None:
        host = urlparse(url).hostname
        if not host:
            raise ValueError("invalid URL")
        for info in socket.getaddrinfo(host, None):
            address = ipaddress.ip_address(info[4][0])
            if not address.is_global:
                raise ValueError("private or local address is not allowed")

    def _playwright_fetch(self, url: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RuntimeError("Playwright is not installed") from error
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=self.settings.web_search_user_agent)
                page.goto(url, wait_until="domcontentloaded",
                          timeout=self.settings.web_search_page_timeout_seconds * 1000)
                return page.content()
            finally:
                browser.close()

    @staticmethod
    def _strip_html(value: str) -> str:
        return re.sub(r"\s+", " ", unescape(re.sub(r"<.*?>", "", value))).strip()

    @staticmethod
    def _is_usable_text(text: str, minimum: int) -> bool:
        if len(text.strip()) < minimum:
            return False
        replacement_ratio = text.count("\ufffd") / max(1, len(text))
        return replacement_ratio < 0.01
