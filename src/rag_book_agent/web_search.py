"""Optional web-search adapter. DeepSeek Chat API itself does not expose web search."""

import re
from typing import List

import httpx

from rag_book_agent.config import Settings


class WebSearch:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_provider = "off"

    def search(self, question: str, limit: int = 5, force: bool = False) -> List[dict]:
        if not self.settings.web_search_enabled and not force:
            self.last_provider = "off"
            return []
        if not self.settings.web_search_url and self.settings.web_search_provider == "duckduckgo":
            return self._fetch_duckduckgo(question, limit)
        if not self.settings.web_search_url:
            return []
        response = httpx.post(
            self.settings.web_search_url,
            json={"query": question, "limit": limit},
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        return response.json().get("results", [])

    def _fetch_duckduckgo(self, question: str, limit: int) -> List[dict]:
        try:
            response = httpx.get(
                "https://html.duckduckgo.com/html/",
                params={"q": question},
                headers={"User-Agent": "rag-book-agent/1.0"},
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            html = response.text
            self.last_provider = "fetch"
        except (httpx.HTTPError, OSError):
            html = self._playwright_fetch(question)
            self.last_provider = "playwright"
        pattern = re.compile(r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', re.S)
        results = []
        for url, title in pattern.findall(html)[:limit]:
            results.append({"title": re.sub(r"<.*?>", "", title), "url": url, "text": ""})
        return results

    @staticmethod
    def _playwright_fetch(question: str) -> str:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent="rag-book-agent/1.0")
            page.goto(
                "https://html.duckduckgo.com/html/?q=" + question,
                wait_until="domcontentloaded",
            )
            html = page.content()
            browser.close()
            return html
