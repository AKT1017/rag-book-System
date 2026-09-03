"""Lazy RapidOCR adapter for pages without a usable text layer."""

from typing import Any


class RapidPdfReader:
    _pipeline = None
    _formula_enabled = False
    _table_enabled = False
    _engine = "text"

    def __init__(self, dpi: int = 220, formula: bool = True, tables: bool = True):
        self.dpi = dpi
        self.formula = formula
        self.tables = tables

    @classmethod
    def _get_pipeline(cls):
        if cls._pipeline is None:
            from rapidocr_onnxruntime import RapidOCR
            cls._pipeline = RapidOCR()
        return cls._pipeline

    def read_page(self, pdf_page) -> tuple[str, float, int]:
        import cv2
        import numpy as np
        scale = self.dpi / 72.0
        pixmap = pdf_page.get_pixmap(matrix=__import__("fitz").Matrix(scale, scale), alpha=False)
        image = pixmap.tobytes("png")
        image_array = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_array is None:
            raise RuntimeError("无法将 PDF 页面渲染为 OCR 图像")
        type(self)._formula_enabled = self.formula
        type(self)._table_enabled = self.tables
        result, _ = self._get_pipeline()(image_array)
        text, confidence, table_count = self._extract(result)
        return text.strip(), confidence, table_count

    def _extract(self, result: Any) -> tuple[str, float, int]:
        texts = []
        scores = []
        tables = 0
        # RapidOCR returns a list of [box, text, confidence] records.
        for page in result or []:
            if isinstance(page, list):
                for line in page:
                    try:
                        text_value, score = line[1], line[2]
                        texts.append(str(text_value).strip())
                        scores.append(float(score))
                    except (IndexError, TypeError, ValueError):
                        pass
                continue
        for item in result or []:
            data = item.json if hasattr(item, "json") else item
            if callable(data):
                data = data()
            if isinstance(data, dict):
                markdown = data.get("markdown") or data.get("markdown_text")
                if isinstance(markdown, str) and markdown.strip():
                    texts.append(markdown)
                self._walk(data, texts, scores)
                tables += sum(1 for value in data.values() if isinstance(value, str) and "<table" in value)
            elif isinstance(data, str) and data.strip():
                texts.append(data)
        confidence = sum(scores) / len(scores) if scores else (1.0 if texts else 0.0)
        unique = []
        for value in texts:
            if value not in unique:
                unique.append(value)
        return "\n\n".join(unique), confidence, tables

    def _walk(self, value: Any, texts: list[str], scores: list[float]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"text", "rec_text", "latex", "formula"} and isinstance(item, str) and item.strip():
                    texts.append(item.strip())
                elif key in {"score", "confidence", "rec_score"} and isinstance(item, (int, float)):
                    scores.append(float(item))
                else:
                    self._walk(item, texts, scores)
        elif isinstance(value, list):
            for item in value:
                self._walk(item, texts, scores)


# Compatibility alias for integrations written before the OCR engine migration.
PaddlePdfReader = RapidPdfReader
