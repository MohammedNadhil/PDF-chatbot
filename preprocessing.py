# 1. PDF Processing Module (preprocessing.py)
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def _extract_text_from_page(self, page):
        text = page.get_text()
        if text.strip():  # If text exists
            return text
        
        # OCR for image-only pages
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        return pytesseract.image_to_string(img)

    def process_pdfs(self):
        all_text = []
        for pdf_path in self.pdf_paths:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = self._extract_text_from_page(page)
                all_text.append(text)
        return self.text_splitter.split_text('\n'.join(all_text))
