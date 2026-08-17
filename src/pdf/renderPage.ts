import fs from "node:fs";
import path from "node:path";
import * as mupdf from "mupdf";

const CACHE_ROOT = path.resolve(process.cwd(), "data", "pdf-pages");
const DPI = 250;
const ZOOM = DPI / 72;

/**
 * Rasterizes one PDF page to PNG, caching to disk (mirrors the Python
 * originals' `PDF_이미지/<basename>/page_N.png` cache via PyMuPDF). Uses
 * Artifex's official `mupdf` WASM binding — the same rendering engine as
 * PyMuPDF (`fitz`) — so no native build toolchain is required.
 */
export function renderPdfPageToPng(
  pdfPath: string,
  pageNumber: number // 1-indexed, matching the Python UI's page numbering
): Buffer {
  const baseName = path.basename(pdfPath, path.extname(pdfPath));
  const cacheDir = path.join(CACHE_ROOT, baseName);
  fs.mkdirSync(cacheDir, { recursive: true });
  const cachePath = path.join(cacheDir, `page_${pageNumber}.png`);

  if (fs.existsSync(cachePath)) {
    return fs.readFileSync(cachePath);
  }

  const fileBuffer = fs.readFileSync(pdfPath);
  const doc = mupdf.Document.openDocument(fileBuffer, "application/pdf");
  const page = doc.loadPage(pageNumber - 1);
  const pixmap = page.toPixmap(mupdf.Matrix.scale(ZOOM, ZOOM), mupdf.ColorSpace.DeviceRGB, false);
  const pngBytes = pixmap.asPNG();
  const pngBuffer = Buffer.from(pngBytes);

  fs.writeFileSync(cachePath, pngBuffer);
  return pngBuffer;
}

export function getPdfPageCount(pdfPath: string): number {
  const fileBuffer = fs.readFileSync(pdfPath);
  const doc = mupdf.Document.openDocument(fileBuffer, "application/pdf");
  return doc.countPages();
}
