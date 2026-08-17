import fs from "node:fs";
import path from "node:path";
import multer from "multer";

/** Persists uploads under data/uploads/<namespace>/<original filename> so PDF page rendering can re-read them later. */
export function createUpload(namespace: string) {
  const dir = path.resolve(process.cwd(), "data", "uploads", namespace);
  fs.mkdirSync(dir, { recursive: true });
  const storage = multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, dir),
    filename: (_req, file, cb) => cb(null, file.originalname),
  });
  return multer({ storage });
}

export function uploadDir(namespace: string): string {
  return path.resolve(process.cwd(), "data", "uploads", namespace);
}
