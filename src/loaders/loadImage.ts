import fs from "node:fs";
import path from "node:path";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import { createChatModel, generateStructured } from "../llm/provider.js";
import { env } from "../config/env.js";

const MIME_BY_EXT: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".gif": "image/gif",
  ".bmp": "image/bmp",
  ".webp": "image/webp",
};

const ImageAnalysisSchema = z.object({
  extracted_text: z.string().describe("이미지 안에 실제로 적힌 텍스트를 그대로 옮겨적은 것 (없으면 빈 문자열)"),
  description: z.string().describe("이미지의 시각적 내용에 대한 설명 (표, 도장, 사진, 레이아웃 등)"),
});

/**
 * Gemini is natively multimodal, so unlike the Python original (which called
 * a separate GPT vision model), this reuses the same chat model used
 * everywhere else in the app.
 */
export async function loadImageAsDocument(filePath: string, filename: string): Promise<Document> {
  if (env.llmProvider === "ollama") {
    throw new Error(
      `이미지 OCR(${filename})은 현재 로컬 모델(${env.ollamaChatModel})에서 지원되지 않습니다. ` +
        "텍스트 전용 모델이라 이미지를 처리할 수 없어요 — 이미지 분석이 필요하면 LLM_PROVIDER=gemini로 전환하거나, " +
        "Ollama에서 qwen2.5vl/llava 같은 비전 모델을 pull 받아 OLLAMA_CHAT_MODEL을 바꿔주세요."
    );
  }
  const ext = path.extname(filename).toLowerCase();
  const mime = MIME_BY_EXT[ext] ?? "application/octet-stream";
  const base64 = fs.readFileSync(filePath).toString("base64");

  const model = createChatModel({ temperature: 0 });

  const message = new HumanMessage({
    content: [
      {
        type: "text",
        text:
          "이 이미지를 분석하세요. 이미지 안에 텍스트가 있다면 그대로 추출하고(extracted_text), " +
          "표/도장/사진/레이아웃 등 시각적 내용을 설명하세요(description). 한국어로 답변하세요.",
      },
      {
        type: "image_url",
        image_url: `data:${mime};base64,${base64}`,
      },
    ],
  });

  const result = await generateStructured(model, ImageAnalysisSchema, [message]);
  const pageContent = result.extracted_text?.trim() || result.description;

  return new Document({
    pageContent,
    metadata: {
      source_file: filename,
      file_type: "image",
      ocr_engine: "gemini-vision",
      extracted_text: result.extracted_text,
      description: result.description,
    },
  });
}

export function isImageFile(filename: string): boolean {
  return Object.prototype.hasOwnProperty.call(MIME_BY_EXT, path.extname(filename).toLowerCase());
}
