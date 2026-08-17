import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const ARTICLE_RE = /제\s*\d+\s*조(?:의\s*\d+)?/g;
const MIN_ARTICLES_FOR_ARTICLE_SPLIT = 3;

/**
 * 소기업 정책 문서는 "제N조" 형식을 따르지 않는 경우가 많다 — 전체 텍스트에서
 * 조문 마커 밀도를 먼저 검사하고, 충분하면 조 단위로, 아니면 일반 재귀 분할로
 * 폴백한다. 각 청크에 chunk_strategy를 태깅해 어느 경로를 탔는지 추적한다.
 */
export async function chunkPolicyDoc(docs: Document[]): Promise<Document[]> {
  const fullText = docs.map((d) => d.pageContent).join("\n");
  const articleCount = (fullText.match(ARTICLE_RE) ?? []).length;

  if (articleCount < MIN_ARTICLES_FOR_ARTICLE_SPLIT) {
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const chunks = await splitter.splitDocuments(docs);
    return chunks.map((c) => new Document({ ...c, metadata: { ...c.metadata, chunk_strategy: "recursive" } }));
  }

  const articleChunks: Document[] = [];
  const subSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });

  for (const doc of docs) {
    const matches = [...doc.pageContent.matchAll(ARTICLE_RE)];
    if (matches.length === 0) {
      // This page has no article marker of its own (e.g. a preamble page) —
      // keep it as one chunk rather than silently dropping it.
      articleChunks.push(
        new Document({ pageContent: doc.pageContent, metadata: { ...doc.metadata, chunk_strategy: "article" } })
      );
      continue;
    }

    for (let i = 0; i < matches.length; i++) {
      const start = matches[i].index ?? 0;
      const end = i + 1 < matches.length ? (matches[i + 1].index ?? doc.pageContent.length) : doc.pageContent.length;
      const articleText = doc.pageContent.slice(start, end).trim();
      if (!articleText) continue;

      if (articleText.length <= 1200) {
        articleChunks.push(
          new Document({ pageContent: articleText, metadata: { ...doc.metadata, chunk_strategy: "article" } })
        );
      } else {
        const subChunks = await subSplitter.splitDocuments([
          new Document({ pageContent: articleText, metadata: { ...doc.metadata, chunk_strategy: "article" } }),
        ]);
        articleChunks.push(...subChunks);
      }
    }
  }

  return articleChunks;
}
