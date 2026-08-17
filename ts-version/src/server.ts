import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import session from "express-session";
import { env } from "./config/env.js";

import module00Router from "./routes/00-openai-basics.js";
import module01Router from "./routes/01-langchain-basics.js";
import module02Router from "./routes/02-chatbot.js";
import module03Router from "./routes/03-single-pdf-rag.js";
import module04Router from "./routes/04-multi-pdf-rag.js";
import module05Router from "./routes/05-metadata-filtered-rag.js";
import module06Router from "./routes/06-rag-multi-format.js";
import module07Router from "./routes/07-langgraph-basics.js";

declare module "express-session" {
  interface SessionData {
    chatHistoryBasic?: { role: "user" | "assistant"; content: string }[];
    chatHistoryPersona?: { role: "user" | "assistant"; content: string }[];
    currentPersona?: string;
  }
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(express.json());
app.use(
  session({
    secret: env.sessionSecret,
    resave: false,
    saveUninitialized: true,
  })
);

app.use(express.static(path.join(__dirname, "..", "public")));
// Serve rasterized PDF page images and the shared /samples corpus for the RAG modules.
app.use("/data/pdf-pages", express.static(path.resolve(process.cwd(), "data", "pdf-pages")));
app.use("/samples", express.static(path.resolve(process.cwd(), "..", "samples")));

app.use("/api/00-openai-basics", module00Router);
app.use("/api/01-langchain-basics", module01Router);
app.use("/api/02-chatbot", module02Router);
app.use("/api/03-single-pdf-rag", module03Router);
app.use("/api/04-multi-pdf-rag", module04Router);
app.use("/api/05-metadata-filtered-rag", module05Router);
app.use("/api/06-rag-multi-format", module06Router);
app.use("/api/07-langgraph-basics", module07Router);

app.listen(env.port, () => {
  console.log(`RAG TS server running at http://localhost:${env.port}`);
});
