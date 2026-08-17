import { z } from "zod";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createChatModel, generateStructured } from "../../llm/provider.js";

const State = Annotation.Root({
  topic: Annotation<string>(),
  draft: Annotation<string>(),
  feedback: Annotation<string>(),
  score: Annotation<number>(),
  iteration: Annotation<number>(),
  maxIterations: Annotation<number>(),
});

const EvaluationSchema = z.object({
  score: z.number().min(1).max(10).describe("1~10점 사이의 글 품질 점수"),
  feedback: z.string().describe("개선을 위한 구체적인 피드백"),
});

async function writeDraft(state: typeof State.State) {
  const iteration = (state.iteration ?? 0) + 1;
  const model = createChatModel({ temperature: 0.8 });
  const prompt =
    iteration === 1
      ? `다음 주제로 짧은 에세이(3~4문장)를 작성하세요: ${state.topic}`
      : `다음 초안을 피드백을 반영해서 다시 작성하세요.\n\n초안: ${state.draft}\n\n피드백: ${state.feedback}`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { draft: response.content as string, iteration };
}

async function evaluateDraft(state: typeof State.State) {
  const model = createChatModel({ temperature: 0 });
  const result = await generateStructured(model, EvaluationSchema, [
    new HumanMessage(`다음 글을 1~10점으로 평가하고 피드백을 주세요.\n\n주제: ${state.topic}\n\n글: ${state.draft}`),
  ]);
  return { score: result.score, feedback: result.feedback };
}

function shouldContinue(state: typeof State.State): "continue" | "end" {
  if (state.score < 7 && state.iteration < state.maxIterations) return "continue";
  return "end";
}

const graph = new StateGraph(State)
  .addNode("write", writeDraft)
  .addNode("evaluate", evaluateDraft)
  .addEdge(START, "write")
  .addEdge("write", "evaluate")
  .addConditionalEdges("evaluate", shouldContinue, { continue: "write", end: END });

export const loopGraphApp = graph.compile();

export type LoopIteration = { iteration: number; draft: string; score: number; feedback: string };

/** Streams node updates (rather than a single invoke) to reconstruct per-iteration history for the UI. */
export async function runLoopGraph(topic: string, maxIterations: number): Promise<LoopIteration[]> {
  const history: LoopIteration[] = [];
  const stream = await loopGraphApp.stream(
    { topic, draft: "", feedback: "", score: 0, iteration: 0, maxIterations },
    { streamMode: "updates" }
  );
  let latest: Partial<typeof State.State> = {};
  for await (const update of stream) {
    for (const nodeUpdate of Object.values(update)) {
      latest = { ...latest, ...(nodeUpdate as Partial<typeof State.State>) };
    }
    if ("score" in latest) {
      history.push({
        iteration: latest.iteration!,
        draft: latest.draft!,
        score: latest.score!,
        feedback: latest.feedback!,
      });
    }
  }
  return history;
}
