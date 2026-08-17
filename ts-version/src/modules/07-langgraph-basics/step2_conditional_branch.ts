import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createChatModel } from "../../llm/provider.js";

const State = Annotation.Root({
  question: Annotation<string>(),
  questionType: Annotation<string>(),
  answer: Annotation<string>(),
});

async function classifyQuestion(state: typeof State.State) {
  const model = createChatModel({ temperature: 0 });
  const response = await model.invoke([
    new HumanMessage(
      `다음 질문을 factual(사실), opinion(의견), creative(창작) 중 하나로만 분류하세요. ` +
        `단어 하나만 답하세요.\n\n질문: ${state.question}`
    ),
  ]);
  const raw = (response.content as string).trim().toLowerCase();
  const questionType = ["factual", "opinion", "creative"].find((t) => raw.includes(t)) ?? "factual";
  return { questionType };
}

/** Pure routing function — also re-run by the UI layer purely to display whether an override occurred. */
export function routeQuestion(state: { question: string; questionType: string }): "factual" | "opinion" | "creative" {
  if (state.questionType === "factual" && state.question.includes("비교")) {
    return "opinion";
  }
  return state.questionType as "factual" | "opinion" | "creative";
}

async function answerFactual(state: typeof State.State) {
  const model = createChatModel({ temperature: 0 });
  const response = await model.invoke([new HumanMessage(`사실에 기반해서 정확하게 답변하세요: ${state.question}`)]);
  return { answer: response.content as string };
}

async function answerOpinion(state: typeof State.State) {
  const model = createChatModel({ temperature: 0.7 });
  const response = await model.invoke([new HumanMessage(`균형 잡힌 의견을 제시하세요: ${state.question}`)]);
  return { answer: response.content as string };
}

async function answerCreative(state: typeof State.State) {
  const model = createChatModel({ temperature: 1.0 });
  const response = await model.invoke([new HumanMessage(`창의적으로 답변하세요: ${state.question}`)]);
  return { answer: response.content as string };
}

const graph = new StateGraph(State)
  .addNode("classify", classifyQuestion)
  .addNode("answer_factual", answerFactual)
  .addNode("answer_opinion", answerOpinion)
  .addNode("answer_creative", answerCreative)
  .addEdge(START, "classify")
  .addConditionalEdges("classify", routeQuestion, {
    factual: "answer_factual",
    opinion: "answer_opinion",
    creative: "answer_creative",
  })
  .addEdge("answer_factual", END)
  .addEdge("answer_opinion", END)
  .addEdge("answer_creative", END);

export const conditionalGraphApp = graph.compile();

export async function runConditionalGraph(question: string) {
  const result = await conditionalGraphApp.invoke({ question });
  const finalRoute = routeQuestion(result);
  return { ...result, rerouted: finalRoute !== result.questionType };
}
