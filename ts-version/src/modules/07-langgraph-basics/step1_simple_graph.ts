import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createChatModel } from "../../llm/provider.js";

const State = Annotation.Root({
  question: Annotation<string>(),
  analysis: Annotation<string>(),
  answer: Annotation<string>(),
});

async function analyzeQuestion(state: typeof State.State) {
  const model = createChatModel({ temperature: 0 });
  const response = await model.invoke([
    new HumanMessage(`다음 질문의 의도를 한 문장으로 분석하세요: "${state.question}"`),
  ]);
  return { analysis: response.content as string };
}

async function generateAnswer(state: typeof State.State) {
  const model = createChatModel();
  const response = await model.invoke([
    new HumanMessage(
      `질문: ${state.question}\n분석: ${state.analysis}\n\n위 분석을 참고해서 친절하게 답변하세요.`
    ),
  ]);
  return { answer: response.content as string };
}

const graph = new StateGraph(State)
  .addNode("analyze", analyzeQuestion)
  .addNode("generate", generateAnswer)
  .addEdge(START, "analyze")
  .addEdge("analyze", "generate")
  .addEdge("generate", END);

export const simpleGraphApp = graph.compile();

export async function runSimpleGraph(question: string) {
  return simpleGraphApp.invoke({ question });
}
