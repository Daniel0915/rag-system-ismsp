import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { createChatModel } from "../../llm/provider.js";

export type ChatTurn = { role: "user" | "assistant"; content: string };

/**
 * Reconstructs LangChain message objects from the plain history array held in
 * the Express session — the server-side equivalent of Streamlit's
 * `st.session_state.messages` + per-rerun reconstruction.
 */
export async function replyToChat(
  history: ChatTurn[],
  systemPrompt: string,
  newMessage: string
): Promise<string> {
  const messages: BaseMessage[] = [new SystemMessage(systemPrompt)];
  for (const turn of history) {
    messages.push(turn.role === "user" ? new HumanMessage(turn.content) : new AIMessage(turn.content));
  }
  messages.push(new HumanMessage(newMessage));

  const model = createChatModel();
  const response = await model.invoke(messages);
  return response.content as string;
}
