import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import {
  ChatStateAdapterProvider,
  useChatStateAdapter,
} from "@/features/chat/ChatStateAdapter";
import { StreamingStatus } from "@/features/chat/trace";
import { initI18n } from "@/i18n/init";

initI18n("en");
const transport = vi.hoisted(() => ({
  emit: undefined as
    undefined | ((event: Record<string, unknown>) => void),
}));
vi.mock("@/features/chat/transport/UnifiedTurnClient", () => ({
  UnifiedTurnClient: class {
    connected = true;
    constructor(emit: typeof transport.emit) {
      transport.emit = emit;
    }
    connect() {}
    disconnect() {}
    send() {}
    setResumeState() {}
  },
}));
function Harness() {
  const { state, sendMessage, regenerateLastMessage } =
    useChatStateAdapter();
  const message = state.messages.at(-1);
  return (
    <>
      <button onClick={() => sendMessage("Hello")}>Send</button>
      <button onClick={() => regenerateLastMessage()}>Retry</button>
      {message?.role === "assistant" && (
        <StreamingStatus
          events={message.events ?? []}
          traceBounds={message.trace}
          content={message.content}
          isStreaming={state.isStreaming}
        />
      )}
    </>
  );
}
afterEach(() => vi.useRealTimers());
it("counts submission latency before the first frame, preserves it on completion, and resets on retry", async () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-09-15T00:00:00Z"));
  render(
    <ChatStateAdapterProvider>
      <Harness />
    </ChatStateAdapterProvider>,
  );
  fireEvent.click(screen.getByText("Send"));
  expect(screen.getByRole("status")).toHaveTextContent("0s");
  await act(async () => {
    await vi.advanceTimersByTimeAsync(120_000);
  });
  expect(screen.getByRole("status")).toHaveTextContent("2m");
  await act(async () =>
    transport.emit?.({
      type: "session",
      session_id: "session-clock",
      turn_id: "turn-clock",
      metadata: {},
    }),
  );
  await act(async () =>
    transport.emit?.({
      type: "content",
      source: "chat",
      stage: "responding",
      content: "Hello",
      turn_id: "turn-clock",
      seq: 1,
      timestamp: Date.now() / 1000,
      metadata: {},
    }),
  );
  expect(screen.getByRole("status")).toHaveTextContent("2m");
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1000);
    transport.emit?.({
      type: "done",
      source: "chat",
      stage: "responding",
      content: "",
      turn_id: "turn-clock",
      seq: 2,
      timestamp: Date.now() / 1000,
      metadata: {
        status: "completed",
        user_message_id: 1,
        assistant_message_id: 2,
      },
    });
  });
  expect(screen.getByRole("status")).toHaveTextContent("2m 1s");
  fireEvent.click(screen.getByText("Retry"));
  expect(screen.getByRole("status")).toHaveTextContent("0s");
});
