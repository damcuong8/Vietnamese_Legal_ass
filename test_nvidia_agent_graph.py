import os
import sys
import uuid

from langchain_openai import ChatOpenAI


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CHECKPOINTER_BACKEND", "none")


def get_nvidia_llm(
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 20,
    enable_thinking: bool = True,
    thinking_token_budget: int | None = None,
) -> ChatOpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY. Export it before running this test.")

    return ChatOpenAI(
        model=os.getenv("NVIDIA_LLM_MODEL", "qwen/qwen3.5-122b-a10b"),
        base_url=os.getenv("NVIDIA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=int(os.getenv("NVIDIA_LLM_MAX_TOKENS", "8192")),
        max_retries=2,
    )


def _preview(value: object, limit: int = 500) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "..."


def main() -> int:
    question = (
        "Công ty nhập khẩu lô hàng bị đối tác yêu cầu tạm dừng thủ tục hải quan vì nghi xâm phạm quyền sở hữu trí tuệ, sau đó phát hiện hàng không phù hợp hợp đồng và khách mua là người cao tuổi thì công ty phải xử lý trách nhiệm hàng hóa, nghĩa vụ bồi thường cho bên yêu cầu kiểm soát và quy trình tiếp nhận khiếu nại từ khách hàng này như thế nào?"
    )
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    from Agents import graph as agent_graph

    agent_graph.get_llm = get_nvidia_llm
    app = agent_graph.stateless_app
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Running real Agents.graph with NVIDIA endpoint")
    print(f"Question: {question}\n")

    final_state = None
    for event in app.stream({"question": question}, config, stream_mode="values"):
        final_state = event
        if event.get("plan"):
            plan = event["plan"]
            print(
                "[planner]",
                _preview(plan.get("intent")),
                "| targets:",
                len(plan.get("search_targets", [])),
            )
        if event.get("retrieved_documents"):
            print("[retriever] docs:", len(event["retrieved_documents"]))
        if event.get("extracted_evidence"):
            print("[compress] evidence chars:", len(event["extracted_evidence"]))
        if event.get("messages"):
            msg = event["messages"][-1]
            msg_type = getattr(msg, "type", type(msg).__name__)
            tool_calls = getattr(msg, "tool_calls", None)
            print(f"[message:{msg_type}] tool_calls={len(tool_calls or [])}")
            if getattr(msg, "content", None):
                print(_preview(msg.content, 700))

    print("\n=== FINAL ===")
    messages = (final_state or {}).get("messages", [])
    if messages:
        print(_preview(messages[-1].content, 4000))
    else:
        print("No final message returned.")

    print("\n=== IDS ===")
    print("relevant_chunk_ids:", (final_state or {}).get("relevant_chunk_ids", []))
    print("applied_chunk_ids:", (final_state or {}).get("applied_chunk_ids", []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
