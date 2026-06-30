import os
import uuid
import langchain
langchain.debug = True
from Agents.graph import app

def main():
    print("Khởi tạo phiên chạy kiểm thử (RAG Agent)...")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    question = "Công ty nhập khẩu lô hàng bị đối tác yêu cầu tạm dừng thủ tục hải quan vì nghi xâm phạm quyền sở hữu trí tuệ, sau đó phát hiện hàng không phù hợp hợp đồng và khách mua là người cao tuổi thì công ty phải xử lý trách nhiệm hàng hóa, nghĩa vụ bồi thường cho bên yêu cầu kiểm soát và quy trình tiếp nhận khiếu nại từ khách hàng này như thế nào?"
    print(f"Câu hỏi: {question}\n")
    
    try:
        for event in app.stream({"question": question}, config, stream_mode="values"):
            print("--- Trạng thái mới ---")
            for k, v in event.items():
                if k == "messages" and v:
                    last_msg = v[-1]
                    print(f"[{last_msg.type.upper()}] {last_msg.content[:200]}...")
                elif k == "plan":
                    intent = v.get("intent", "Không rõ")
                    targets = len(v.get("search_targets", []))
                    print(f"[PLAN] Mục tiêu: {intent} | Số hướng tra cứu: {targets}")
                elif k == "retrieved_documents":
                    print(f"[RETRIEVER] Đã truy xuất {len(v)} tài liệu sau Rerank")
                elif k == "extracted_evidence":
                    print(f"[COMPRESSION] Đã nén thành công, độ dài: {len(v)} ký tự")
                    
        print("\n--- HOÀN TẤT ---")
        state = app.get_state(config)
        messages = state.values.get("messages", [])
        if messages:
            print(f"\nFinal Response:\n{messages[-1].content}")
            
    except Exception as e:
        print(f"Lỗi khi chạy: {e}")

if __name__ == "__main__":
    main()
