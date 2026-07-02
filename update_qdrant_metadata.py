import time
import argparse
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es-host", default="http://localhost:9201")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--index", default="legal_chunks")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    print(f"[*] Kết nối Elasticsearch: {args.es_host}")
    es = Elasticsearch(args.es_host)
    if not es.ping():
        raise RuntimeError("Không thể kết nối Elasticsearch!")

    print(f"[*] Kết nối Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=3600)
    
    print("[*] Bắt đầu đồng bộ trường 'is_local' từ Elasticsearch sang Qdrant (Chế độ Resume)...")
    
    start_time = time.time()
    processed = 0
    
    while True:
        # Tìm các docs trong Qdrant chưa có trường is_local
        try:
            scroll_result, next_page = qdrant.scroll(
                collection_name=args.index,
                scroll_filter=Filter(
                    must=[IsEmptyCondition(is_empty=PayloadField(key="is_local"))]
                ),
                limit=args.batch_size,
                with_payload=False,
                with_vectors=False
            )
        except Exception as e:
            print(f"[!] Lỗi khi lấy dữ liệu từ Qdrant (có thể do timeout): {e}")
            time.sleep(5)
            continue
            
        if not scroll_result:
            print("[*] Không còn document nào thiếu trường 'is_local' trong Qdrant!")
            break
            
        missing_ids = [point.id for point in scroll_result]
        
        # Lấy is_local từ Elasticsearch
        try:
            mget_res = es.mget(index=args.index, body={"ids": missing_ids})
        except Exception as e:
            print(f"[!] Lỗi khi mget từ ES: {e}")
            time.sleep(5)
            continue
            
        local_ids = []
        central_ids = []
        
        for doc in mget_res.get("docs", []):
            if not doc.get("found"):
                continue
            is_local = doc.get("_source", {}).get("is_local")
            doc_id = doc["_id"]
            
            if is_local is True:
                local_ids.append(doc_id)
            elif is_local is False:
                central_ids.append(doc_id)
                
        # Update lại Qdrant
        try:
            if local_ids:
                qdrant.set_payload(collection_name=args.index, payload={"is_local": True}, points=local_ids, wait=True)
            if central_ids:
                qdrant.set_payload(collection_name=args.index, payload={"is_local": False}, points=central_ids, wait=True)
        except Exception as e:
            print(f"[!] Lỗi khi update payload lên Qdrant: {e}")
            time.sleep(5)
            continue
            
        processed += len(missing_ids)
        elapsed = time.time() - start_time
        print(f"[+] Đã xử lý bù {processed} docs... ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    print(f"[*] Hoàn tất đồng bộ (resume) trong {total_time:.1f}s!")

if __name__ == '__main__':
    main()
