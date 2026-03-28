# scripts/quick_test.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import get_llm_response
import time

test_cases = [
    "Con lắc lò xo có k=100N/m, m=0.1kg. Tính chu kỳ",
    "Giải thích hiện tượng giao thoa ánh sáng",
    "Hai điện tích q1=2μC, q2=3μC cách nhau 10cm. Tính lực Coulomb",
    "Định luật bảo toàn cơ năng áp dụng khi nào?",
]

for i, q in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"Câu {i}: {q}")
    print('-'*60)
    
    start = time.time()
    response = get_llm_response(q)
    elapsed = time.time() - start
    
    print(f"PhysBot: {response}")
    print(f"\n⏱️ LLM latency: {elapsed:.2f}s")
    print(f"📏 Độ dài response: {len(response)} ký tự")