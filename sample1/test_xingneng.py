# tests/test_perf.py
import time
import psutil
import os
from inference import ChatInference

def measure(prompt, history, max_tokens):
    bot = ChatInference()
    # 记录开始时间和内存
    t0 = time.time()
    process = psutil.Process(os.getpid())
    mem0 = process.memory_info().rss

    _ = bot.generate_reply(prompt, history, max_tokens=max_tokens)

    t1 = time.time()
    mem1 = process.memory_info().rss
    return (t1 - t0), (mem1 - mem0)

def test_performance_short():
    latency, mem = measure("今天天气如何？", [], max_tokens=50)
    print(f"50 tokens: 耗时 {latency:.2f}s，额外内存 {mem/1024/1024:.2f} MB")

def test_performance_long():
    long_history = ["问：请介绍人工智能。", "答：……"] * 10
    latency, mem = measure("请帮我写一段 AI 论文开头。", long_history, max_tokens=100)
    print(f"100 tokens + 多轮上下文：耗时 {latency:.2f}s，额外内存 {mem/1024/1024:.2f} MB")
