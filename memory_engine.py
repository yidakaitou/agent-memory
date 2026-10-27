#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_engine.py —— 可复用的 Agent 长期记忆引擎（优化版）

在原「紫苑」四层记忆设计的基础上，做了三点优化：
  ② 全量注入 → 按需检索（TF-IDF 检索 + 近因保底）
  ③ 只压不删 → 加遗忘（时间衰减 + 过期淘汰）
  ① 规则压缩 → 分层压缩，摘要位留给离线模型（本版用抽取式占位）

设计目标：一个 Agent 挂上它，重启/换会话/上下文清空之后，仍然能
「记住」该记住的东西，且记忆不随历史无限膨胀。

零第三方依赖：只用 numpy + 标准库。
持久化：JSONL 追加，重启可恢复（这才是"上下文之外"的证明）。
"""

import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

# —— 分层压缩：年龄越老，保留越短 ——
# 各档：(最大年龄秒数, 保留字符数, 该层预算占比)
TIERS = [
    (2 * 3600, 40, 0.45),      # 近 2 小时：截 40 字
    (24 * 3600, 80, 0.25),     # 近 24 小时：截 80 字
    (7 * 86400, 40, 0.20),     # 近 7 天：截 40 字
    (30 * 86400, 16, 0.10),    # 近 30 天：只留 16 字索引
]
FORGET_AFTER = 30 * 86400      # 超过 30 天：遗忘（淘汰）
INJECTION_BUDGET = 1200        # 每次注入的总字符预算


def _tokenize(text: str) -> list[str]:
    """极简中文分词：连续汉字、英文词、数字各算一个 token。"""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+", text.lower())


class MemoryEngine:
    def __init__(self, path: str):
        self.path = Path(path)
        self.memories: list[dict] = []
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.memories.append(json.loads(line))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for m in self.memories:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---------- 写入 ----------
    def add(self, text: str, tags: Optional[list[str]] = None) -> str:
        """记一条记忆，返回 id。"""
        m = {
            "id": uuid.uuid4().hex[:8],
            "text": text,
            "tags": tags or [],
            "ts": time.time(),
        }
        self.memories.append(m)
        self._save()
        return m["id"]

    # ---------- 遗忘（优化 ③）----------
    def decay(self, now: Optional[float] = None):
        """淘汰过期记忆，返回淘汰数。"""
        now = now or time.time()
        before = len(self.memories)
        self.memories = [m for m in self.memories if now - m["ts"] <= FORGET_AFTER]
        removed = before - len(self.memories)
        if removed:
            self._save()
        return removed

    # ---------- 压缩 ----------
    @staticmethod
    def _compress(text: str, limit: int) -> str:
        t = text.strip().replace("\n", " ")
        return t if len(t) <= limit else t[: limit - 1] + "…"

    # ---------- 按需检索（优化 ②）----------
    def _search(self, query: str, topk: int) -> list[tuple[dict, float]]:
        """TF-IDF 检索，返回 [(memory, score), ...] 按分数降序。"""
        if not self.memories:
            return []
        corpus = [m["text"] for m in self.memories]
        vocab: dict[str, int] = {}
        for t in _tokenize(query):
            vocab.setdefault(t, len(vocab))
        for c in corpus:
            for t in _tokenize(c):
                vocab.setdefault(t, len(vocab))

        matrix = np.zeros((len(corpus), len(vocab)))
        for i, c in enumerate(corpus):
            for t in _tokenize(c):
                matrix[i, vocab[t]] += 1
        df = (matrix > 0).sum(axis=0)
        idf = np.log((len(corpus) + 1) / (df + 1)) + 1
        matrix = matrix * idf

        q = np.zeros((1, len(vocab)))
        for t in _tokenize(query):
            if t in vocab:
                q[0, vocab[t]] += 1
        scores = (matrix @ q.T).ravel()
        idx = np.argsort(scores)[::-1][:topk]
        return [(self.memories[i], float(scores[i])) for i in idx if scores[i] > 0]

    # ---------- 注入块生成（分层压缩 + 预算）----------
    def inject(self, query: str, topk: int = 3, recent_n: int = 2) -> str:
        """
        生成要注入 prompt 的记忆块：
        - 检索最相关的 topk 条（按需，而非全量）
        - 保底带上最近 recent_n 条（近因，防止检索漏掉刚发生的）
        - 按年龄分层压缩，总预算 INJECTION_BUDGET 字
        """
        now = time.time()
        picked: dict[str, tuple[dict, float]] = {}

        for m, score in self._search(query, topk):
            picked[m["id"]] = (m, score)
        for m in sorted(self.memories, key=lambda x: -x["ts"])[:recent_n]:
            picked.setdefault(m["id"], (m, 0.0))

        blocks: list[tuple[int, str]] = []  # (age_bucket, text)
        for m, _ in picked.values():
            age = now - m["ts"]
            for max_age, limit, _ in TIERS:
                if age <= max_age:
                    blocks.append((limit, self._compress(m["text"], limit)))
                    break
            else:
                continue  # 该记的都不该活到 30 天，跳过

        # 按"越新越先"排序，再按预算截断
        blocks.sort(key=lambda b: b[0], reverse=True)
        out, budget = [], INJECTION_BUDGET
        for limit, text in blocks:
            cost = min(len(text), budget)
            if cost <= 0:
                break
            out.append(text[:cost])
            budget -= cost
        return "\n".join(out)

    # ---------- 离线摘要（优化 ① 的占位）----------
    def offline_summary(self, days: int = 1) -> str:
        """抽取式摘要：把近 N 天的高频词句拼成摘要（替代离线模型调用）。"""
        now = time.time()
        recent = [m for m in self.memories if now - m["ts"] <= days * 86400]
        if not recent:
            return ""
        words: dict[str, int] = {}
        for m in recent:
            for t in _tokenize(m["text"]):
                words[t] = words.get(t, 0) + 1
        top = sorted(words, key=words.get, reverse=True)[:12]
        return " ".join(top)

    def stats(self) -> str:
        return f"{len(self.memories)} 条记忆"


# ---------- 演示：模拟一次"重启后仍然记得" ----------
if __name__ == "__main__":
    import tempfile

    path = Path(tempfile.gettempdir()) / "agent_memory_demo.jsonl"

    print("=== 第一次会话：写入记忆 ===")
    eng = MemoryEngine(str(path))
    eng.add("用户是周怿晨，在张家界读飞机机电设备维修，大一")
    eng.add("他最近在弄 Freelancer，英语不太行，接单不顺")
    eng.add("他入驻了学校孵化基地，项目是数据处理服务")
    eng.add("他喜欢被叫名字，讨厌别人发'在吗'")
    eng.add("他决定做 Agent 长期记忆的项目，充实技能库")
    print(eng.stats())

    print("\n=== 模拟重启：新建引擎实例，从磁盘恢复 ===")
    eng2 = MemoryEngine(str(path))
    print(f"重启后 {eng2.stats()}（记忆没丢）")

    print("\n=== 按需检索：问一个相关问题 ===")
    query = "他想做什么项目充实技能"
    for m, score in eng2._search(query, 3):
        print(f"  相关度 {score:.1f}  {m['text'][:40]}")

    print("\n=== 生成注入块（受预算约束）===")
    injection = eng2.inject(query)
    print(f"注入 {len(injection)} 字（预算 {INJECTION_BUDGET}）：")
    print("  " + injection.replace("\n", "\n  "))

    print("\n=== 遗忘机制 ===")
    # 手工造一条 31 天前的记忆
    eng2.memories.append({
        "id": "old", "text": "这条太老了应该被忘掉", "tags": [], "ts": time.time() - 31 * 86400,
    })
    eng2._save()
    removed = eng2.decay()
    print(f"淘汰了 {removed} 条过期记忆，剩 {eng2.stats()}")

    print("\n=== 离线摘要（抽取式占位）===")
    print("  " + eng2.offline_summary(30))
