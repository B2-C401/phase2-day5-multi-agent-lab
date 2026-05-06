### Trương Minh Tiền - 2A202600438
# Design Template

## Problem

Người dùng đặt câu hỏi nghiên cứu phức tạp (ví dụ: "Research GraphRAG state-of-the-art and
write a 500-word summary"). Hệ thống cần tìm kiếm thông tin, phân tích, và viết câu trả lời
chất lượng cao với citations — ba nhiệm vụ có yêu cầu prompt và kỹ năng rất khác nhau.

## Why multi-agent?

Single-agent thực hiện cả ba việc trong một lần gọi LLM dẫn đến:
- Thiếu citations (0 citations trong baseline vs 5 ở multi-agent)
- Không có bước critical analysis riêng biệt
- Prompt quá tổng quát → output đạt 7/10 thay vì 9/10
- Không thể tối ưu temperature riêng cho từng bước (research cần 0.2, writing cần 0.4)

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Quyết định agent nào chạy tiếp; enforce max_iterations | ResearchState | route_history updated | Loop vô hạn nếu thiếu iteration guard |
| Researcher | Tìm nguồn, viết research notes | query + sources | research_notes (300–500 words) | Search API down → fallback mock; LLM timeout → retry 3x |
| Analyst | Phân tích claims, evidence strength, gaps | research_notes | analysis_notes (200–350 words) | Empty research_notes → skip + log warning |
| Writer | Tổng hợp final answer với citations | research + analysis notes | final_answer (~500 words) | Missing context → degrade gracefully |

## Shared state

| Field | Type | Lý do cần |
|---|---|---|
| `request` | ResearchQuery | Query gốc, max_sources, audience — cần ở mọi agent |
| `iteration` | int | Enforce max_iterations guard |
| `route_history` | list[str] | Debug trace + detect consecutive failure |
| `sources` | list[SourceDocument] | Researcher viết, Writer cite |
| `research_notes` | str | Kết quả Researcher → input Analyst và Writer |
| `analysis_notes` | str | Kết quả Analyst → input Writer |
| `final_answer` | str | Output cuối cùng |
| `trace` | list[dict] | Token count, cost, timing per agent |
| `errors` | list[str] | Debug; failure guard đọc để quyết định retry/abort |

## Routing policy

```
START
  │
  ▼
[Supervisor]──iteration >= max_iterations──▶ done
  │
  ├─ research_notes is None ──▶ [Researcher] ──┐
  │                                             │
  ├─ analysis_notes is None ──▶ [Analyst]  ──┤
  │                                             │
  ├─ final_answer is None ────▶ [Writer]   ──┤
  │                                             │
  └─ all complete ────────────▶ done            │
       ▲                                        │
       └────────────────────────────────────────┘
                 (loop back to Supervisor)

Consecutive failure guard: nếu route_history[-3:] == [X, X, X] → abort → done
```

## Guardrails

- **Max iterations**: `MAX_ITERATIONS=6` (env var) — Supervisor enforce hard cap
- **Timeout**: `TIMEOUT_SECONDS=60` — OpenAI client timeout per call
- **Retry**: `tenacity` retry 3 lần với exponential backoff (2–10s) trong LLMClient
- **Fallback**: Search → Tavily nếu có key, fallback mock nếu không; Agent fail 3 lần → route done
- **Validation**: `pydantic` validate tất cả input/output schemas; empty research_notes → skip + warn

## Benchmark plan

| Query | Metric | Expected outcome |
|---|---|---|
| "Research GraphRAG state-of-the-art and write a 500-word summary" | quality /10, latency, cost | Multi-agent quality cao hơn 2+ điểm, latency 2× nhưng có citations |
| "Compare single-agent and multi-agent workflows for customer support" | citation_coverage, word_count | Multi-agent cite nguồn cụ thể, single-agent generic |
| "Summarize production guardrails for LLM agents" | structure score (##, Key Takeaways) | Multi-agent có structure rõ hơn nhờ Analyst pass thông tin cho Writer |

**Actual results (2026-05-06):**

| Run | Latency | Cost | Quality | Citations |
|---|---:|---:|---:|---:|
| single-agent | 11.16s | $0.0004 | 7.0/10 | 0 |
| multi-agent | 24.99s | $0.0015 | 9.0/10 | 5 |

Multi-agent tốn 3.75× cost và 2.2× latency để đổi lấy +2 quality và 5 citations — đáng giá
cho research task dài, không đáng giá cho Q&A đơn giản.
