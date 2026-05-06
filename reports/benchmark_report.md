# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |
|---|---:|---:|---:|---|
| single-agent | 12.09 | 0.0004 | 6.0 | iterations=0 | words=560 | citations=0 |
| multi-agent | 24.87 | 0.0013 | 8.9 | iterations=4 | words=487 | citations=5 |

## Comparison

| Metric | single-agent | multi-agent | Delta |
|---|---:|---:|---:|
| Latency (s) | 12.09 | 24.87 | +12.78 |
| Cost (USD) | 0.0004 | 0.0013 | +0.0009 |
| Quality /10 | 6.0 | 8.9 | +2.9 |

## Notes

- Quality scores are heuristic unless replaced by peer review.
- Cost uses gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output).

## Exit Ticket

1. **When to use multi-agent?** — Complex tasks needing specialisation: search + analysis + writing.
2. **When NOT to use multi-agent?** — Simple Q&A where LLM overhead exceeds quality gain.
