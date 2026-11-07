# Live models (optional)

Scientist3 runs end to end with **scripted agents and zero network access**. A
live OpenAI-compatible backend is optional and touches only the *creative*
stages. This document covers the flags, the trust model, and honest warnings.

## Quick start

```powershell
# OpenAI
$env:SCIENTIST3_API_KEY = "sk-..."
python -m scientist3 run toy_regression --run-id live-demo --live --model gpt-4o-mini

# Local Ollama (OpenAI-compatible endpoint, no key required by default)
$env:SCIENTIST3_BASE_URL = "http://localhost:11434/v1"
$env:SCIENTIST3_API_KEY = "ollama"
python -m scientist3 run toy_regression --run-id local-demo --live --model llama3.1
```

CLI flags: `--live`, `--model`, `--base-url`, `--api-key`. Environment
variables (used when a flag is absent):

| Variable | Default | Meaning |
|---|---|---|
| `SCIENTIST3_MODEL` | `gpt-4o-mini` | Model name sent to the endpoint |
| `SCIENTIST3_BASE_URL` | `https://api.openai.com/v1` | Base URL of an OpenAI-compatible API |
| `SCIENTIST3_API_KEY` | (empty) | Bearer token; live mode refuses to start without one |

Any endpoint speaking `POST {base_url}/chat/completions` works: OpenAI, Azure
OpenAI-compatible gateways, OpenRouter, Together, vLLM, LM Studio, Ollama. The
client is stdlib-only; errors (HTTP failures, malformed JSON, unparseable
content) raise `LLMError` and the affected stage falls back to its scripted
response.

## What the live model may do

| Stage | Live behaviour | Fallback |
|---|---|---|
| S1 limitations | Proposes a limitation list (JSON) | Contract limitations |
| S2 ideation | Proposes ideas; only entries with a `command` replace seed ideas | Contract ideas |
| S7 review | Scores the manuscript and returns strengths/weaknesses/questions | Scripted reviewer |
| S7 rebuttal | Selects at most two declared supplementary templates | Contract supplementaries in order |
| S8 meta-review | Returns `accept`/`revise` + critique | Threshold + integrity gate |

Disable individual live steps per case with `[variables] live_reviewer = false`
or `live_rebuttal = false`.

## What the live model can never do

* Choose promotions, rejections, or selection ranks — those come from
  `runner.compare_results` over stored observations.
* Write tables, numbers, or the evidence summary — `paper.py` builds them
  deterministically; live prose is limited to stage outputs above.
* Bypass the integrity audit — the audit reads stored observations, contract
  templates, and re-executed scores.

## Security and reproducibility warnings

1. **Live ideation can propose commands.** In live mode, an idea returned by
   the model is executed if it parses and carries a `command`. Treat the
   endpoint as trusted code execution: use it in a sandbox, or keep the
   contract's `command` fields authoritative by not enabling live mode.
2. **Non-determinism.** Temperature is low (0.2) but not zero; two live runs
   will differ. The sealed observations and thresholds keep decisions
   well-defined even when prose varies, but reruns are not byte-identical.
3. **No cost control beyond `max_tokens`.** Long papers × many review rounds can
   spend real money. Prefer local endpoints for experimentation.
4. **Model text is untrusted data.** Prompts include case text and stage
   context; responses are parsed with strict fallbacks. A model that emits
   instructions cannot alter thresholds, budgets, or audit results.

## Recommended posture

* CI and demos: scripted (default).
* Case authoring: scripted until the contract is stable, then one live run to
  sanity-check prose.
* Live runs you intend to keep: pin the model name and base URL, keep the run
  directory, and note the endpoint in your README — the manifest does not
  record provider identity beyond the model string you pass.
