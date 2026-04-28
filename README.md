# Holistic Temporal User Map Starter

This project is a FastAPI-based reference implementation for persistent personalization in an AI assistant.

The central idea is that a useful assistant should not depend on replaying raw chat history forever. Instead, it should build a compact and continuously updated model of the user: what level of detail they prefer, which topics recur, whether recent answers are landing well, and which interests are durable versus temporary. This repository implements that idea as a structured user map, persists it locally, and uses it to produce lower-token prompt context for future responses.

At a high level, the system tries to answer three product questions:

- How can an assistant remember user preferences without resending entire conversations?
- How can that memory adapt over time instead of locking in stale assumptions?
- How can we test whether the persistence layer is actually worth its token overhead?

The codebase answers those questions with a combination of structured memory, controlled decay, prompt compaction, live chat integration, and a deterministic benchmark harness.

## Core Concept

Each user has a graph-like profile made of:

- preference nodes
- communication style nodes
- cognitive/intellectual depth nodes
- sentiment and satisfaction signals
- topic-interest nodes
- trust/attachment/engagement signals

Each interaction updates the map.

The map can then generate compact personalization context that can be passed into an LLM before answering the next message.

User maps and interaction evidence are persisted in `data/user_maps.db` so the profile survives API restarts during local development.

The important distinction is that the assistant is not storing "memory" as a giant transcript window. It is storing interpretable state. That means the system can:

- inspect what it believes about the user
- decay or prune stale beliefs
- promote recurring interests into longer-lived traits
- explain why a prompt contains a given preference or topic
- benchmark the memory layer independently of a live provider call

## Trusted Building Blocks

This starter now leans on stable, well-understood infrastructure for the persistence overhead:

- SQLite for durable local profile storage
- Pydantic models for profile and interaction validation
- FastAPI for typed API surfaces

The database keeps two layers of memory:

- `user_profiles`: the latest structured user map used for personalization
- `interaction_events`: the evidence log showing how the profile changed over time

That split gives you a practical foundation for an AI assistant that needs both:

- a compact current profile for prompting
- an audit trail for debugging, retraining, or later profile rebuilding

This architecture matters because it separates operational concerns cleanly:

- SQLite handles durability and caching rather than overloading the model with storage responsibilities.
- Pydantic defines strict contracts for interactions, profiles, prompt assemblies, streaming chunks, and evaluation results.
- FastAPI exposes each step of the memory lifecycle as a typed surface that can be tested independently.

## Cost And Retention Strategy

The current prompt-facing context is intentionally compressed to reduce token overhead:

- it emits short grouped summaries instead of raw metric-heavy node dumps
- it prefers only active, high-signal traits
- it keeps the durable evidence in SQLite instead of sending history back into the model

The profile also now decays over time:

- all traits soften gradually when they are not reinforced
- topic interests decay more slowly than other traits
- frequently revisited topics decay slowest because higher evidence counts increase retention

This lets the assistant stay adaptive without paying to resend raw historical interactions or stale preferences.

The retention strategy is intentionally asymmetric. Topics that appear repeatedly are treated as stronger evidence than one-off mentions, and signal types decay at different rates. The result is a profile that is designed to be opinionated enough to be useful, but not so sticky that one bad interaction permanently distorts future prompting.

## Next-Layer Profile Memory

The project now separates topic memory into two bands:

- short-term topics capture recent spikes in interest
- long-term topics are promoted from repeated short-term evidence and decay more slowly

This reduces over-retention from one-off conversations while keeping durable recurring interests.

The app also supports profile compaction:

- stale low-signal nodes are pruned
- repeated short-term topics are promoted into long-term interests
- a manual compaction endpoint can compact all stored profiles

For prompt construction, use:

```text
/api/users/{user_id}/prompt-context
```

This returns a prompt-ready payload with:

- compact profile context
- profile confidence score
- up to a few recent evidence items only when the profile is weak

That confidence gating is one of the main token-saving mechanisms in the project. If the profile is already strong, the system avoids paying for extra evidence in the prompt. If the profile is weak or still forming, it selectively includes a small amount of recent evidence to stabilize behavior.

## LLM Integration

The app can now call an LLM directly through:

```text
/api/chat
```

Request body:

```json
{
  "user_id": "scott",
  "message": "Explain LLM memory in more detail."
}
```

Flow:

- load the persisted user profile
- assemble compact prompt context from the profile and recent evidence
- call the configured LLM
- analyze the generated reply and update the persistent profile
- store the interaction for future prompt assembly and compaction

This turns the project from a static memory demo into a closed personalization loop:

1. The system reads the current structured profile.
2. It turns that profile into a compact prompt payload.
3. The selected provider generates a response.
4. The response is analyzed for style, satisfaction, confusion, and topical evidence.
5. The persistent profile is updated so the next turn can be better targeted.

Environment variables:

- `LLM_PROVIDER`: optional, defaults to `openai`
- `OPENAI_API_KEY`: required for `/api/chat`
- `OPENAI_MODEL`: optional, defaults to `gpt-4o-mini`
- `OPENAI_BASE_URL`: optional, for compatible endpoints or proxies

Generated replies can now be corrected after the fact through:

```text
/api/interactions/{interaction_id}/feedback
```

Request body:

```json
{
  "explicit_feedback": "That was too shallow. I wanted more detail."
}
```

This re-analyzes the original interaction with the explicit feedback, updates the persisted profile, and overwrites the stored analysis for that interaction.

Streaming chat is available at:

```text
/api/chat/stream
```

It returns NDJSON chunks while the model is generating, then persists the final response and analysis once streaming completes.

Prompt assembly is now cached in SQLite by:

- `user_id`
- `profile_version`

That lets stable user profiles reuse the same prompt payload without rebuilding it on every request.

In practice, this means the system pays the cost of profile interpretation only when the profile changes. Stable users with stable preferences can reuse prompt context across requests, which is one of the main operational wins the project is trying to prove.

Provider switching is now active:

- `LLM_PROVIDER=openai` uses the OpenAI client
- `LLM_PROVIDER=anthropic` uses the Anthropic client

Anthropic environment variables:

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL`
- `ANTHROPIC_BASE_URL`

The live chat path is provider-backed. The benchmark path is not. The scientific evaluation described below runs locally and deterministically; it uses provider settings only for pricing assumptions and report metadata, not for live response generation.

## Scientific Viability Test

The project now includes a reproducible evaluation harness for the core hypothesis that persistent personalization improves answer quality.

Use:

```text
/api/evaluations/viability
```

This runs a deterministic A/B-style experiment:

- baseline condition: a no-memory assistant that ignores the user profile
- personalized condition: the same interaction loop with the persistent user map updating across rounds
- judge: a fixed scoring function that measures preference alignment and simulated satisfaction using the same hidden synthetic user preferences in both conditions

The benchmark is designed to test the memory architecture itself, not the raw capability of a frontier model. It answers a narrower engineering question:

- If the same assistant loop is run with and without persistence, does the persistent condition converge toward user preferences strongly enough to justify its extra prompt cost?

Output includes:

- mean preference-alignment score for baseline and personalized conditions
- mean simulated satisfaction for both conditions
- mean token-cost estimates for baseline and personalized conditions
- cost-benefit ratios showing alignment lift per added token cost
- per-scenario lift values
- an overall `supports_viability` flag based on the predefined lift threshold

The scoring dimensions mean the following:

- Alignment: whether the answer matches the hidden preferred style and preferred topics of the evaluation scenario.
- Satisfaction: a deterministic simulated proxy derived from alignment so the project can compare directional user happiness without requiring live raters for every run.
- Token cost: an estimated provider-specific billing cost using tokenizer-aware counting and model pricing assumptions.
- Cost-benefit: how much alignment or satisfaction lift is achieved per unit of added token cost.

The benchmark panel now mixes:

- synthetic scenarios generated from the project hypothesis
- human-labeled scenarios stored in [app/evaluation/human_scenarios.json](app/evaluation/human_scenarios.json)

That mixed panel matters because a synthetic-only benchmark can become self-referential. Human-labeled scenarios add an external check on whether the evaluation is rewarding patterns that look useful from a product perspective rather than just rewarding its own handcrafted assumptions.

Every viability run also exports report artifacts under `data/reports/`:

- JSON for full machine-readable metrics
- CSV for revision-to-revision tracking in spreadsheets or dashboards

Each exported report now includes explicit run metadata so longitudinal comparisons do not depend on filenames alone:

- git revision when available
- app version
- LLM provider and model
- tokenizer identifier used for billing estimates

The cost-benefit layer now uses provider-aware tokenizer and pricing logic instead of a raw word-count heuristic, so the benchmark cost metric is closer to real model billing behavior.

## Validated Results

The current implementation has two layers of verified results: unit and integration-style project tests, and benchmark outcomes from the scientific viability harness.

### Automated Test Status

Latest full validation run:

- `27 passed in 2.72s`
- validated with `C:/Users/scott/AppData/Local/Programs/Python/Python311/python.exe -m pytest -q`

The passing suite currently covers:

- interaction analysis and preference updates
- context building and compact summaries
- decay behavior for frequent versus infrequent topics
- profile compaction and long-term topic promotion
- prompt assembly confidence gating
- SQLite persistence and interaction history retrieval
- streaming chat persistence
- explicit feedback replay
- prompt-context caching keyed by profile version
- multi-provider settings resolution
- viability report generation and artifact export
- provider-specific token cost estimation
- CLI benchmark execution

### Latest Benchmark Outcome

Latest benchmark artifact used for this documentation:

- [data/reports/viability_report_0_2_0_20260428T003226Z.json](data/reports/viability_report_0_2_0_20260428T003226Z.json)

Observed result for `--rounds-per-user 4`:

- `scenario_count`: 7
- `supports_viability`: `true`
- `baseline_mean_alignment`: `0.15`
- `personalized_mean_alignment`: `0.7107`
- `mean_alignment_lift`: `0.5607`
- `baseline_mean_satisfaction`: `-0.7`
- `personalized_mean_satisfaction`: `0.4214`
- `mean_satisfaction_lift`: `1.1214`
- `baseline_mean_token_cost`: `0.00002012`
- `personalized_mean_token_cost`: `0.00002443`
- `mean_token_cost_delta`: `0.00000431`
- `alignment_lift_per_token_delta`: `130092.8074`
- `worth_it`: `true`

Interpretation:

- The persistent condition materially outperformed the no-memory baseline on both alignment and simulated satisfaction.
- The personalized path did cost more than the baseline, as expected, because it carries profile-memory prompt overhead.
- In this deterministic benchmark, the quality lift was large relative to the added estimated token cost, so the current implementation clears its own viability threshold.

Those numbers should be treated as an internal engineering signal, not a universal claim about production users. They show that this persistence design is behaving as intended in the controlled scenarios currently included in the repository.

This does not claim a universal proof for all real users. It gives you a reproducible internal benchmark that can falsify the approach if the personalized condition fails to beat the baseline under controlled conditions.

For maintenance, use:

```text
/api/admin/compact-profiles
```

to compact all persisted profiles.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run the benchmark without starting the API server:

```bash
py -3.11 -m app.cli viability-benchmark --rounds-per-user 4
```

On Windows in this repo, avoid plain `python -m app.cli ...` if `python` resolves to Python 3.14. The benchmark should use the same Python 3.11 environment used for tests.

You can also use the included wrapper:

```powershell
./scripts/run-viability-benchmark.ps1 --rounds-per-user 4
```

The benchmark does not require API keys because it does not call a live LLM provider. It runs locally and deterministically, then uses configured provider metadata only for tokenization and billing estimates in the report.

Open:

```text
http://127.0.0.1:8000/docs
```

## Test Example

POST to:

```text
/api/interactions
```

Body:

```json
{
  "user_id": "scott",
  "user_message": "That was not what I meant. I wanted a deeper explanation.",
  "assistant_response": "Here is a short answer.",
  "explicit_feedback": null
}
```

Then request:

```text
/api/users/scott/context
```

This returns compact personalization context.

You can also inspect:

```text
/api/users/scott/profile
```

for a structured view of what the system currently believes about the user, and:

```text
/api/users/scott/history
```

for the recent evidence used to shape that profile.
