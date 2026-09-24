# ADR-0002: Hexagonal (ports-and-adapters) layout

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

DevRAG depends on several swappable technologies: embedding providers (OpenAI
vs. on-device), LLM providers, rerankers (cross-encoder, LLM, identity),
parsers, and storage. Locking these choices into business logic would make the
project hard to test offline, hard to swap providers, and hard to evolve. The
project also wants a demo/eval path that works without paid APIs.

## Decision

Adopt a layered, hexagonal (ports-and-adapters) style with these rules:

- **`app/domain`** holds entities, value objects, and *ports* (interfaces) —
  pure business logic with no framework or infrastructure imports.
- **`app/infrastructure`** holds *adapters* — concrete implementations (SQL
  repositories, parsers, chunkers, embedding/LLM/reranker providers, health
  probes) that implement domain interfaces.
- **`app/application`** holds use-case workflows (ingest-and-index,
  retrieve-context, generate-answer, evaluate) that orchestrate ports with
  injected dependencies.
- **`app/services`** composes use cases into higher-level services for the API.
- **`app/api/v1`** exposes FastAPI routes and schemas only.
- Dependencies point **inward**; infrastructure may depend on domain, never the
  reverse.

Providers are chosen at runtime from `app/core/settings.py` via factory
functions, so the same code path serves OpenAI-backed production and offline
local/canned providers.

## Consequences

- **Positive:** units can be tested with fakes; providers are swappable;
  migrations to new infrastructure are localized; the demo and evaluation
  harness reuse the same wired services as the API.
- **Negative:** more indirection (interfaces + factories) than a flat design;
  larger per-feature file count.
- **Risk:** interfaces can over-generalize — avoided by adding ports only when a
  second adapter is plausible or offline testing requires it.