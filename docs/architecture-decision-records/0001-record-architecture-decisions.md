# ADR-0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

DevRAG is a multi-phase project covering ingestion, retrieval, generation,
evaluation, observability, and deployment. As capabilities are added, important
structural decisions are being made (storage engine, retrieval strategy,
provider interfaces, observability approach). Without records, future
contributors cannot tell *why* the system is shaped the way it is, what
alternatives were rejected, and when decisions may need revisiting.

## Decision

Significant architecture decisions are recorded as Architecture Decision
Records (ADRs) in `docs/architecture-decision-records/`, following the
Nygard template:

- **Status** — Accepted / Proposed / Deprecated / Superseded
- **Date** — when the decision was made
- **Context** — the forces and constraints that motivate the decision
- **Decision** — what was decided
- **Consequences** — tradeoffs, positive and negative

Each ADR is a short, immutable document; revisions are new ADRs that supersede
old ones. ADRs accompany the change that implements them and are reviewed in
the same pull request.

## Consequences

- **Positive:** Decisions are discoverable and reviewable; onboarding and
  architectural review are faster; supersession history is explicit.
- **Negative:** ADRs need maintenance discipline; small changes occasionally
  carry doc overhead.
- **Risk:** ADRs can drift from reality if they are not updated or superseded on
  material changes — mitigated by reviewing ADRs alongside code changes.