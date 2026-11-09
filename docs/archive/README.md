# Archive

Frozen on 9 November 2026. Nothing here is maintained, and links inside these files may
point at paths that have since moved. Where one of them disagrees with `../SPEC.md`,
`../PLAN.md`, `../protocol.md` or `../../STATUS.md`, it is out of date.

| Document | Why it is kept |
|---|---|
| `design-v0.1.md` | The first design. SPEC.md still relies on it for the model physics (sections 2 and 3), the execution model (6), replay logs and failover (7), routing (8), weight distribution (13) and the API (14), and cites it as "v0.1 section N". Its trust, money and phasing sections are superseded. |
| `phase0-findings.md` | What was learned building the LAN pod: what each architecture needs at a stage boundary, tiny-model and real-model results, the two-machine run, cold start. |
| `phase1-findings.md` | What was learned building the chain: real-model measurements, the f16 near-tie finding, wide stage state, what fuzzing found. Its sections on cloaked nodes and region fencing describe code that was later removed. |
| `privacy-review.md` | The adversarial review of the Tor transport and request isolation: findings P01 to P16 with reproductions, fixes and test results. |
| `privacy-review-handoff.md` | The state that review was stopped in. The CI failure it calls unresolved is PLAN task 1.8. |
| `anonymity-work-log.md` | Dated decisions and checks from making Tor the default. |
| `mac-node-prompt.md` | A setup prompt for a second machine, written for the first TCP build. Its commands no longer work. |

The design that SPEC.md grew out of was `design-v0.2.md`. It was edited in place into
SPEC.md; `git log --follow docs/SPEC.md` shows the original.
