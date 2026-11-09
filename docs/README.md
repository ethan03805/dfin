# Documents

Start with `../AGENTS.md` if you are about to change anything.

## Current

| Document | What it is |
|---|---|
| `SPEC.md` | What dfin must do: threat model, participants, trust tiers, confidentiality, integrity, anonymity, money, what is taken from Monero, acceptance tests, risks. All of it is required. |
| `PLAN.md` | The order of work. Every task has a "done when" and says what it needs. |
| `../STATUS.md` | What exists today, what does not, and the test or log behind each claim. |
| `protocol.md` | The wire contract: identities, transport and limits, messages, route setup, return modes, relay, directory, shards, wide stage state, fuzzing. |
| `anonymity.md` | How Tor mode works, how to set it up, and what it does not protect. |
| `prior-art-2026-11.md` | Other split-inference projects and the state of llama.cpp for the target models, as of November 2026. Input to PLAN task 1.1. |
| `logs/` | Raw benchmark and verification logs, named by what and when. |

## Archive

`archive/` holds documents that are no longer maintained. They explain how dfin got
here and still carry detail the spec points to. Where one disagrees with a current
document, the current document is right. See `archive/README.md`.
