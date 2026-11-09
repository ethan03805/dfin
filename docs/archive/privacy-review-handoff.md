# Privacy review handoff — 7 November 2026

The user stopped the review. Work is preserved on `codex/strict-anonymity` in
[draft PR #2](https://github.com/ethan03805/dfin/pull/2). Nothing was merged or
deployed publicly, and no wallets or real funds were used. The original checkout
was not changed.

## What was accomplished

- Made authenticated Tor onion connections the default, with no direct fallback,
  fresh request identities, and isolated directory/route connections.
- Fixed cancelled-request retention, route/session ownership and capacity races,
  and queued model work that ran after its caller disappeared.
- Hardened node-key validation and key-file creation; validated signed directory
  records at their consumers.
- Protected local HTTP boundaries, removed arbitrary request-key logging, and
  stopped honest stages from exposing hidden-hop details through errors.
- Prevented unsafe model-shard paths, download overflows and partial-file
  collisions; made message buffers grow with received data rather than headers.
- Added adversarial regressions and dependency scans to CI. Removed an unused
  dependency feature; retained one documented, unsuppressed maintenance warning
  for Tor's build-time `paste` dependency.

The [review log](privacy-review.md) records P01–P16, reproductions, changes, and
limits. These are engineering findings, including documentation corrections, not
16 independently audited vulnerabilities. The [setup guide](anonymity.md) explains
how the resulting system works and how to run it.

## Evidence and current readiness

- Final local Windows workspace: **124 tests passed**, zero failed. The two slow
  integrations are separate checks, not included in that number.
- A bounded concurrency run passed **768 checks**, including **4,480** disconnect
  and cancellation cycles, before the later queue/frame changes.
- Private Tor integration passed all return modes on Windows at `062d7f2`; earlier
  Linux CI also passed real Tor, the two-minute relay expiry, and decoder fuzzing.
- Workspace and fuzz-driver RustSec scans found **zero listed vulnerabilities**;
  one maintenance warning remains. This does not audit native code or prove safety.
- **Unresolved:** [CI at `0e730be`](https://github.com/ethan03805/dfin/actions/runs/42404522275)
  failed during the repeated-disconnect fixture's handshake. The cause was not
  established before stopping. Its fuzzing and dependency checks passed. The final
  frame-buffer changes have local coverage; any newly triggered CI must be assessed
  separately. Keep the PR in draft until the failure is understood and checks pass.

## Cleanup and remaining scope

No dfin or Tor test processes remained at handoff. Source changes and documentation
were preserved. Automatic tool policy rejected the cleanup command without a
specific reason, so the six completed test-network directories, generated test
identities, downloaded tools, and build artifacts remain under this chat's `work/`
directory. The directories are `tor-test-01`, `tor-test-02`, and
`privacy-review-tor-01` through `privacy-review-tor-04`. None were deleted by that
command. The isolated llama.cpp checkout also retains the normal generated
layer-range build patch; it is not an extra source change for this PR.

Tor hides network addresses within its threat model. Compute nodes can still
inspect activations, an optional dfin application relay can read model states,
and timing/size correlation, collusion, and Sybil attacks remain unresolved.
Anonymous credits, Monero payment/payout integration, and verified computation
were not implemented. No frontier-scale or geographically distributed benchmark
was performed in this review.

If work resumes, first investigate the Linux handshake failure and validate the
final snapshot. Then address the documented confidentiality, integrity, and
payment-linkability boundaries as separate milestones.
