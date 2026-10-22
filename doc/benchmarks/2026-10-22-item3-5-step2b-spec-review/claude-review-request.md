# Review the 2b preflight questions before implementation

User-authorized Claude review. Read only: do not run tests or experiments,
edit files, implement or publish. Give recommendations for Alec, not
decisions on his behalf.

2a has landed on both mains: BasicModel `0b030e3eb`, WikiOracle `baa2e7f`.
Its frozen source was `d4f316411`; your separate review approved it after
223 passes and seven existing skips. This request is the next round's
spec review. Production and tests still exactly match that publication.

Read `doc/plans/2026-10-21-step2-one-ltm.md` §§10–12, §§16–18,
this receipt's README and `review-input.json`, and the cited source/test
locations. The latest handoff explicitly requires dated questions and
recommendations for the three §16 representations, then a stop for you
and Alec before implementation. It also requires proposing the
coarsening fixture port when refs are replaced, and returning any
assertion depending on the implicit negative-pair swap to Alec.

Please review these five items separately:

1. Can the current standalone leaf-posting wire contract remain while a
   complete owner rebuilds from tagged LTM1 reading rows? Check the
   malformed/stale-cache boundaries and whether the recommendation adds
   any second authority or new compatibility layer.
2. Are the adjacency data, their identifiers, consumers and positional
   witness requirements described correctly? Is retaining the current
   witness columns in 2b a sound recommendation pending a specified
   LTM1 layout, rather than pretending they are already derived?
3. Verify that `unspecified` is produced by the live close as well as
   migration. Does the decided question layout alone make the mode
   column derivable for the accepted API? Assess the recommendation to
   retain the three-mode column without inventing an extra flag.
4. Are the two proposed expected tuples precisely the consequences of
   Alec's §16 polarity/evidence decision, with every assertion body
   preserved? Identify any additional existing expectation depending on
   the swap, if apparent from inspection.
5. Is the coarsening case's conflict with §§10–11 real, or is there an
   interpretation that preserves its semantic assertions under the
   decided graph/protection rule? Check the proposed replacement behavior
   (parent/child and references retained, no in-pass cascade/coarsening;
   separate next-pass release case). This must not be mislabeled a simple
   column-name port if the expected behavior changes. Flag any overlooked
   assertion authorization or layout question before implementation.

Correct wrong code facts with locations. Give a concise recommended
disposition and the exact choices that still require Alec. Do not reopen
the already decided owner, tags, whole-entry addresses, DAG or transitive
protection. No 2b source is frozen or tested yet; no learning claim and no
item-3.0 transition. The two R10 arrival notes stay in 2c, and bounded
recall's `SentenceEndState.where` observation stays with the 5.5 spec.
