# Twelve first draws on 575cab06c

All twelve pass the revised correct-or-open certificate. No checked x is bound; all retain their named open referent. These are mechanism results, with no learning claim.

| Draw | Result | Bound | Open |
|---|---|---:|---:|
| fresh-final-forward-1 | pass | 0 | 16 |
| fresh-final-forward-2 | pass | 0 | 16 |
| fresh-final-forward-3 | pass | 0 | 16 |
| fresh-final-forward-4 | pass | 0 | 16 |
| fresh-final-forward-5 | pass | 0 | 16 |
| fresh-final-forward-6 | pass | 0 | 16 |
| fresh-final-reverse-1 | pass | 0 | 32 |
| fresh-final-reverse-2 | pass | 0 | 32 |
| fresh-final-reverse-3 | pass | 0 | 32 |
| fresh-final-reverse-4 | pass | 0 | 32 |
| fresh-final-reverse-5 | pass | 0 | 32 |
| fresh-final-reverse-6 | pass | 0 | 32 |

Every draw retains the initial process RNGs, private generator state, initial model hashes, native read captures, certificate and both-trial keep probes from the outset. Closing reference credit is absent in every probe.

Forward draw six completed inside pytest and wrote its passing worker result, outcome and certificate. Its outer runner stopped before finalizing result.json. That incomplete runner record is preserved, with fresh-final-forward-6-recovery.json documenting the worker evidence; the draw was not rerun or replaced.
