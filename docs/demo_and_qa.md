# Live demo and Q&A

## Rehearsal sequence (about three minutes)

1. Run `python run_demo.py --T 10 --N 0 --slice 5300 5600`; note printed errors and open `results/demo_map.html`.
2. Run `python run_demo.py --T 10 --N 7 --slice 5300 5600`; compare identical points. Reload the HTML (the file is overwritten).
3. Show `results/error_vs_buffer.png` and explain that T=10/N=7 was selected on validation, not test.
4. Show `results/ablation.png`, including the improvement after removing discrepancy.
5. Use `results/demo_map.png` or `docs/presentation.pdf` as offline backups. HTML requires no tiles or internet. Loading the first time builds a local cache and may take longer; pre-run once before review.

## Answers both members should understand

- **States and observations?** States are directed arcs; observations are projected GPS readings with causal speed/heading estimates.
- **Why an HMM?** Distance alone ignores sequence context; transitions encourage stable, plausible motion, and Viterbi searches the sequence of candidates.
- **What does N mean?** Commit point t-N when point t arrives; N is future readings, about N seconds except across four gaps.
- **Does N always help?** No. Our held-out curves have minima/plateaus, not uniformly convex curves. More delay reduces available committed past inside a fixed window.
- **Why no shortest path?** This project tests the paper's inexpensive local feature approximation. Nearby arcs are not guaranteed reachable; disconnected predictions are a known failure.
- **Exact feature equations?** E1 = exp(-d²/(2 sigma²))/(sigma sqrt(2 pi)); E2 = heading dot local directed tangent; E3 = 0 above 40 mph and >=1.2 road speed, else 1. T1 = 0 on already committed arcs except staying on current arc; T2 = 1/(abs(GPS displacement - midpoint separation)+1).
- **Are these probabilities?** No. SVM coefficients become positive potentials after clipping/normalization; we omit the classifier intercept and use an epsilon floor. This is a disclosed adaptation.
- **What does L1 do?** Encourages sparse coefficients. Balanced class weights correct many more negative candidates than positives. Standardization is fitted on each training fold only.
- **How is leakage prevented?** First 60% training, next 10% selection, final 30% test; ten expanding training folds with a 15-reading gap. GPS features are causal. Route labels are offline references, never decoder inputs. One trip and related roads can still bias evaluation.
- **Where did truth come from?** The published ordered route is aligned to points using monotone distance DP. These are derived proxy labels, not independent surveyed point truth. Inspect both map and route-progress diagnostics.
- **What are the measured results?** T=10, N=7: directed error 9.60%, edge error 4.60%; nearest-road edge error 8.01%. Matcher 0.490 s/1,000 points on this machine; loading/plotting excluded.
- **Why report edge error?** It ignores direction; a geometric nearest-road method cannot resolve equal-distance forward/reverse arcs. Its directed error depends on an arbitrary forward tie rule.
- **Why does dropping discrepancy help?** Whole-road midpoint distance can be a poor displacement proxy for long/curved edges. The test result is descriptive, not used to retroactively select the submitted model.
- **How is backtrack history handled?** A visited set of committed states in the current window preserves first-order Viterbi. Training uses route-label history; inference uses its own commitments. This mismatch is disclosed.
- **Why does T sometimes matter little?** Overlapping committed states are fixed; only N+1 states remain revisable. T also controls the available history for backtrack penalties.
- **What is EOF flush?** Final N states cannot receive full future context. Full error includes them; online-only error excludes them.
- **Who did what?** Use the actual record in docs/contributions.md. Do not claim fictional authorship or artificial commit history.

## Submission actions

Publish all source to a private GitHub repository; add the faculty and assigned TAs, and verify access. Provide section, assigned problem number and repository URL in team.json, then regenerate documents. Submit the PDF; use the PPTX/PDF deck and demonstrate live. The assignment's stated deadline is October 10, 2026, 11:59 PM; confirm current acceptance arrangements with faculty.
