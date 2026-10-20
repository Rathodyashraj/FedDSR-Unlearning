# Unlearning: hand-over for Stage II (GPU machine)

This folder turns the trained 4-client FedDSR model into an unlearned model for one withdrawn
client (0 = COVID-19, 1 = pancreas, 2 = kidney, 3 = brain stroke). Stage I (this code) was written
and tested on a CPU laptop against a small synthetic stand-in, the *fixture*. Stage II runs it on the
real trained model. The fixture proved the code is correct; how well the method works is only known
after Stage II.

## 1. What to put on the GPU machine

| What | Where it comes from |
| --- | --- |
| The code folder | `git clone git@github.com:Rathodyashraj/FedDSR-Unlearning.git`. It holds the training code with the changed `models.py` and `server.py`, the Stage-2 script, and this `unlearning/` folder. |
| The trained model | Either copy `FedDSR_results/` from the training machine, or retrain (section 3). |
| The datasets | At the paths in `config.py` (`../dataset/...`), the same as for training. Steps 1, 3, 4 and 5 read the remaining clients' training slices. |

Not needed: `.venv/` and `FedDSR_results_fixture/` (laptop only). `UNLEARNING_PLAN.md` is not in git; copy it by hand if you want it.

Do not rename `Stage2 decoder finetunning.py`: Step 5 loads it by that exact name.

## 2. Environment

The packages the training code imports: `torch`, `torchvision`, `lpips`, `piq`, `torchmetrics`, `torch-fidelity`,
`scipy`, `opencv-python`, `pydicom>=3`, `scikit-image`, `pandas`, `matplotlib`, `tqdm`, `pillow`, `numpy`.
lpips and FID download their pretrained weights on first use (internet once).

**If the model was copied rather than retrained, use the same Python environment the model was trained in.**
Step 0 then regenerates the initial weights from the seed, and the random draws depend on the library versions.

Run everything **from the project folder**, as a module: `python -m unlearning.<script>`.
The device comes from `config.DEVICE` (CUDA when available).

## 3. Commands, in order

### 3.0 Only if you retrain (instead of copying `FedDSR_results/`)

```
python main.py --rounds 30 --clients 4 --gpu 0
python "Stage2 decoder finetunning.py" --gpu 0 --epochs 10 --clients 4
```

The updated `server.py` now saves `FedMedSR/round_0_global_model.pth` before round 1, so Step 0 will take Case 1.

### 3.1 Inventory: what is in the results folder

```
python -m unlearning.inventory --results-dir FedDSR_results
```

Prints one `[x]`/`[ ]` line per expected file, each client's deployed file (Stage-2 refined if present),
`Step 0: Case 1` or `Case 2`, the sizes read from the model (expect codebook `1024 x 128`, grids `[64, 64]` and
`[128, 128]`, patch `1153`, encoder weights `5,193,280`), and ends with `Wrote FedDSR_results/Unlearning/inventory.json`.

### 3.2 Step 0: initial weights (once per trained model)

```
python -m unlearning.step0_initial_weights --results-dir FedDSR_results
```

* Case 1 (round-0 file present): `Wrote .../Unlearning/phi0/phi0.pth (34 tensors, 5,193,280 weights)`.
* Case 2 (copied model): also prints `Against round 1 over 5,193,280 weights: regenerated r=..., r*sqrt(N)=...;
  control ... -> PASS`. The pass rule is `r*sqrt(N) > 10` for the regenerated weights and `|r*sqrt(N)| < 5`
  for the control. **If it prints FAIL, phi0.pth is not written: stop.** Rerun in the original training
  environment, or retrain once (section 3.0). Never use the round-1 checkpoint as the anchor.

### 3.3 Checks on the real model

```
python -m unlearning.checks --results-dir FedDSR_results --trial
```

`--trial` uses 64 slices per client and the short schedule; without it several checks run complete steps
on all data (the Phase 10 smoke test runs the whole pipeline once per withdrawn client). Ends with
`ALL CHECKS PASSED (n skipped)`. Expected skips: the comparison with the original `models.py` (enable it with
`--original-models /path/to/unmodified/models.py`), and in Case 2 the two Step 0 checks that need the round-0 file.
The checks write their outputs into the `_trial` withdrawal folders and into `Unlearning/phi0/`.

### 3.4 Trial run on real data

```
python -m unlearning.run --results-dir FedDSR_results --withdraw 3 --trial
```

64 slices per client, 1 round of 2 shrinks. Output goes to `Unlearning/withdraw_3_Brain_Stroke_CT_trial/`,
so it never blocks the full run. Ends with `Unlearning run complete: step 1 .. s, ..., step 6 .. s`.
**Read the Step 3 lines before the full run** (see 5.1).

### 3.5 Full run

```
python -m unlearning.run --results-dir FedDSR_results --withdraw 3
python -m unlearning.run --results-dir FedDSR_results --withdraw 3 --resume     # after an interruption
```

`--resume` reuses every finished step and continues Step 3 from its last finished round (bit-identical to an
uninterrupted run). Other flags: `--steps 4,5,6` runs only those steps, reusing earlier outputs;
`--overwrite` redoes steps whose output exists. A finished run is refused without `--overwrite`.

Each step can also be run on its own: `python -m unlearning.step1_statistics ... step6_release`, each with
`--results-dir` and `--withdraw`.

### 3.6 Clean-up (optional, off by default)

```
python -m unlearning.step6_release --results-dir FedDSR_results --withdraw 3                  # lists only
python -m unlearning.step6_release --results-dir FedDSR_results --withdraw 3 --purge          # deletes in Unlearning/
python -m unlearning.step6_release --results-dir FedDSR_results --withdraw 3 --purge --purge-training-files
```

The last form deletes the old model in the training folders. **Do not use it while the original model is still
needed for comparison.** `round_0_global_model.pth`, `phi0.pth`, `metrics.csv`, the logs and `release/` are kept.

## 4. Where the outputs land

`FedDSR_results/Unlearning/`:

| Path | Content |
| --- | --- |
| `inventory.json` | Inventory of the results folder |
| `phi0/phi0.pth`, `phi0/phi0_info.json` | Initial encoder weights; shared by all withdrawals |
| `withdraw_<id>_<name>/step1_stats/stats_client_k.pt` | Counts, sums and median margins per remaining client |
| `withdraw_<id>_<name>/step2_server/server_quantities.pt` | Weights, clean codebooks, masks, margins |
| `withdraw_<id>_<name>/step3_rounds/` | Per round: global encoder, optimizer states, `state.json`; `encoder_unlearned.pth`, `step3_summary.json` |
| `withdraw_<id>_<name>/step4_fit/` | `fit_client_k.pth`, `fit_summary.json` |
| `withdraw_<id>_<name>/step5_finetune/` | `finetuned_client_k.pth` |
| `withdraw_<id>_<name>/release/` | **The unlearned model**: `best_model.pth` (with masks), `unlearned_client_k.pth`, `phi0.pth`, `summary.json` |
| `withdraw_<id>_<name>/logs/` | `run.log` (or one log per step) and `repairs.jsonl` (one line per repair) |

`FedMedSR/` and `DecoderFineTuning_Local/` are never written to; the code refuses to.

The released files load with the existing code: Stage 2's `load_stage1_model` and `VQVAE2.load_shared_state_dict`.

## 5. What Stage I found, and what to watch in Stage II

### 5.1 Repairs did not keep up on the fixture

On the fixture every repair stalled: the old encoder's latents miss their target row in the clean codebook at
20 to 33 % of positions, and repairs reduce that only slightly within one pass. Even with the old codebook as
target, the margin term alone made Adam's steps (about the learning rate, 5e-5, per weight) cross the fixture's
tiny margins (0.003 to 0.009). So Step 3 went straight to the closing round, with 0 shrinks and leftover rho = 1.
This is the plan's listed risk "repairs that cannot keep up"; the fixture model is barely trained, so it may not
happen on the real model. **In the trial run, look at** the `round 1 shrink 0: ... -> done/STALLED` lines in the
log and at `shrinks`, `stalled` and `leftover_rho` in `step3_rounds/step3_summary.json`. No setting was changed;
if it stalls on the real model, report it rather than change settings silently.

### 5.2 The decoder fit changes the entry layers a lot

On the fixture the fit cut the feature error by 55 to 80 %, but the two entry layers changed by 60 to 135 % of
their size: the small ridge barely restrains directions the code patches do not vary in. Watch
`feature error ... -> ...; layer changed by ...` in the Step 4 log and `fit_summary.json`.

### 5.3 Choices made where the plan left room

* Step 0 control: "near 0" is taken as `|r*sqrt(N)| < 5` (fixture control: 2.57).
* A stalled round-1 alignment repair counts as a stall: no shrink on an unaligned encoder.
* A repair that reaches the cap in the middle of a 50-batch block is judged on that partial block.
* Each client's shuffled loader restarts every round with a seed set by client and round; this makes `--resume` exact.
* Step 4 clears the vitalization usage counters copied from the deployed file.
* The clean-up list also covers the sample images the training code saves in `round_r/` and
  `DecoderFineTuning_Local/`, which include the withdrawn client's test scans.

### 5.4 Other things to know

* Step 1's check requires every training slice to be read. The training loader silently drops images that fail
  to load or are blank; if any are, Step 1 logs a warning and that check fails, showing how many were lost.
* Memory: distances and the hinge are computed in blocks of `DISTANCE_CHUNK_POSITIONS` (32,768) positions in
  `unlearning/settings.py`. Lower it if the GPU runs out of memory; results do not depend on it (checked).
* Expected time: Step 3 dominates (worst case about 32 repair passes of up to one epoch each per client);
  Steps 1 and 4 are one encoder pass each; Step 5 is two Stage-2 epochs. On the fixture a whole run took about 70 s.
* In the existing decoder, the cross-attention query and key projections never train (the keys are pooled to a
  single vector, so the attention is always 1). This is a property of `models.py`, also in the original Stage 2.
