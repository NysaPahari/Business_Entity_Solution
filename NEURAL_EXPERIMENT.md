# Learned text matcher — active follow-up

## Completed result — 2 October

The GPU pilot completed successfully. On the same 500 calibration comparison
entities: baseline macro F0.5 0.9463226919 (32 FP,185 FN); neural residual
0.9432432309 (25 FP,203 FN). Three epochs were evaluated on the fit half;
epoch3 was selected before the tune comparison. The neural model has 69,137
parameters. It was not promoted. Results: artifacts/accuracy/neural_probe/results.json.
The user has been notified. This experiment branch is complete; do not relaunch
the finished launcher or training. Continue the production-submission branch
until validated improved outputs and the clean ZIP are ready, then disable the
heartbeat after reporting completion. The user's latest packaging requirement
is minimal files, specified folder layout, and code comments/docstrings <=2 lines;
src.package_improved now invokes src.clean_submission to enforce it.

## Latest status: heartbeat implementation

Dependencies installed successfully; CUDA verification passed. Implemented
`src.neural_probe`, a small FROM-SCRATCH byte-level cross-encoder residual model
(not pretrained, and not a learned-retrieval encoder). It sees both names and
addresses with bounded per-field byte budgets, uses two transformer layers,
and learns a correction to the frozen base model's log-odds. Training is the
same 1,000 previously unused training entities as group_probe. Training sampling
keeps all retrieved positives, eight hard negatives and up to eight uniformly
sampled remaining negatives per query, with sampling weights. No truth-positive
injection into retrieval. Three epochs and cutoff are selected on calibration
fit half; final comparison uses the tune half. Fresh audit remains untouched.

Syntax compilation passed. Runtime/training and model quality remain unverified.
`src.launch_neural_when_ready` has been started using the pipeline environment.
It waits until at least 2 GiB host RAM is available, then launches the isolated
neural environment. Inspect actual processes for both modules before relaunching.
Do not duplicate this waiting launcher. Status files:
`artifacts/accuracy/neural_probe/launcher_status.json` and `status.json`.
Logs: `work_accuracy_test/neural_probe.log`, `neural_launcher.log`, and
`neural_launcher_errors.log`. Results will be `results.json` in the same artifact
directory. On failure, inspect the recorded traceback, fix, and relaunch safely.
Waiting for RAM is expected while production retrieval consumes memory; do not
stop production to force this pilot to run. No new measured neural score yet.

The user authorized work on learned retrieval/text matching and notification on
completion. An hourly thread heartbeat named `Finish matcher experiment and
submission` is active (automation ID `finish-matcher-experiment-and-submission`).
Stay quiet for unchanged status; notify on measured completion or actionable
failure. Do not create duplicate heartbeat or training jobs.

Hardware verified: NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM, driver 610.47.
Free host RAM was about 1.77 GB while production inference was running.
Keep that inference and its selected model intact. GPU training must use a
small model, short sequences, small batches, and bounded data loading.

Status: isolated `.neural-venv` created. `setup_neural.ps1` installs PyTorch 2.6
CUDA12.6 and Transformers, verifies GPU access, and saves resolved dependency
versions. Logs: `work_accuracy_test/neural_setup.log` and
`work_accuracy_test/neural_setup_errors.log`. Check actual processes before
retrying setup. A completed environment is NOT a trained model.

First setup failed due to sandbox network error WinError10051. An escalated
retry was launched; its output is `work_accuracy_test/neural_setup_retry.log`
(tool session79582, not reliable across turns). Inspect that log and live setup
processes before taking further action. Do not interpret the first failed log
as the status of the retry.

Next work: implement a small cross-encoder text-pair pilot, with complete name
and address from both records, trained on supplied training labels and hard
negatives retrieved without labels. If using pretrained weights, verify Apache
2.0/MIT license and save model revision; never use external business data or
entity lookup. Fit only on train; choose epochs/cutoff on disjoint calibration
partitions and leave the exposed fresh audit untouched. Compare against the
frozen LightGBM on exactly the same entities and candidates, including empty
candidate groups. Require a material repeatable gain before promotion. Learned
retrieval is a subsequent stage if the text matcher shows useful signal.

`src.group_probe` and `work/group_probe.duckdb` contain 1,000 previously unused
training entities and 1,000 calibration entities suitable for a smoke pilot;
larger training is required for a credible neural comparison. Do not mistake
that small pilot for the final evaluation. All current measured results and
production resume instructions are in HANDOVER_CURRENT.md.

No neural model has been trained and no new score exists yet. Best independent
fresh-audit F0.5 remains 0.94547987. Report measured outcomes honestly, including
failure to improve. Continue production monitoring and validated ZIP generation
as well; completion of one branch does not complete the other.
