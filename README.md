# pgolf-8xh100

A Kupanga benchmark for [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf), main track
(`track_10min_16mb`): the lowest val_bpb on the FineWeb validation set from a model that trains in 10 minutes on
**8×H100 SXM** and fits in a **16,000,000-byte** artifact (the bytes of `train_gpt.py` plus the compressed model).
Each scoring runs the candidate on three seeds, and its parent's code once on the same machine.

| File | What it is |
|---|---|
| `eval.toml` | Runs `python bench.py`; `val_bpb` is the primary metric, lower is better |
| `job.toml` | `python:3.12`, 8×H100 (GCP a3-highgpu-8g, spot, 3 retries), 32+ CPUs, 256+ GB, 150 min limit |
| `requirements.txt` | torch 2.9.1+cu128, numpy, sentencepiece, zstandard, brotli (the same as the subject's) |
| `bench.py` | Downloads the data, rebuilds the parent, runs the seeds, measures the runs, checks the runtime rules |

## What it does

1. **Picks the data variant** from `train_gpt.py`'s default `DATA_PATH`: `fineweb10B_sp1024` (the baseline's) or
   `fineweb10B_sp4096`.
2. **Downloads that variant's data** once, pinned to a Hugging Face revision:
   - sp1024: `willdepueoai/parameter-golf` at a85b0e6, with the val shard and 80 train shards (~16 GB, the
     downloader's default)
   - sp4096: `kevclark/parameter-golf` at c2ba4d2, with the val shard and all 143 train shards (~29 GB)

   It also downloads the tokenizer. The files go into a read-only folder laid out as
   `data/cached_challenge_fineweb.py` leaves it, and each shard's header is checked.
3. **Rebuilds the parent's code** by reverse-applying `$HARNESS_CHANGES_FILE` to a copy of the subject. The
   base node has no parent.
4. **Runs** `torchrun --standalone --nproc_per_node=8 train_gpt.py` for seeds 1337, 42 and 2024, and once for the
   parent with seed 1337, in the order c p c c. The environment carries `DATA_PATH`, `TOKENIZER_PATH`,
   `PGOLF_DATA_ROOT`, `RUN_ID`, `SEED`, `MAX_WALLCLOCK_SECONDS=600` and `NCCL_IB_DISABLE=1`. The output is
   read line by line and timestamped here.
5. **Measures each run:**
   - **val_bpb:** the last `<label>_exact val_loss:X val_bpb:Y` line
   - **train_time:** the run's own last `train_time:Nms`. As on the leaderboard, compiling, the warmup steps
     whose weights are restored, and periodic validation are excluded.
   - **pre_serialize_s:** train_time plus the wall time from the last training line (`step:`/`stopping_early`) to
     the first `Serialized model` / `Total submission size` line, less any evaluation the run logs in between
     (`eval_time:Nms`, e.g. a pre-quantization check). Calibration on data after the clock and before that line
     shows here. Calibration after it (the baseline's first such line is the raw `final_model.pt`) doesn't: that is
     left to pgolf-smoke's audit (a training shard opened after the clock with no time reserved for it) and to the
     code review.
   - **eval_time:** the wall time from the last training line to the run's exit (export plus evaluation)
   - **the artifact:** the size of `train_gpt.py` plus the largest `final_model*.ptz` the run wrote, both
     measured on disk, then compared with the last `Total submission size …: N bytes` line
   - **the scored set:** val_bpb·ln2/val_loss is the run's scored tokens per byte. It must equal the tokens per
     byte of a prefix of the validation targets that leaves at most 16,384 unscored, computed here from the
     shard and the tokenizer with the baseline's byte counts. The `_exact` line's 8 decimals pin it to about
     1e-8. Skipped tokens, miscounted bytes, and twice-scored tokens (unless their bytes per token are exactly
     average) all show.
6. **Reports** the mean val_bpb and its standard error, each seed's val_bpb, pre-quantization val_bpb, the
   largest artifact, model and times, the parent's val_bpb on the same machine and the delta, and `meets_rules`.

**meets_rules** is 1 only if every candidate run meets all of these:
- the artifact is ≤ 16,000,000 bytes and equals the logged total
- train_time ≤ 600 s (+1 s for the step that crosses the cap)
- pre_serialize_s ≤ 600 s (+5 s for saving and logging)
- eval_time ≤ 600 s
- the score line comes after the size line, so the score is that of the round-tripped artifact
- the scored set is the whole validation split
- the run exited cleanly after printing its score

A miss is reported with `meets_rules` = 0 and the reason in `rule_misses`, not raised. A run with no score line
(a crash before it, or a timeout) raises, with the first traceback. Downloads are retried, truncated shards
included.

## Rules not checked here

`pgolf-rules` checks the static rules before this runs:
- one file and no `exec`
- no network imports or URLs
- the data, tokenizer and variant
- no validation data in a training loader
- the metric functions unchanged
- `MAX_WALLCLOCK_SECONDS` and `SEED` read

`pgolf-smoke` runs a short training on one H100 with an audit hook for network use and for training-data reads
after training. This eval trusts the run's own `train_time` and validation, as the challenge does, and doesn't
re-score the artifact itself.

## Settings

| Variable | Default | |
|---|---|---|
| `KX_SEEDS` | 1337,42,2024 | Candidate seeds (the challenge's usual three) |
| `KX_BASELINE_RUNS` | 1 | Parent runs |
| `KX_TRAIN_SHARDS` | the variant's (80 / 143) | Training shards to download |
| `KX_RUN_TIMEOUT_S` | 2100 | One run: startup and compiling, 600 s of training, periodic validation, up to 600 s of evaluation |

If the time limit gets close, the eval stops starting new runs and scores what it has, as long as it has at
least 2 candidate runs.

## Notes

- The baseline's published result is val_bpb 1.2244, with an artifact of 15,863,489 bytes. That run used an older
  validation export (63,779,840 tokens) and 25 train shards. Every later record used this export (62,021,632
  scored tokens) and, like this eval, 80 shards. So the base can land a little away from 1.2244.
- Checked offline against the challenge's records: all 49 distinct round-trip `_exact` lines in the sp1024 logs
  match a full-split prefix. Every sliding-window line fails at 0.41052189 or 0.41052133 tokens/byte. Those
  records slide windows up to the end of the stream and score each short tail window's last 64 targets, so the
  last 64 tokens are scored 16 (seq 1024) or 32 (seq 2048) times. A simulation of exactly that reproduces both
  ratios. A correct sliding window passes.
- The sp4096 record (2026-04-01) uses a fixed sliding window, and all three of its final lines pass: tokens/byte
  0.30123378 is the first 45,508,608 targets, its own logged val_tokens. The validation split's byte total is the
  same for both tokenizers (151,080,891), as it should be for the same documents.
- Local checks, no GPU:
  - `python bench.py --parse <run.out> ...` parses saved outputs.
  - `python bench.py --reference <fineweb_val_000000.bin> <tokenizer.model> [val_loss,val_bpb ...]` prints the
    validation split's token and byte totals and checks scored lines.
