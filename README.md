# train_experiment

Run: **main_1.py / flow_3_4.json / seed=0 / 3.2 M env-steps / Dueling+Noisy, PER, n-step=3**

## What is here

| Path | What |
|------|------|
| `code/main_1.py` | Trainer/evaluator (identical to the file that produced the run) |
| `checkpoints/final_improved2.pt` | Final checkpoint (eval R = +7.518) |
| `checkpoints/final_improved2.pt.best` | Best-eval checkpoint (eval R = +7.804 @ step 76 800) |
| `logs/training_run.log` | Raw console transcript of the run |
| `logs/train_log2.csv` | Per-iteration metrics (step, loss, Q, R_dec, R_ep, eval_R, baselines) |
| `logs/train_log.jsonl` | Same as above in line-delimited JSON |
| `plots/final_improved2_curves.png` | 6-panel training dashboard |
| `configs/flow_3_4.json` | Flow file used (14 boundary approaches, 12 intersections) |
| `analysis/multi_seed_eval.py` | Paired-seed evaluation of agent vs. baselines |
| `analysis/convergence_analysis.py` | Plateau fit + CUSUM change-point on eval_R |

## Reproduce the run

From this folder:

    python code/main_1.py \
      --flow-file configs/flow_3_4.json \
      --total-steps 3200000 --num-envs 128 --batch-size 256 \
      --buffer-size 200000 --ckpt checkpoints/final_improved2.pt \
      --log-csv logs/train_log2.csv --log-jsonl logs/train_log.jsonl \
      --eval-every 200 --eval-steps 100 --log-every 300 \
      --ckpt-every 5000 --n-step 3 --min-green 2 --seed 0 --plot-after

## Next step: statistics (not retraining)

    python analysis/multi_seed_eval.py \
        --ckpt checkpoints/final_improved2.pt.best \
        --flow configs/flow_3_4.json --seeds 30 --steps 1000

    python analysis/convergence_analysis.py --csv logs/train_log2.csv

Both are read-only — they do not modify the checkpoint or logs.

## Known caveat

Demand in `flow_3_4.json` is ~7 veh/s over 12 intersections (≈14× under capacity).
Under that regime fixed-time, max-pressure and the DQN agent all sit within ~5 %
of the work-conserving reward ceiling, so **single-seed point estimates of % gain
are within eval noise**. `multi_seed_eval.py` quantifies this explicitly.
