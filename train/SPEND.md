# Vast.ai spend log (open-revisualizer GPU training)

Cap for this job: USD 5.50 total; stop if account credit would fall below USD 3.80.
Never touch instance 57631969 (not ours; it stayed exited and untouched the whole time).

| When (UTC) | Event | Instance | Rate | Credit after |
|---|---|---|---|---|
| 2026-11-22 00:35 | Start: credit check | - | - | 9.364 |
| 2026-11-22 00:36 | Rented 1x RTX 4090, Texas US, offer 48986252, 80 GB disk, pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel | 61883072 | 0.447 USD/h incl. storage | - |
| 2026-11-22 01:07 | Check during training (setup done, 30k-iter run started 01:03) | 61883072 | | 9.162 |
| 2026-11-22 01:47 | Check during training | 61883072 | | 8.860 |
| 2026-11-22 02:13 | Export copied to the Mac and checked (md5 match); instance destroyed, API lists only 57631969 | 61883072 | | 8.668 |

**Total: about USD 0.70** (9.364 -> 8.668, about 1 h 37 min of box time; this also includes the small storage
charge for 57631969 over the same period). Remaining credit: USD 8.67.

## HD retrain (native 2048x1550)

Cap for this job: USD 2.50; stop if account credit falls below USD 3.00 (another agent spends in parallel on its own box,
61894207, so the credit drops faster than this job alone). Never touch 57631969 or any instance not created here.

| When (UTC) | Event | Instance | Rate | Credit after |
|---|---|---|---|---|
| 2026-11-22 02:31 | Start: credit check (other agent's 61894207 running) | - | - | 8.518 |
| 2026-11-22 02:33 | Rented 1x RTX 4090, Sweden, offer 53250552, 171 GB RAM (full-res images need ~70 GB host RAM), 100 GB disk, pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel | 61896562 | 0.452 USD/h | - |
