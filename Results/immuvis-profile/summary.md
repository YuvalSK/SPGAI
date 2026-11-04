# ImmuVis training on a Kaggle T4: why it crashed, and the fix

**Runs:** `immuvis-profile` v1 and v2, 2026-10-13. Notebook `Scripts/ImmuVisProfile.ipynb`, built by the same builder as `ImmuVisJackson*.ipynb`.

**Problem:** two training runs (`immuvis-jackson` v1 and `immuvis-jackson-train` v1) died: "Kernel died", with no outputs saved.

**v1 (profiler and memory trace):**
- Host RAM grew about 1.2 GB per 25 steps and never levelled off (`immuvis_memory_trace.csv`).
- Each step took about 8 s, and most of that was CPU time inside the cuDNN convolution calls (`immuvis_profiler_top_ops.txt`).
- At that rate RAM reaches the 29 GB limit after about 430 steps, before the first validation at step 500. That matches the crash.

**v2 (each setting run in its own process for up to 6 min; `immuvis_profile_v2.csv`):**

| Setting | s/step | Host RAM (start → end) |
|---|---|---|
| As before (cuDNN on, 2 GPUs, batch 16, fp16) | 6.18 | 3.6 → 5.9 GB in 59 steps |
| cuDNN plan cache limited to 50 | 6.16 | 3.6 → 6.0 GB |
| Fixed channel counts (shapes constant), cuDNN benchmark | 8.14 | 3.7 → 5.4 GB in 45 steps |
| **cuDNN off** | **0.58** | **2.39 → 2.39 GB in 120 steps** |
| Fixed channel counts, cuDNN off | 0.56 | flat |
| cuDNN off, 1 GPU | 0.95 | flat (GPU peak 10.5 GB) |

**Conclusion:**
- The slowdown and the leak both come from cuDNN on the T4 with this model.
- The changing tensor shapes caused by random channel masking are not the cause: fixed shapes are just as slow with cuDNN on.
- **Fix:** `torch.backends.cudnn.enabled = False`. It is about 11× faster, RAM stays flat, and the published masking recipe is kept unchanged.
