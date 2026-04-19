---
name: benchmark
description: Measure CUDA kernel performance using NSight Compute on the mixed-precision grouped GEMM kernels. Use when the user asks to benchmark, measure performance, check Duration, compare before/after a change, or run NCU. Also use when asked "how fast is it", "what's the current performance", or "run a performance test".
---

# Kernel Benchmark Skill

Accurate performance measurement for the mixed-precision grouped GEMM kernels via NSight Compute.

---

## Primary Method: Use shao_rebuild.sh (MANDATORY)

**Per project CLAUDE.md: always use `shao_rebuild.sh` Task 2 for standard benchmarks.**

```bash
echo "2" | bash ./shao_rebuild.sh
# Output goes to: shao_run_69_lut.log
grep "Duration" shao_run_69_lut.log
```

This is the required method because the script has the correct NCU parameters, kernel-id, warmup count, and problem configuration already tuned for this project.

**Only use the ad-hoc NCU method below when:**
- You need a different problem size than the standard config
- You need to run multiple iterations for variance analysis
- The script is unavailable or broken

---

## Ad-hoc NCU Method (Non-Standard Configs Only)

For non-standard problem sizes or multi-run variance analysis, capture output directly to a variable — no temp files:

```bash
output=$(ncu --clock-control none --set full \
  --kernel-id ::device_kernel:100 \
  ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm \
  --groups=<G> --m=<M> --n=<N> --k=<K> --c=<C> \
  --warmup=10 --iterations=100 \
  --alpha=1 --beta=0 \
  --shuffle=false --explore=false --compare=false \
  --benchmark 2>&1)

duration=$(echo "$output" | grep "Duration" | grep "us" | awk '{print $NF}')
echo "Duration: $duration us"
```

**File discipline**: Use variable capture (`output=$(...)`) — never write to `/tmp/` or create log files. The reason: temp files accumulate and pollute the workspace; variable capture is equally fast and leaves nothing behind.

---

## Multi-Run Variance Analysis

When you need stability data (5 runs recommended):

```bash
durations=()
for i in {1..5}; do
  output=$(ncu --clock-control none --set full \
    --kernel-id ::device_kernel:100 \
    ./build/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm \
    --groups=32 --m=32 --n=5888 --k=2944 --c=32 \
    --warmup=10 --iterations=100 --alpha=1 --beta=0 \
    --shuffle=false --explore=false --compare=false --benchmark 2>&1)
  dur=$(echo "$output" | grep "Duration" | grep "us" | awk '{print $NF}')
  durations+=($dur)
  echo "Run $i: $dur us"
done

python3 -c "
data = [${durations[*]}]
mn, mx, avg = min(data), max(data), sum(data)/len(data)
var = (mx - mn) / mn * 100
print(f'Min: {mn:.2f}  Max: {mx:.2f}  Avg: {avg:.2f} us  Variation: {var:.1f}%')
stability = 'stable' if var < 3 else 'HIGH VARIANCE - investigate'
print(f'Assessment: {stability}')
"
```

---

## Standard Problem Sizes

```bash
# MXFP4 x BF16 (standard benchmark)
--groups=32 --m=32 --n=5888 --k=2944 --c=32

# Small correctness test
--groups=1 --m=1 --n=128 --k=256 --c=32

# INT4 x FP8 medium
--groups=32 --m=16 --n=4096 --k=7168 --c=128

# Large throughput
--groups=128 --m=16 --n=1536 --k=2048 --c=128
```

---

## Understanding the Output

- **Duration metric format**: `Duration   us   154.50`
- This is the actual kernel execution time, NOT total program runtime
- NCU provides more accurate timing than CUDA events (which include launch overhead)
- Normal run-to-run variation: <2% is stable; >4% indicates contention or thermal throttling

---

## Presenting Results

```
| Run | Duration (us) |
|-----|--------------|
| 1   | 153.1        |
| 2   | 152.9  (min) |
| 3   | 155.3  (max) |
| 4   | 153.1        |
| 5   | 153.6        |

Avg: 153.6 us | Variation: 1.5% | Stable
Config: groups=32, m=32, n=5888, k=2944, c=32 (MXFP4xBF16)
```
