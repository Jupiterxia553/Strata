# Community benchmark on AMD Radeon RX 6900 XT

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 6900 XT, 16 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 127.9 GiB; storage: GALAX HA1K1000N SSD/NVMe
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.21045.16010; CUDA (driver supports up to): not read
- Strata: project 0.1.57, commit 5d37280d; engine 0.1.57
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:16384 --spec 4 --spec-min-p 0.7 --mtp rt --max-context 262144 --kv k8v4 --kv-resident 32768 --vram-reserve-mib 1500 --prompt-cache 2 --conversation-cache-mib 8192 --conversation-cache-slots 32 --expert-profile-save expert-profile-learned-iq3_s.bin
```

Environment of the config: `STRATA_DENSE_MMQ=1`, `STRATA_HIP_ADAPT_KERNEL_COPY=1`, `STRATA_HIP_PROMPT_F16=1`, `STRATA_IO_THREADS=32`, `STRATA_KV_PREFETCH=1`, `STRATA_ROCBLAS_TUNING=hip`, `STRATA_ROUTE_TAIL_SKIP=7`, `STRATA_HC_PERSIST=1`, `STRATA_SELECT_SIMT=1`, `STRATA_ROUTE_PRIOR=0.75`, `STRATA_PREFILL_LEND_PCT=70`, `STRATA_KVA=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 64.3 (56.2-79.3) | 0.50 (0.44-0.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 999.9 (964.0-1008.2) | not measured | 4.05 (4.01-4.20) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1296.9 (1291.3-1302.9) | not measured | 12.42 (12.38-12.47) |
| agent session, follow-up turns | 13286 (13286-13305) at the last turn | see below | up to 64 | 3 | see below | 77.3 (63.0-88.4) | 3.86 (3.81-3.91) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.5-69.5); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12415 (12415-12416) new tokens of the files are read once); first turn (nothing cached) first token: 3.31 s (3.28-3.33).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 187.0 | 3.282 | 53.2 |
| 1 | 3526 | 632 | 2894 | 756.4 | 3.857 | 63.0 |
| 2 | 5059 | 3589 | 1470 | 389.1 | 3.809 | 66.8 |
| 3 | 8303 | 5122 | 3181 | 832.2 | 3.856 | 71.5 |
| 4 | 10531 | 8366 | 2165 | 567.8 | 3.844 | 79.1 |
| 5 | 13286 | 10576 | 2710 | 710.0 | 3.861 | 74.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 163 s.
