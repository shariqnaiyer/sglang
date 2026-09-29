# Roadmap: from zero to SGLang contributor

## Where you're starting

| Area | Level | What it means for the plan |
|---|---|---|
| Programming | Python (built a web backend), some Rust | Strong base, so Stage 1 is skipped |
| Git / command line | Daily use | Skipped; `ssh`, `tmux`, `nvidia-smi` and `perf` get added when needed |
| Python depth | Classes, async, type hints: yes. Multiprocessing and profiling: no | Stage 2 is short and targets the gaps |
| Data structures | Know them, would need to look things up to implement | Build a trie and an LRU cache in Stage 2 (they lead to the radix cache) |
| Math | Matrices, calculus and probability all "vaguely remember" | Refresh each one just in time (Stages 3–4), no separate math stage |
| Hardware | RTX 5060 Ti 16 GB (Blackwell, sm_120), CachyOS Linux | Every stage runs locally. Needs CUDA 12.8+ builds of PyTorch |
| Time | ~20 h/week | ≈ 22 weeks (about 5 months) to a first PR |

## How each stage works

1. **Learn**: a short explanation from me, plus 1–2 resources.
2. **Build**: you write the code and I review it. You get hints before answers.
3. **Predict, then measure**: before every experiment, you write down what you expect.
4. **Checkpoint quiz**: you must pass it to move on. If you don't, I re-teach the material a different way and quiz you again.

Put all your code in one repo on your machine (for example `~/inference-journey/`, one folder per stage). It becomes your portfolio.

---

## Stage 1: Programming foundations. **Skipped**
Your git, shell and Python are already at the level this stage covers.

## Stage 2: Python depth (≈1 week, 20 h)
**Concepts:** the GIL; threads vs processes vs asyncio and when to use each; `multiprocessing` and inter-process queues; ZeroMQ basics (SGLang's processes talk to each other over ZMQ); `cProfile`, `py-spy`, `time.perf_counter`; implementing a trie and an LRU cache.
**Project:** a 3-process "fake LLM server":
`frontend (asyncio HTTP) → worker ("generates" tokens with sleep + CPU work) → detokenizer → back to the frontend`,
connected with ZMQ. Profile it, find the bottleneck, and fix it. Separately, build a trie-based prefix cache with LRU eviction and write tests for it.
**Resources:** Python docs on [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) and [`asyncio`](https://docs.python.org/3/library/asyncio.html); [ZeroMQ guide, ch. 1–2](https://zguide.zeromq.org/).
**Checkpoint quiz:** why can't threads speed up CPU-bound Python? When is asyncio the right choice? Read a py-spy flame graph I give you and name the hotspot. What is the time complexity of a trie lookup?

## Stage 3: PyTorch basics (≈2 weeks, 40 h)
**Concepts:** tensors, shapes, broadcasting, and matmul (this is the matrix refresh); autograd and the chain rule (the calculus refresh); `nn.Module`, optimizers and training loops; moving work to the GPU, `torch.cuda.synchronize()`, and how to time GPU code correctly; dtypes (fp32 / bf16 / fp16).
**Project:** (a) build micrograd, a tiny autograd engine, by hand. (b) Train an MLP on MNIST on CPU, then on GPU, and compare. (c) Benchmark a matmul across sizes and dtypes on your 5060 Ti and plot achieved TFLOPS against the card's spec.
**Resources:** Karpathy, [*The spelled-out intro to neural networks and backpropagation* (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0); [PyTorch "Learn the Basics"](https://pytorch.org/tutorials/beginner/basics/intro.html). Optional math refresh: 3Blue1Brown, *Essence of Linear Algebra* ch. 1–4.
**Checkpoint quiz:** work out output shapes by hand for 10 tensor operations; hand-derive the gradients of a small graph; explain why timing GPU code without synchronizing is wrong; explain what bf16 gives up compared with fp32.

## Stage 4: How transformers work (≈3 weeks, 60 h)
**Concepts:** tokenization (BPE); embeddings; self-attention (Q, K, V, causal mask, softmax, which is the probability refresh); multi-head attention; MLP blocks, residual connections, LayerNorm/RMSNorm; positional encodings (learned embeddings, then RoPE); sampling (greedy, temperature, top-k, top-p).
**Project:** build a tiny GPT from scratch (about 10M parameters), train it on TinyShakespeare on your GPU, and generate text. Then load GPT-2 small's weights into *your* architecture and check that your outputs match Hugging Face's.
**Resources:** Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY); 3Blue1Brown, *Neural Networks* ch. 5–6 (transformers and attention).
**Checkpoint quiz:** give the shape of every tensor in one attention layer; explain why the causal mask exists; count your model's parameters by hand; explain what temperature does to the distribution; find the bug in an attention implementation I hand you.

## Stage 5: Inference fundamentals (≈2 weeks, 40 h)
**Concepts:** prefill vs decode; the KV cache and its memory cost (`2 × layers × heads × head_dim × seq × bytes`); arithmetic intensity and the roofline model; why decode is memory-bandwidth-bound and prefill is compute-bound; static batching and padding waste; TTFT, TPOT/ITL and throughput; GPU basics (SMs, HBM vs SRAM, kernel launches, CUDA graphs).
**Project:** add a KV cache to your tiny GPT. **Predict** the speedup first, then measure it across sequence lengths. Add batched generation and plot tokens/s against batch size, and explain the curve with a roofline calculation using your 5060 Ti's memory bandwidth. Profile one decode step with `torch.profiler`.
**Resources:** kipply, [*Transformer Inference Arithmetic*](https://kipp.ly/transformer-inference-arithmetic/); Horace He, [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html).
**Checkpoint quiz:** compute the KV cache size for Llama-3-8B at 8k context; predict whether doubling the batch size roughly doubles decode throughput, and explain why; explain why TTFT and ITL respond differently to load.

## Stage 6: Serving concepts (≈4 weeks, 80 h)
Each concept gets a short explanation, a small experiment in your own code, and a mini-quiz.
| Concept | Experiment |
|---|---|
| Continuous batching | Turn your static-batch generator into an iteration-level scheduler, then compare throughput and latency under a Poisson arrival workload |
| Paged attention | Replace your contiguous KV cache with fixed-size blocks and a block table; measure fragmentation before and after |
| Prefix caching | Plug your Stage 2 trie into the KV cache and measure the TTFT win on a shared-system-prompt workload |
| Chunked prefill | Split long prefills into chunks and show the effect on decode latency for requests running at the same time |
| Quantization | Implement int8 weight-only quantization; measure memory, speed and quality (perplexity) |
| Speculative decoding | Use a small draft model with your GPT as the target; measure acceptance rate and speedup |
| Parallelism (concept only) | Tensor, pipeline, data and expert parallelism, explained on paper, since you have one GPU |
**Resources:** Anyscale, [*How continuous batching enables 23x throughput in LLM inference*](https://www.anyscale.com/blog/continuous-batching-llm-inference); the [SGLang paper (RadixAttention)](https://arxiv.org/abs/2312.07104). Skim the [vLLM / PagedAttention paper](https://arxiv.org/abs/2309.06180) as a companion.
**Checkpoint quiz:** a whiteboard-style design question ("design an LLM server for workload X: what do you turn on, and why?"), plus trade-off questions on each concept.

## Stage 7: Using SGLang (≈1.5 weeks, 30 h)
**Concepts:** installing SGLang (on Blackwell sm_120 we'll check what's needed on current `main` when we get here); `sglang.launch_server`; the OpenAI-compatible API; the key flags (`--mem-fraction-static`, `--max-running-requests`, `--chunked-prefill-size`, quantization); `bench_serving` and `bench_one_batch`; reading server logs and metrics.
**Project:** first read the model's page under `docs/cookbook/autoregressive/` (that is the deployment SGLang recommends to users). Then serve a small model (≈1–4B) and benchmark it at several request rates. Toggle radix cache, chunked prefill and quantization **one at a time**, predicting each result before you measure. Write up a one-page benchmark report, and compare against your Stage 6 toy server.
**Resources:** the SGLang docs ([docs.sglang.ai](https://docs.sglang.ai)) and `docs/cookbook/` in this repo.
**Checkpoint quiz:** I give you a benchmark output and you explain every number, then diagnose a "why is my throughput low?" scenario from its logs.

## Stage 8: Reading SGLang (≈4 weeks, 80 h)
**Concepts:** we explore the *current* code together rather than relying on memory. Trace one request end to end: HTTP entrypoint → TokenizerManager → Scheduler (over ZMQ) → batch formation → ModelRunner forward → sampling → DetokenizerManager → response streaming. Then do deep dives into the Scheduler (the event loop, batch policies, overlap scheduling) and the radix cache (matching, insertion, eviction, how it connects to the memory pool).
**Project:** (a) a hand-drawn request-flow diagram with file:line references. (b) Instrument a local build with your own logging to confirm the trace. (c) Write a stripped-down "mini-scheduler" that mirrors SGLang's design, using your Stage 6 code. (d) Read and summarize 3 recently merged PRs in the scheduler/cache area.
**Resources:** the code itself, plus the repo's `docs/` and `.claude/skills/` notes.
**Checkpoint quiz:** "Here's a symptom; which file and function do you open first?" (×10). Also: explain how the radix cache decides what to evict.

## Stage 9: First contribution (≈3–4 weeks, 60–80 h)
**Steps:** find `good first issue` / `help wanted` issues, or a small perf or bug item you notice yourself. Reproduce it, write the fix and a test, and run the repo's lint and tests. I review your change like a strict maintainer. We then write the PR with before/after benchmarks and handle the review rounds.
**Resources:** `CONTRIBUTING` / `docs/developer_guide/` in the repo and `test/README.md`.
**Checkpoint:** a merged PR, which is the real test.

---

## Timeline (at 20 h/week)
| Stage | Weeks | Cumulative |
|---|---|---|
| 2 Python depth | 1 | 1 |
| 3 PyTorch | 2 | 3 |
| 4 Transformers | 3 | 6 |
| 5 Inference fundamentals | 2 | 8 |
| 6 Serving concepts | 4 | 12 |
| 7 Using SGLang | 1.5 | 13.5 |
| 8 Reading SGLang | 4 | 17.5 |
| 9 First contribution | 3–4 | ~21 |

Expect some stages to take longer than planned. That's normal, and failing a checkpoint means we slow down, not that you skip ahead.

## Career track (runs alongside the stages)
- Each stage's project goes into your portfolio repo with a short README (what you built, what you measured, what surprised you).
- From Stage 5 on, write one short blog post per stage (for example "I added a KV cache to my GPT, and here's the roofline math"). Inference teams hire people who can **measure and explain**.
- Stage 7 onward is where inference-engineer interview material overlaps: latency/throughput trade-offs, GPU memory math, and serving-system design.
