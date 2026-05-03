# CS336 Spring 2026 Assignment 2 (Systems) Leaderboard

> [!NOTE]
> If you're a non-Stanford student and interested in submitting to the leaderboard, please create a pull request adding your result to the **second** table. To remain in the top 5, your submission must be verified, for which you should invite `marcelroed` to a minimal repo containing a uv project with `pyproject.toml`, `uv.lock`, and `main.py`. Your script should be reproducible on two B200 GPUs by running `uv run main.py`.

To submit to the leaderboard, submit a pull request that adds your results to the Markdown table below.
The table should be sorted by increasing wall-clock time in milliseconds.

Assignment 2's leaderboard tests the speed of a full training step for an 8B model.
We challenge you to benchmark your code and optimize memory and runtime using any tricks you can come up with.
The key restriction is that you cannot change the input/output behavior of the model.
Your implementation will be tested against the model in the `cs336_basics` directory.
Your inputs will be tested at BF16 with causal masking, and they must pass the same tests as your regular implementation.
The implementation must be your own, and you cannot use or copy pre-existing implementations.

Your timing should be measured on two B200 GPUs using batch size `2` and sequence length `32768`.
It is intentionally difficult to fit the model in memory.
From an empty PyTorch/Triton cache, your benchmarking run must complete within 10 minutes, so be careful with overly aggressive `torch.compile` and Triton autotuning.

The top 3 submissions will receive a prize at the end of the quarter.
To make this fair, we will reorder the top 5-10 scoring submissions based on our reproduced timing runs.
**Make sure you save a snapshot of your best code so it can be reproduced by us!**
Leading submissions that cannot be verified or are invalid will be removed.

In your pull request description, you should include:

- The wall-clock time you recorded for one complete training step.
- A description of what you did.
- Any important compile, autotuning, cache, or hardware assumptions needed to reproduce your timing.

We expect leaderboard submissions to beat the naive baseline of 10 seconds.

## Full Training Step Leaderboard

Stanford class leaderboard (Spring 2026) - 2 B200 GPUs

| Name           | Training Step Time (ms) | Verification status (leave empty) |
| :------------- | ----------------------: | --------------------------------: |
| Keshav Patel Keval |             3837 ms |                                   |
| Max Liu        |                 3868 ms |                                   |
| Bodo Wirth     |                 4728 ms |                                   |
| Nick Rui       |                 4998 ms |                                   | 
| Aayush Gupta   |                 5174 ms |                                   |
| Tushar Aggarwal|                 5584 ms |                                   | 
| Asanshay Gupta |                 5745 ms |                                   |
| Weiran Xu      |                 6089 ms |                                   |
| Tim Chen       |                 6199 ms |                                   | 
| Aniket Gupta   |                 6237 ms |                                   | 
| Adam Alhousiki |                 6469 ms |                                   |
| Shiye Su       |                 7006 ms |                                   | 
| Thomas Li      |                 7190 ms |                                   |
| Sara Kothari   |                 7431 ms |                                   | 
| Tushar Dalmia  |                 8133 ms |                                   | 
| Eric Chen      |                 8158 ms |                                   |
| Jiaming Shen   |                 8453 ms |                                   |
| Qinan Yu       |                 8503 ms |                                   |
| Tanush Talati  |                 8794 ms |                                   | 
| Javier Nieto   |                 8829 ms |                                   | 
| Silin Du       |                 9083 ms |                                   |
| Yufei Liu      |                 9277 ms |                                   | 
| Jenn Wang      |                 9542 ms |                                   |
| Jason Meng     |                 9555 ms |                                   |
| naive baseline |                10000 ms |                          Verified |

<details markdown="1">
<summary>Global leaderboard (Spring 2026) - 2 B200 GPUs</summary>

| Name           | Training Step Time (ms) | Verification status (leave empty) |
| :------------- | ----------------------: | --------------------------------: |
| naive baseline |                10000 ms |                          Verified |

</details>

<details markdown="1">
<summary>Stanford class leaderboard (Spring 2025) - Forward/Backward flash attention on 1 H100</summary>

| Name             | Forward + Backward Time (ms) | Verification status (leave empty) |
| :--------------- | ---------------------------: | --------------------------------: |
| Herman Brunborg  |                      5.364ms |                          Verified |
| Matthew Noto     |                      6.778ms |                          Verified |
| Varun Desai      |                      11.07ms |                          Verified |
| Prateek Varshney |                      22.59ms |                          Verified |
| Stephen Ge       |                      28.45ms |                                   |
| naive baseline   |                        80 ms |                          Verified |

</details>

## Timing Code

We will time your implementation with a test like the following:

```python
class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2


cfg = Config()


def test_timing_forward_backward():
    labels, targets = torch.randint(high=cfg.vocab_size, size=(2, cfg.batch_size, cfg.ctx_len))

    model = BasicsTransformerLM(Config())
    optimizer = AdamW(model.parameters())

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = model(labels)
        loss = cross_entropy(res, targets).sum()
        loss.backward()
        optimizer.step()

    timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
    print(timing_results)
```

For testing purposes, you can reduce the repetition and warmup time, given in milliseconds, to something shorter.

## Optimization Ideas

- Tune the tile sizes for your kernels. Triton autotuning can help, but keep the total benchmark runtime under the 10 minute limit.
- Tune additional Triton and `torch.compile` config parameters.
- Implement fused AdamW.
- The base implementation materializes the full logits tensor, shaped `[batch, seq_len, vocab_size]`. Consider writing a kernel that fuses the LM head and cross-entropy loss. You can also compute the backward pass immediately in a fused manner.
- Improve FlashAttention.
- Implement the FlashAttention backward pass in Triton instead of only relying on `torch.compile`.
- Do two passes over your input for the backward pass, one for `dQ` and another for `dK` and `dV`, to avoid atomics or synchronization between blocks.
- Stop program instances early when doing causal masking, skipping tiles that are guaranteed to be all zero.
- Separate non-masked tiles from tile diagonals, computing the first without index comparisons and the second with a single comparison.
- Use TMA (Tensor Memory Accelerator) functionality on architectures later than Hopper, following a similar pattern to our tutorial.
- Use activation checkpointing to trade runtime speed for memory savings only if you need it.
