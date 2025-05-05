# CS336 Spring 2025 Assignment 2 (Systems) Leaderboard

To submit to the leaderboard, submit a pull request that adds your results to the Markdown table below.
The table should be sorted by increasing time in milliseconds.

You should be running on a single H100 with batch size `1`, sequence length `16384`, `d_model=1024` and `num_heads=16`.

The top 3 submissions will receive a prize at the end of the quarter.
To make this fair, we will reorder the top 5 scoring students based on our own tests.
**Make sure you save a snapshot of your best code so it can be reproduced by us!**
We will reach out to the top 5 students after results have stabilized.

In your pull request description, you should also include:

- The time you recorded
- A description of what you did

## FlashAttention 2 Forward + Backward Leaderboard

| Name            | Forward + Backward Time (ms) | Verification status (leave empty) |
| :-------------- | ---------------------------: | --------------------------------: |
| Matthew Noto    |                     4.027 ms |                                   |
| Herman Brunborg |                      5.286ms |                                   | 
| Varun Desai     |                     12.40ms  |                                   | 
| Prateek Varshney|                     17.18ms  |                                   | 
| Stephen Ge      |                     28.45 ms |                                   |
| naive baseline  |                        80 ms |                          Verified |
