# Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2502.00234"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2502.00234-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2502.00234](https://arxiv.org/abs/2502.00234) |

## Summary

This reproduction implements the core high-order inference algorithms from "Fast Solvers for Discrete Diffusion Models" and evaluates them on the paper's analytic 15-state toy model. The experiment tests whether `theta`-Trapezoidal and `theta`-RK-2 achieve faster KL convergence than first-order tau-leaping under the same discrete diffusion reverse-process setting.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +114.7 |
| `gpt-5.6-luna-xhigh` | +93.3 |
| `gpt-5.6-sol-xhigh` | +86.8 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2502.00234 --mode innovate --agent claude-code
sw evaluate 2502.00234
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

