# Linear Bandits Lab

**A unified platform to implement, benchmark, and reproduce research on linear
and generalized linear bandits.**

- 23 algorithms and baselines covering stationary, non-stationary, and heavy-tailed settings.
- Includes our implementations for [ICML 2025 Hvt-UCB](https://www.lamda.nju.edu.cn/wangjing/publication/ICML%2725_HvtLB.pdf) and [TIT 2026 WeightUCB](https://www.lamda.nju.edu.cn/wangjing/publication/TIT%2726_weightNS.pdf).

## 1. Install

```bash
bash setup.sh
source .venv/bin/activate
```

Installs Python 3.12.14 and pinned dependencies. Linux, macOS, or WSL; CPU only.
Requires internet access and `curl` or `wget`.

## 2. What's included

| Setting | Algorithms |
| --- | --- |
| Linear | [NIPS'11 OFUL](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html) |
| Non-stationary linear | [AISTATS'20 RestartUCB](https://proceedings.mlr.press/v108/zhao20a.html) \| [AISTATS'19 SlidingWindowUCB](https://proceedings.mlr.press/v89/cheung19b.html) \| [NeurIPS'19 D-LinUCB](https://papers.neurips.cc/paper_files/paper/2019/hash/263fc48aae39f219b4c71d9d4bb4aed2-Abstract.html) \| [TIT'26 WeightUCB](https://www.lamda.nju.edu.cn/wangjing/publication/TIT%2726_weightNS.pdf) \| [COLT'21 MASTER](https://proceedings.mlr.press/v134/wei21b.html) |
| Non-stationary GLB | [NIPS'10 GLM-UCB](https://proceedings.neurips.cc/paper/2010/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html) \| [AISTATS'20 GLB-RestartUCB](https://proceedings.mlr.press/v108/zhao20a.html) \| [TIT'26 GLB-WeightUCB](https://www.lamda.nju.edu.cn/wangjing/publication/TIT%2726_weightNS.pdf) \| [arXiv'21 BVD-GLM-UCB](https://arxiv.org/abs/2103.05750) \| [COLT'21 MASTER-GLB](https://proceedings.mlr.press/v134/wei21b.html) |
| Non-stationary SCB | [ICML'20 LogUCB1](https://proceedings.mlr.press/v119/faury20a.html) \| [AISTATS'20 SCB-RestartUCB](https://proceedings.mlr.press/v108/zhao20a.html) \| [TIT'26 SCB-WeightUCB](https://www.lamda.nju.edu.cn/wangjing/publication/TIT%2726_weightNS.pdf) \| [COLT'21 MASTER-SCB](https://proceedings.mlr.press/v134/wei21b.html) |
| Heavy-tailed linear / GLB | [ICML'25 Hvt-UCB](https://proceedings.mlr.press/v267/wang25et.html) \| [NeurIPS'23 HeavyOFUL](https://papers.nips.cc/paper_files/paper/2023/hash/b11393733b1ea5890100302ab8a0f74c-Abstract-Conference.html) \| [NeurIPS'23 CRTM](https://papers.nips.cc/paper_files/paper/2023/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html) \| [NeurIPS'23 CRMM](https://papers.nips.cc/paper_files/paper/2023/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html) |
| Binary / multinomial logistic | [ICML'16 OL2M](https://proceedings.mlr.press/v48/zhangb16.html) \| [NIPS'10 GLM-UCB](https://proceedings.neurips.cc/paper/2010/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html) \| [NeurIPS'21 MNL-UCB](https://papers.nips.cc/paper_files/paper/2021/hash/16f852a6d01b6065c8ff5cc11caae9c6-Abstract.html) \| [NeurIPS'23 MLogB variant](https://papers.nips.cc/paper_files/paper/2023/hash/5ef04392708bb2340cb9b7da41225660-Abstract-Conference.html) |

Environments support fixed or changing actions, stationary or rotating reward
parameters, and linear, sigmoid, or softmax rewards. Feedback can be additive,
Bernoulli, or categorical; noise options include Gaussian, Student-t, Pareto,
Lomax, Fisk, and zero noise.

- [`algorithms/`](algorithms/): implementations and their source papers.
- [`bandit_environments/`](bandit_environments/): actions, parameters, reward models, and noise.
- [`experiments/`](experiments/): shared runner, plotting, and [configs](experiments/configs/).
- [`utils/`](utils/): numerical solvers, random seeds, and result-file helpers.

## 3. Reproduce NSLB and HvtLB results

```bash
python exp_nonstationary_LB.py
python exp_heavy-tailed_LB.py --workers 4
```

Both scripts use the same runner with different configs and generate PDF figures:

| Experiments | Rounds × seeds per method | Results |
| --- | --- | --- |
| [NSLB / TIT](experiments/configs/nonstationary_LB.json): 5 linear, GLB, and SCB settings | 6,000 × 20 | [Regret](results/nonstationary_LB/regret.pdf) · [Runtime](results/nonstationary_LB/runtime.pdf) |
| [HvtLB / ICML](experiments/configs/heavy_tailed_LB.json): 2 main-text + 9 appendix settings | Main: 18,000 × 10; appendix: 6,000 × 5 | [Regret](results/heavy_tailed_LB/regret.pdf) · [Runtime](results/heavy_tailed_LB/runtime.pdf) |

Results are saved in `results/nonstationary_LB/` and `results/heavy_tailed_LB/`,
each containing `regret.pdf`, `runtime.pdf`, and `curves.npz`.
Add `--plot` to redraw saved results, or `--resume` to continue an interrupted run.
Runtime comparisons run serially; `--workers` applies to the ICML appendix.
The bundled results are seeded reruns; some curves and rankings differ from the papers.

## 4. Build your own experiment

**Use a new setting.** Copy an example, edit `environment`, `algorithms`,
`horizon`, and `seeds`, then run it:

```bash
cp experiments/configs/examples/standard-linear-fixed.json my-experiment.json
python experiments/run.py --config my-experiment.json --workers 4
```

Results go to `results/my-experiment/`. Use a [collection config](experiments/configs/nonstationary_LB.json)
to run multiple settings; set `"runtime": true` on an entry to include runtime plots.

**Add an algorithm.**

1. Implement [`Policy`](algorithms/base.py): `scores`, `update`, and `reset`,
   with `REQUIREMENTS` and `DEFAULTS`.
2. Register it in `ALGORITHMS` and `REFERENCES` in [`algorithms/__init__.py`](algorithms/__init__.py).
3. Select it in your config and run the same command above.

## Credits and license

The original experimental code was written by human researchers in the pre-agent
era as part of Jing Wang's prior research. We recently used Codex to refine this
code and integrate multiple bandit algorithms and settings into a unified framework.
Released under the [MIT License](LICENSE).

## Citation

If this framework is useful for your research, please consider citing:

```bibtex
@inproceedings{ICML'25:hvtlb,
  author    = {Jing Wang and
               Yu-Jie Zhang and
               Peng Zhao and
               Zhi-Hua Zhou},
  title     = {Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  pages     = {65595--65617},
  year      = {2025}
}

@article{TIT'26:weightUCB,
  author       = {Jing Wang and
                  Peng Zhao and
                  Zhi{-}Hua Zhou},
  title        = {Revisiting Weighted Strategy for Non-Stationary Parametric Bandits
                  and MDPs},
  journal      = {IEEE Transactions on Information Theory},
  volume       = {72},
  number       = {3},
  pages        = {1798--1816},
  year         = {2026}
}
```
