# Third-party code

Pinned submodules. None of them is needed to plan with this repository or to rebuild any table
in the paper; they are here so the baselines and the upstream world models can be checked out at
the commit this work was built and compared against.

| Path | Used for | Upstream |
| :--- | :--- | :--- |
| `LeFlow` | the Cube protocol and its reference success test, which `writeup/make_cube_table.py` matches exactly | https://github.com/hsiangwei0903/LeFlow |
| `GCIDM` | baseline, amortized planning in world models | https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models |
| `dino_wm` | the world model behind Point maze, Wall and Push-T; this repository is a derivative of it | https://github.com/gaoyuezhou/dino_wm |
| `le-wm-grasp` | the LeWM/JEPA world model behind Reacher and Cube (its checkpoints are bundled, so nothing here is needed to run them) | https://github.com/michael-psenka/le-wm-grasp |
| `grasp` | the GRASP baseline of `planning/grasp.py` | https://github.com/michael-psenka/grasp |
| `blackjax` | the MCMC kernels of `planning/mcmc_kernels.py` are ported from it, and `planning/test_mcmc_kernels.py` checks ours against it | https://github.com/blackjax-devs/blackjax |

```bash
bash scripts/baselines.sh              # all of them
bash scripts/baselines.sh LeFlow       # one, by directory name
git submodule update --init third_party/blackjax   # the same thing, by hand
```

`LeFlow` and `GCIDM` are pinned at the commits the paper's runs used. The other four are pinned at
their upstream head at the time of release; this repository carries its own implementation of each,
so the pin is a reference point, not a dependency.

Outputs that these produce go in `third_party/work/`, which is not tracked.
