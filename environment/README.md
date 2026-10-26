# Analysis environment

`environment.yml` is the portable full-analysis environment specification for this repository. The environment name is `rl-behavioral-history-first-passage`.

```bash
conda env create -f environment/environment.yml
conda activate rl-behavioral-history-first-passage
```

`conda-linux-64-explicit.txt` records the frozen Linux package set from the release environment and is useful when exact platform-compatible package reconstruction is required.

For public figure reproduction only, `requirements_reporting.txt` is sufficient.
