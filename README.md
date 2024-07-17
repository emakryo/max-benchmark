# max-benchmark

Benchmark script of [MAX engine](https://www.modular.com/max)

## Setup

### Install rye

See https://rye.astral.sh/guide/installation/

### Install MAX engine

See https://docs.modular.com/max/install

### Configure pyproject.toml

Replace MAX wheel path with your own in pyproject.toml.

```toml
[[tool.rye.sources]]
name = "modular"
# Replace MAX wheel path with your own.
# Run `modular config max.path` to get the path.
# Typically, it is `~/.modular/pkg/packages.modular.com_max/wheels`.
url = "file:///<HOME>/.modular/pkg/packages.modular.com_max/wheels"
type = "find-links"
```

### Install dependencies

Run the following commands to install dependencies.

```
rye sync
```

### Run benchmark

Running the following command produces the benchmark results in `benchmark.csv`.

```
rye run benchmark
```