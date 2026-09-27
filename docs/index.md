# ML Training Infrastructure

An ML pipeline is a sequence of steps: clean and prepare data, train a model on it, evaluate how well it performs, and deploy it. On a laptop with a small dataset, this is a few Python scripts. At production scale (terabytes of data, large models, many concurrent jobs), the same scripts crash, and you need a fundamentally different approach.

This site documents that approach from first principles: why single machines run out of memory, how containers and Kubernetes manage a fleet of workers, and how Argo Workflows orchestrates a multi-step training pipeline reliably on top of all of it.

A working, runnable pipeline implementing all the patterns discussed here is in [`pipeline/`](https://github.com/tejask-42/ml-training-infrastructure/tree/main/pipeline).

---

## What's covered

| Section | What you'll understand |
| :--- | :--- |
| [Why Distributed Computing](foundations/why-distributed.md) | What breaks when your dataset doesn't fit in RAM, and the architectural shifts that fix it |
| [Docker](foundations/docker.md) | How containers guarantee identical code runs on every machine in a cluster |
| [Kubernetes](foundations/kubernetes.md) | How Kubernetes schedules, restarts, and manages containers across many machines |
| [Why Argo Workflows](argo-workflows/why-argo.md) | Why Kubernetes alone isn't enough for a multi-step ML pipeline, and what Argo adds |
| [Argo Internals](argo-workflows/internals.md) | The underlying architecture: controller reconcile loop, pod execution lifecycle, artifact handoff, and distributed training boundaries |
| [Data Movement at Scale](experiments/data-movement.md) | Empirical benchmarks on inter-step data passing (Shared PV vs. Argo Artifacts vs. Direct S3A), Parquet byte-range mechanics, and key-salting for partition skew |
