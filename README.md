# ML Training Infrastructure

A from-first-principles study of ML pipeline orchestration on Kubernetes: from container primitives to a working Argo Workflows pipeline, with empirical benchmarks on data movement patterns at scale.

**[→ Read the full documentation site](https://tejask-42.github.io/ml-training-infrastructure)**

---

## What's here

- **[Foundations](docs/foundations/)**: Docker and Kubernetes mechanics explained from first principles: why distributed compute is needed, how containers work, how the Kubernetes scheduler places workloads
- **[Argo Workflows](docs/argo-workflows/)**: What raw Kubernetes Jobs cannot do, why Argo is needed, and deep internal mechanics (the 3-container Pod pattern, artifact propagation, memoization, and the distributed training boundary with Kubeflow)
- **[Experiments](docs/experiments/)**: Empirical benchmarks on distributed data movement: inter-step data passing architectures (Shared PV vs. Argo Artifacts vs. Direct S3A Streaming), Parquet byte-range mechanics, iterative HPO tradeoffs, live crash failure recovery, and two-phase key-salting for extreme partition skew
- **[`pipeline/`](pipeline/)**: Runnable `WorkflowTemplate` covering the full training lifecycle with setup instructions for a local k3s cluster

## Stack

- **Orchestration:** Argo Workflows on k3s
- **Artifact store:** MinIO (S3-compatible)
- **Data processing:** Apache Spark
- **Compute:** Single-node k3s cluster (expandable)
