# Running the Pipeline

This directory contains the `WorkflowTemplate` defining all nine pipeline entrypoints. To run it, you need a Kubernetes cluster with Argo Workflows and MinIO installed.

---

## Prerequisites

### 1. A Kubernetes cluster

The pipeline was built and tested on **k3s**, a lightweight Kubernetes distribution that installs as a single binary. It's the easiest way to get a real Kubernetes cluster running on one machine or a small server.

```bash
# Install k3s (single-node cluster)
curl -sfL https://get.k3s.io | sh -

# Verify it's running
kubectl get nodes
```

### 2. Argo Workflows + MinIO

The quickest way to get both is Argo's Postgres-backed quick start manifest, which bundles Argo Workflows, MinIO (artifact store), and Postgres (workflow archive) in one install.

```bash
# Create the argo namespace
kubectl create namespace argo

# Install Argo Workflows (quick start with MinIO + Postgres)
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-postgres.yaml

# Wait for all pods to be ready (takes ~2 minutes)
kubectl wait --for=condition=ready pod --all -n argo --timeout=300s
```

### 3. Argo CLI

```bash
# macOS
brew install argo

# Linux
curl -sLO https://github.com/argoproj/argo-workflows/releases/latest/download/argo-linux-amd64.gz
gunzip argo-linux-amd64.gz && chmod +x argo-linux-amd64
sudo mv argo-linux-amd64 /usr/local/bin/argo
```

---

## Setup

### Step 1: Update the data volume path

The `main`, `hp-tuning`, `hp-tuning-gated`, and `conditional-gate` entrypoints mount a `training-data` host directory. Edit `ml-training-pipeline.yaml` and update this path to where your dataset lives on the cluster node:

```yaml
volumes:
  - name: training-data
    hostPath:
      path: /data/training-datasets   # ← change this to your actual path
      type: Directory
```

The dataset should contain the files your `prep.py` script expects. For the demo scripts, a standard tabular CSV dataset (e.g. any Kaggle classification dataset) works.

### Step 2: Create the scripts ConfigMap

All step scripts live in the `scripts/` directory and are mounted into pods via a ConfigMap. Apply it before submitting any workflow:

```bash
kubectl create configmap pipeline-scripts \
  --from-file=scripts/ \
  -n argo
```

To update scripts after the ConfigMap is created:

```bash
kubectl delete configmap pipeline-scripts -n argo
kubectl create configmap pipeline-scripts --from-file=scripts/ -n argo
```

### Step 3: Apply the WorkflowTemplate

```bash
kubectl apply -f ml-training-pipeline.yaml -n argo
```

Verify it was accepted:

```bash
kubectl get workflowtemplates -n argo
# NAME                   AGE
# ml-training-pipeline   5s
```

---

## Submitting a workflow

Each entrypoint is submitted by name. Replace `<entrypoint>` with any of the names in the table below.

```bash
argo submit -n argo --from workflowtemplate/ml-training-pipeline \
  --entrypoint <entrypoint> \
  --watch
```

### Entrypoints

| `--entrypoint` | What runs |
| :--- | :--- |
| `main` | Full pipeline: prep → train → eval → human approval → promote |
| `hp-tuning` | Fan-out over N configs, pick single best |
| `hp-tuning-gated` | Per-trial F1 gate, promote all that pass |
| `conditional-gate` | Data quality check gates whether pipeline continues |
| `sync-hold` | Mutex demo: submit twice to see priority queuing |
| `data-passing-demo` | All five data-movement mechanisms side by side |
| `join-demo` | Explicit S3 key control + non-uniform artifact selection |
| `pv-vs-object-storage` | PVC vs MinIO artifact passing benchmark across 4 hops |
| `parallel-selective-process` | Templated ConfigMap key in `withParam` fan-out |

### Watching a running workflow

```bash
# Live status in terminal
argo watch <workflow-name> -n argo

# Inspect per-step timing and artifacts
argo get <workflow-name> -n argo

# Stream logs from a specific step
argo logs <workflow-name> -n argo --node-field-selector displayName=train
```

### Accessing the Argo UI

```bash
kubectl -n argo port-forward svc/argo-server 2746:2746
```

Then open `https://localhost:2746` in your browser. The UI shows the live DAG graph, per-step logs, and artifact download links.

---

## The human approval gate

The `main` entrypoint pauses at the `approve` step until you explicitly resume it. After evaluating the metrics in the UI:

```bash
# List running workflows to get the name
argo list -n argo

# Resume the suspended workflow
argo resume <workflow-name> -n argo
```

---

## Additional setup for specific entrypoints

### `join-demo`

Requires a `join-demo-data` directory with three subdirectories (`folder1/`, `folder2/`, `folder3/`) each containing four CSV files (`file1.csv` through `file4.csv`). Update the `join-demo-data` hostPath in the YAML, then create the join manifest ConfigMap:

```bash
kubectl create configmap join-manifest \
  --from-literal=manifest='["cleaned/folder1/file1.csv","cleaned/folder1/file3.csv","cleaned/folder2/file2.csv"]' \
  -n argo
```

### `parallel-selective-process`

Requires a `process-manifests` ConfigMap with keys `1`, `2`, and `3`:

```bash
kubectl create configmap process-manifests \
  --from-literal=1='["file_a.parquet","file_b.parquet"]' \
  --from-literal=2='["file_c.parquet"]' \
  --from-literal=3='["file_d.parquet","file_e.parquet","file_f.parquet"]' \
  -n argo
```

### `sync-hold`: cross-workflow priority demo

Submit two workflows simultaneously with different priorities:

```bash
# High priority: should acquire the mutex first
argo submit -n argo --from workflowtemplate/ml-training-pipeline \
  --entrypoint sync-hold \
  --parameter label=HIGH \
  -l workflows.argoproj.io/priority=10

# Low priority: will wait in queue
argo submit -n argo --from workflowtemplate/ml-training-pipeline \
  --entrypoint sync-hold \
  --parameter label=LOW \
  -l workflows.argoproj.io/priority=1
```

Watch both in the Argo UI to see the priority queue in action.

---

## Troubleshooting

**Pod stuck in `Pending`**
```bash
kubectl describe pod <pod-name> -n argo
# Look for "Insufficient memory" or "Insufficient cpu" in Events
```
Reduce the resource `requests` in the YAML to match your cluster's available capacity.

**Artifact upload/download fails**
```bash
# Check the argoexec sidecar logs specifically
argo logs <workflow-name> -n argo --container wait
```
Common cause: MinIO credentials not found. Verify the `my-minio-cred` secret exists:
```bash
kubectl get secret my-minio-cred -n argo
```

**ConfigMap not found**
Ensure `pipeline-scripts` ConfigMap was created in the `argo` namespace (not `default`). Always pass `-n argo` to `kubectl create configmap`.
