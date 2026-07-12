# Windows local-data workflow

The project can discover datasets below an existing project tree, convert them to one standard NPZ schema, deduplicate copies, audit them, and launch the benchmark.

## One command

```powershell
powershell -ExecutionPolicy Bypass -File .\run_full_benchmark.ps1 `
  -SearchRoot "C:\Users\24953\PycharmProjects" `
  -Stage screening
```

Add `-WithOptionalGraphPackages` when PyG `processed/data.pt` or OGB caches must be deserialized.

## Supported local formats

- GraphAtlas or generic NPZ files;
- PyTorch Geometric `processed/data.pt`;
- Planetoid raw files such as `ind.cora.x` and `ind.citeseer.graph`;
- Geom-GCN text files;
- OGB local caches;
- PyG `HeteroData` objects, projected to a typed homogeneous graph while preserving `node_type` and `edge_type`.

## Duplicate handling

`discover_local_data.py` records source paths, formats, byte sizes, and source fingerprints. Multiple Cora copies remain visible in the discovery manifest. `import_local_data.py --prefer first` selects one copy, while `--prefer largest` or `--prefer smallest` changes the policy. The audit report adds feature, label, and order-independent edge-set hashes so equivalent copies can be identified.

## Manual steps

```powershell
python scripts/discover_local_data.py `
  --search-root "C:\Users\24953\PycharmProjects" `
  --output outputs/manifests/local_candidates.json

python scripts/import_local_data.py `
  --manifest outputs/manifests/local_candidates.json `
  --destination data

python scripts/audit_datasets.py `
  --dataset cora --dataset citeseer --dataset pubmed `
  --output-dir outputs/reports/data_audit
```

The standardized location is:

```text
data/<dataset>/raw/<dataset>.npz
```

## GraphAtlas-only full benchmark

The default workflow no longer retrains external papers' models. It runs GraphAtlas and its ablations, then imports curated published aggregates.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_full_benchmark.ps1 `
  -SearchRoot "C:\Users\24953\PycharmProjects" `
  -Stage all `
  -IncludeTextHeterophily `
  -IncludeH2GB `
  -IncludeLargeNonHomophily
```

Optional groups are skipped when no standardized local NPZ file is available. CITE is deliberately separate because its heterogeneous text pipeline is expensive:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_full_benchmark.ps1 `
  -SearchRoot "C:\Users\24953\PycharmProjects" `
  -IncludeFrontierText
```
