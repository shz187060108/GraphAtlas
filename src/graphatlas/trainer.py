from __future__ import annotations

import copy
import os
import random
from dataclasses import replace
import time
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from graphatlas.config import ExperimentConfig
from graphatlas.data import GraphData
from graphatlas.losses import atlas_regularization_terms, compute_loss
from graphatlas.metrics import accuracy, evaluate_output, output_primary_metric
from graphatlas.nn.model import build_model, wrap_link_prediction_model
from graphatlas.utils import configure_torch_threads, count_parameters, resolve_device, save_json, save_yaml, seed_everything




def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _quarantine_corrupt(path: Path) -> None:
    if not path.exists():
        return
    destination = path.with_name(path.name + ".corrupt")
    counter = 1
    while destination.exists():
        destination = path.with_name(path.name + f".corrupt.{counter}")
        counter += 1
    os.replace(path, destination)

def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints produced by some PyTorch/platform combinations deserialize
    # the byte state as a generic tensor. Normalize it before restoration so
    # resume remains portable across CPU/CUDA and PyTorch minor versions.
    torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu")
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [torch.as_tensor(value, dtype=torch.uint8, device="cpu") for value in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def _loss_config_for_epoch(config: ExperimentConfig, epoch: int):
    if epoch % config.train.regularization_every == 0:
        return config.loss
    return replace(
        config.loss,
        sample_nodes=min(config.loss.sample_nodes, config.train.light_regularization_nodes),
        metric_probes=min(config.loss.metric_probes, config.train.light_metric_probes),
        geometry=0.0,
    )


class Trainer:
    def __init__(self, config: ExperimentConfig):
        config.validate()
        self.config = config
        configure_torch_threads(config.train.num_threads)
        seed_everything(config.train.seed)
        self.device = resolve_device(config.train.device)

    def _uses_neighbor_sampling(self, data: GraphData) -> bool:
        name = str((data.metadata or {}).get("name", self.config.dataset.name)).lower()
        return self.config.train.neighbor_sampling and name in {
            "ogbn_arxiv", "ogbn_products", "ogbn_proteins"
        }

    def _fit_neighbor_sampled(
        self,
        data: GraphData,
        run_dir: Path,
        environment: dict[str, Any],
        evaluate_test: bool = True,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        """NeighborLoader training for OGB graphs without putting the full graph on CUDA."""
        try:
            from torch_geometric.data import Data
            from torch_geometric.loader import NeighborLoader
        except ImportError as error:
            raise RuntimeError("OGB neighbor sampling requires `pip install -e .[graph,ogb]`.") from error

        pyg_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
        model = build_model(data.num_features, data.num_classes, self.config.model, data.num_nodes).to(self.device)
        if self.config.dataset.task == "link_prediction":
            model = wrap_link_prediction_model(model, data, self.config.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.train.learning_rate,
                                      weight_decay=self.config.train.weight_decay)
        target_nodes = torch.arange(data.num_nodes)

        def loader(nodes: torch.Tensor, shuffle: bool):
            return NeighborLoader(
                pyg_data,
                input_nodes=nodes,
                num_neighbors=self.config.train.num_neighbors,
                batch_size=self.config.train.batch_size,
                shuffle=shuffle,
                num_workers=self.config.train.num_workers,
            )

        def batch_graph(batch: Any) -> tuple[GraphData, int, torch.Tensor]:
            batch = batch.to(self.device)
            seeds = int(batch.batch_size)
            n = int(batch.num_nodes)
            empty = torch.zeros(n, device=self.device, dtype=torch.bool)
            # GraphAtlas itself does not consume split masks in forward.  Keep
            # a valid local GraphData carrier without copying full-graph masks.
            local = GraphData(batch.x, batch.edge_index, batch.y, empty, empty, empty,
                              metadata={"name": self.config.dataset.name, "task": "node_classification"})
            return local, seeds, batch.n_id[:seeds]

        def task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            if labels.ndim == 1:
                return F.cross_entropy(logits, labels)
            finite = torch.isfinite(labels)
            if not bool(finite.any()):
                raise RuntimeError("Neighbor batch has no finite multilabel targets")
            return F.binary_cross_entropy_with_logits(logits[finite], labels.to(logits.dtype)[finite])

        @torch.no_grad()
        def infer() -> torch.Tensor:
            model.eval()
            logits = torch.empty((data.num_nodes, data.num_classes), dtype=torch.float32)
            for batch in loader(target_nodes, shuffle=False):
                local, seeds, node_ids = batch_graph(batch)
                output = model(local, compact=True)
                logits[node_ids.detach().cpu()] = output["logits"][:seeds].detach().float().cpu()
            return logits

        save_json({**environment, "neighbor_sampling": True}, run_dir / "environment.json")
        best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
        best_val, best_epoch, stale, start_epoch = -float("inf"), -1, 0, 1
        history: list[dict[str, float]] = []
        prior_runtime_seconds = 0.0
        if self.config.train.resume and last_path.exists():
            checkpoint = torch.load(last_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_val, best_epoch, stale = float(checkpoint["best_val"]), int(checkpoint["best_epoch"]), int(checkpoint["stale"])
            history, start_epoch = list(checkpoint.get("history", [])), int(checkpoint["epoch"]) + 1
            prior_runtime_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
            if "rng_state" in checkpoint:
                _restore_rng_state(checkpoint["rng_state"])

        started = time.time()
        iterator = trange(start_epoch, self.config.train.epochs + 1,
                          disable=not self.config.train.progress,
                          desc=f"{self.config.dataset.name}/sampled/s{self.config.train.seed}", leave=False)
        final_epoch = start_epoch - 1
        for epoch in iterator:
            final_epoch = epoch
            model.train()
            total_loss = 0.0
            batches = 0
            for batch in loader(torch.nonzero(data.train_mask, as_tuple=False).flatten(), shuffle=True):
                local, seeds, _ = batch_graph(batch)
                optimizer.zero_grad(set_to_none=True)
                loss = task_loss(model(local, compact=True)["logits"][:seeds], local.y[:seeds])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite sampled loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.grad_clip)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            if epoch == 1 or epoch % self.config.train.eval_every == 0 or epoch == self.config.train.epochs:
                logits = infer()
                eval_output = {"logits": logits, "embedding": logits}
                val_score = output_primary_metric(eval_output, data, "val", model)
                train_score = output_primary_metric(eval_output, data, "train", model)
                history.append({"epoch": epoch, "loss": total_loss / max(batches, 1),
                                "val_metric": val_score, "train_metric": train_score})
                if val_score > best_val + 1e-8:
                    best_val, best_epoch, stale = val_score, epoch, 0
                    _atomic_torch_save(model.state_dict(), best_path)
                else:
                    stale += self.config.train.eval_every
                pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
                _atomic_torch_save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "best_val": best_val, "best_epoch": best_epoch, "stale": stale,
                                    "history": history, "rng_state": _rng_state(),
                                    "elapsed_seconds": prior_runtime_seconds + time.time() - started}, last_path)
                if stale >= self.config.train.patience:
                    break

        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        logits = infer()
        metrics = evaluate_output(
            torch.nn.Identity(),
            {"logits": logits, "embedding": logits},
            data,
            self.config.train.seed,
            include_intervention=False,
            include_test=evaluate_test,
        )
        metrics.update({"dataset": self.config.dataset.name, "task": self.config.dataset.task,
                        "model": self.config.model.label or self.config.model.name, "model_family": self.config.model.name,
                        "seed": self.config.train.seed, "best_epoch": best_epoch, "last_epoch": final_epoch,
                        "best_val_metric": best_val, "parameters": count_parameters(model), "num_nodes": data.num_nodes,
                        "num_edges": int(data.edge_index.shape[1]), "runtime_seconds": prior_runtime_seconds + time.time() - started,
                        "device": str(self.device), "split": int((data.metadata or {}).get("split", 0)),
                        "neighbor_sampling": True, "batch_size": self.config.train.batch_size,
                        "num_neighbors": list(self.config.train.num_neighbors)})
        _atomic_torch_save({"logits": logits, "labels": data.y, "train_mask": data.train_mask,
                            "val_mask": data.val_mask, "test_mask": data.test_mask}, run_dir / "predictions.pt")
        save_json(metrics, run_dir / "metrics.json")
        return model, metrics

    def fit(
        self,
        data: GraphData,
        run_dir: str | Path,
        on_evaluation: Callable[[int, float], None] | None = None,
        evaluate_test: bool = True,
        include_intervention: bool = True,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(self.config.to_dict(), run_dir / "config.yaml")
        data.validate()
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(self.device),
            "dataset_metadata": data.metadata or {},
        }
        save_json(environment, run_dir / "environment.json")
        if self._uses_neighbor_sampling(data):
            return self._fit_neighbor_sampled(data, run_dir, environment, evaluate_test=evaluate_test)
        data = data.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        model = build_model(data.num_features, data.num_classes, self.config.model, data.num_nodes).to(self.device)
        if self.config.dataset.task == "link_prediction":
            model = wrap_link_prediction_model(model, data, self.config.model)
        tensor_devices = {
            str(value.device)
            for value in data.__dict__.values()
            if isinstance(value, torch.Tensor)
        }
        parameter_devices = {str(parameter.device) for parameter in model.parameters()}
        expected = self.device
        if expected.type == "cuda" and expected.index is None:
            expected = torch.device("cuda", torch.cuda.current_device())
        expected_device = str(expected)
        if tensor_devices != {expected_device} or parameter_devices != {expected_device}:
            raise RuntimeError(
                "Model/data device mismatch: "
                f"expected={expected_device}, tensors={sorted(tensor_devices)}, "
                f"parameters={sorted(parameter_devices)}"
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
        best_state = copy.deepcopy(model.state_dict())
        best_val = -float("inf")
        best_epoch = -1
        stale = 0
        history: list[dict[str, float]] = []
        start_epoch = 1
        prior_runtime_seconds = 0.0
        last_path = run_dir / "last.pt"
        best_path = run_dir / "best.pt"
        history_path = run_dir / "history.csv"

        if self.config.train.resume and last_path.exists():
            try:
                checkpoint = torch.load(last_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                best_val = float(checkpoint["best_val"])
                best_epoch = int(checkpoint["best_epoch"])
                stale = int(checkpoint["stale"])
                history = list(checkpoint.get("history", []))
                start_epoch = int(checkpoint["epoch"]) + 1
                prior_runtime_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
                if best_path.exists():
                    best_state = torch.load(best_path, map_location=self.device, weights_only=True)
                else:
                    best_state = copy.deepcopy(model.state_dict())
                if "rng_state" in checkpoint:
                    _restore_rng_state(checkpoint["rng_state"])
            except (EOFError, RuntimeError, ValueError, KeyError, OSError):
                _quarantine_corrupt(last_path)
                # A checkpoint interrupted before atomic-save support cannot be
                # trusted. Restart this single run from its deterministic seed.
                seed_everything(self.config.train.seed)

        resumed_from_epoch = start_epoch - 1
        start_time = time.time()
        progress_enabled = self.config.train.progress and os.environ.get("GRAPHATLAS_PROGRESS", "1") != "0"
        iterator = trange(
            start_epoch,
            self.config.train.epochs + 1,
            disable=not progress_enabled,
            desc=f"{self.config.dataset.name}/{self.config.model.name}/s{self.config.train.seed}",
            leave=False,
            initial=max(0, start_epoch - 1),
            total=self.config.train.epochs,
        )
        final_epoch = start_epoch - 1
        for epoch in iterator:
            final_epoch = epoch
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss_config = _loss_config_for_epoch(self.config, epoch)
            loss, loss_values = compute_loss(
                model, output, data, loss_config, task_name=self.config.dataset.task
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss_values}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.grad_clip)
            optimizer.step()

            should_evaluate = epoch == 1 or epoch % self.config.train.eval_every == 0 or epoch == self.config.train.epochs
            if should_evaluate:
                model.eval()
                with torch.no_grad():
                    eval_output = model(data)
                if self.config.dataset.task == "node_classification" and not data.is_multilabel:
                    val_accuracy = accuracy(eval_output["logits"].detach(), data.y, data.val_mask)
                    train_accuracy = accuracy(eval_output["logits"].detach(), data.y, data.train_mask)
                else:
                    val_accuracy = float("nan")
                    train_accuracy = float("nan")
                val_score = output_primary_metric(eval_output, data, "val")
                train_score = output_primary_metric(eval_output, data, "train")
                row = {
                    "epoch": epoch,
                    "val_metric": val_score,
                    "train_metric": train_score,
                    "val_accuracy": val_accuracy,
                    "train_accuracy": train_accuracy,
                    **loss_values,
                }
                history.append(row)
                iterator.set_postfix(loss=f"{loss_values['total']:.3f}", val=f"{val_score:.3f}")
                if val_score > best_val + 1e-8:
                    best_val = val_score
                    best_epoch = epoch
                    stale = 0
                    best_state = copy.deepcopy(model.state_dict())
                    _atomic_torch_save(best_state, best_path)
                else:
                    stale += self.config.train.eval_every

                pd.DataFrame(history).to_csv(history_path, index=False)
                _atomic_torch_save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "stale": stale,
                        "history": history,
                        "rng_state": _rng_state(),
                        "elapsed_seconds": prior_runtime_seconds + (time.time() - start_time),
                    },
                    last_path,
                )
                if on_evaluation is not None:
                    on_evaluation(epoch, val_score)
                if stale >= self.config.train.patience:
                    break

        # If the requested epoch budget was already reached before this call,
        # last.pt still makes the run finalizable without retraining.
        if best_path.exists():
            best_state = torch.load(best_path, map_location=self.device, weights_only=True)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            if model.__class__.__name__ == "GraphAtlas":
                final_output = model(data, transport_diagnostics=True)
            else:
                final_output = model(data)
        metrics = evaluate_output(
            model,
            final_output,
            data,
            self.config.train.seed,
            include_intervention=include_intervention,
            include_test=evaluate_test,
        )
        if hasattr(model, "config") and model.__class__.__name__ == "GraphAtlas":
            diagnostic_loss = replace(
                self.config.loss,
                cocycle=1.0,
                inverse_cycle=1.0,
                path_consistency=1.0,
                metric=1.0,
                chart_rank=1.0,
                geometry=1.0 if data.latent_positions is not None else 0.0,
                sample_nodes=min(max(self.config.loss.sample_nodes, 64), data.num_nodes),
            )
            with torch.no_grad():
                diagnostics = atlas_regularization_terms(model, final_output, data, diagnostic_loss)
            metrics["cocycle_error"] = float(diagnostics["cocycle"].detach().cpu())
            metrics["inverse_cycle_error"] = float(diagnostics["inverse_cycle"].detach().cpu())
            metrics["triple_cocycle_error"] = float(diagnostics["triple_cocycle"].detach().cpu())
            metrics["path_consistency_error"] = float(diagnostics["path_consistency"].detach().cpu())
            metrics["metric_compatibility_error"] = float(diagnostics["metric"].detach().cpu())
            metrics["metric_direction_error"] = float(diagnostics["metric_direction_error"].detach().cpu())
            metrics["metric_scale_error"] = float(diagnostics["metric_scale_error"].detach().cpu())
            metrics["chart_rank_error"] = float(diagnostics["chart_rank"].detach().cpu())
            metrics["chart_min_relative_singular_value"] = float(
                diagnostics["chart_min_relative_singular_value"].detach().cpu()
            )
            metrics["chart_condition_number"] = float(diagnostics["chart_condition_number"].detach().cpu())
            metrics["geometry_error"] = float(diagnostics["geometry"].detach().cpu())
        else:
            metrics["cocycle_error"] = float("nan")
            metrics["inverse_cycle_error"] = float("nan")
            metrics["triple_cocycle_error"] = float("nan")
            metrics["path_consistency_error"] = float("nan")
            metrics["metric_compatibility_error"] = float("nan")
            metrics["metric_direction_error"] = float("nan")
            metrics["metric_scale_error"] = float("nan")
            metrics["chart_rank_error"] = float("nan")
            metrics["chart_min_relative_singular_value"] = float("nan")
            metrics["chart_condition_number"] = float("nan")
            metrics["geometry_error"] = float("nan")
        metrics.update(
            {
                "dataset": self.config.dataset.name,
                "task": self.config.dataset.task,
                "model": self.config.model.label or self.config.model.name,
                "model_family": self.config.model.name,
                "condition_id": self.config.dataset.condition_id or "",
                "intervention_profile": self.config.dataset.intervention_profile or "",
                "certified_routing_mode": self.config.model.certified_routing_mode,
                "transportability_beta": self.config.model.transportability_beta,
                "readout_mode": self.config.model.readout_mode,
                "coordinate_activation": self.config.model.coordinate_activation,
                "coordinate_left_linear": self.config.model.coordinate_left_linear,
                "link_decoder": self.config.model.link_decoder,
                "seed": self.config.train.seed,
                "best_epoch": best_epoch,
                "last_epoch": final_epoch,
                "resumed_from_epoch": resumed_from_epoch,
                "trained_epochs_this_call": max(0, final_epoch - resumed_from_epoch),
                "best_val_metric": best_val,
                "parameters": count_parameters(model),
                "num_nodes": data.num_nodes,
                "num_edges": int(data.edge_index.shape[1]),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0,
                "runtime_seconds": prior_runtime_seconds + (time.time() - start_time),
                "device": str(self.device),
                "split": int(data.metadata.get("split", self.config.dataset.split)) if data.metadata else int(self.config.dataset.split),
            }
        )
        transport_names = (
            "transportability_mean", "transportability_std", "transportability_min",
            "transportability_max", "distortion_mean", "fraction_q_below_025",
            "fraction_q_above_075",
            "active_source_chart_count", "q_route_spread_mean", "q_route_spread_p90",
            "routing_kl_from_membership", "fraction_edges_with_multiple_source_charts",
            "routing_total_variation_from_membership", "routing_entropy_mean",
            "active_target_chart_count", "fraction_edges_with_multiple_target_charts",
            "q_route_spread_median", "q_route_spread_max",
            "q_delta_closure_error", "routing_opportunity", "irreducible_transport_risk",
            "cross_chart_mass", "oracle_agreement",
        )
        layer_diagnostics = final_output.get("layer_diagnostics", [])
        for name in transport_names:
            values = [entry[name] for entry in layer_diagnostics if name in entry]
            if not values:
                metrics[name] = float("nan")
            elif name.endswith("_min"):
                metrics[name] = float(torch.stack(values).min().detach().cpu())
            elif name.endswith("_max"):
                metrics[name] = float(torch.stack(values).max().detach().cpu())
            else:
                metrics[name] = float(torch.stack(values).mean().detach().cpu())
        if isinstance(final_output.get("membership"), torch.Tensor):
            membership = final_output["membership"]
            active = (membership > 0.05).sum(dim=-1)
            entropy = -(membership.clamp_min(1e-12) * membership.clamp_min(1e-12).log()).sum(dim=-1)
            utilization = membership.mean(dim=0)
            metrics["mean_active_charts"] = float(active.float().mean().detach().cpu())
            metrics["overlap_ratio"] = float((active > 1).float().mean().detach().cpu())
            metrics["membership_entropy_mean"] = float(entropy.mean().detach().cpu())
            metrics["membership_entropy_p90"] = float(torch.quantile(entropy, 0.90).detach().cpu())
            metrics["chart_utilization_min"] = float(utilization.min().detach().cpu())
            metrics["chart_utilization_max"] = float(utilization.max().detach().cpu())
        else:
            for name in ("mean_active_charts", "overlap_ratio", "membership_entropy_mean", "membership_entropy_p90", "chart_utilization_min", "chart_utilization_max"):
                metrics[name] = float("nan")
        pd.DataFrame(history).to_csv(history_path, index=False)
        prediction_payload = {
            "logits": final_output["logits"].detach().cpu(),
            "probabilities": (torch.sigmoid(final_output["logits"]) if data.is_multilabel else torch.softmax(final_output["logits"], dim=-1)).detach().cpu(),
            "labels": data.y.detach().cpu(),
            "train_mask": data.train_mask.detach().cpu(),
            "val_mask": data.val_mask.detach().cpu(),
            "test_mask": data.test_mask.detach().cpu(),
        }
        for key in ("embedding", "membership", "coordinates", "observation_vectors"):
            value = final_output.get(key)
            if torch.is_tensor(value):
                prediction_payload[key] = value.detach().cpu()
        if layer_diagnostics:
            for key in ("node_cross_chart_mass", "node_q_spread", "node_transport_risk"):
                value = layer_diagnostics[-1].get(key)
                if torch.is_tensor(value):
                    prediction_payload[key] = value.detach().cpu()
        if data.link_split is not None:
            for field_name in ("train_pos", "train_neg", "val_pos", "val_neg", "test_pos", "test_neg"):
                prediction_payload[f"link_{field_name}"] = getattr(data.link_split, field_name).detach().cpu()
        _atomic_torch_save(prediction_payload, run_dir / "predictions.pt")
        save_json(metrics, run_dir / "metrics.json")
        return model, metrics
