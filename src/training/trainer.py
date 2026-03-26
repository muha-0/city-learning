"""
Training loop, history logging, visualisation, and checkpoint saving.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from .masking import make_train_masks, make_fixed_masks
from .losses import (
    corrupt_inputs_with_flags,
    compute_losses,
    compute_metrics,
    evaluate_with_masks,
    evaluate_losses_only,
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    data_train: Data,
    data_val: Data,
    data_test: Data,
    num_highway: int,
    highway_mask_id: int,
    lanes_mask_id: int,
    oneway_mask_id: int,
    device: torch.device,
    epochs: int = 10_000,
    p_mask: float = 0.30,
    eval_every: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_seed: int = 999,
) -> dict:
    """
    Full training loop with fixed-mask validation evaluation.

    Returns the history dict and the model (weights updated in-place).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_masks_fixed = make_fixed_masks(data_val, p_mask=p_mask, seed=val_seed)

    history = {
        "epoch": [],
        "train_total": [],
        "val_total": [],
        "train_losses": {k: [] for k in ["hwy", "lan", "onw", "wid", "max", "min"]},
        "val_losses":   {k: [] for k in ["hwy", "lan", "onw", "wid", "max", "min"]},
        "metric_epoch": [],
        "train_metrics": {k: [] for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae"]},
        "val_metrics":   {k: [] for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae"]},
        "log_vars": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_masks = make_train_masks(data_train, p_mask)
        x_cont, highway_in, lanes_in, oneway_in = corrupt_inputs_with_flags(
            data_train, train_masks, highway_mask_id, lanes_mask_id, oneway_mask_id
        )
        pred = model(x_cont, highway_in, lanes_in, oneway_in, data_train.edge_index)
        total_loss, losses = compute_losses(model, pred, data_train, train_masks, device)
        total_loss.backward()
        optimizer.step()

        val_total, val_losses = evaluate_losses_only(
            model, data_val, val_masks_fixed, device,
            highway_mask_id, lanes_mask_id, oneway_mask_id,
        )

        history["epoch"].append(epoch)
        history["train_total"].append(total_loss.item())
        history["val_total"].append(val_total)
        for k in ["hwy", "lan", "onw", "wid", "max", "min"]:
            history["train_losses"][k].append(losses[k].item())
            history["val_losses"][k].append(val_losses[k])
        history["log_vars"].append(model.log_vars.detach().cpu().numpy().copy())

        do_metrics = (epoch == 1) or (epoch % eval_every == 0)
        if do_metrics:
            train_metrics = compute_metrics(pred, data_train, train_masks, num_highway)
            _, _, val_metrics = evaluate_with_masks(
                model, data_val, val_masks_fixed, num_highway, device,
                highway_mask_id, lanes_mask_id, oneway_mask_id,
            )
            history["metric_epoch"].append(epoch)
            for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae"]:
                history["train_metrics"][k].append(train_metrics[k])
                history["val_metrics"][k].append(val_metrics[k])

            print(
                f"Epoch {epoch:04d} | total={total_loss.item():.4f} | "
                f"hwy={losses['hwy'].item():.3f} lan={losses['lan'].item():.3f} "
                f"onw={losses['onw'].item():.3f} wid={losses['wid'].item():.3f} "
                f"max={losses['max'].item():.3f} min={losses['min'].item():.3f} | "
                f"VAL hwy_F1={val_metrics['hwy_macro_f1']:.3f} "
                f"lan_F1={val_metrics['lan_macro_f1']:.3f} "
                f"onw_AUROC={val_metrics['onw_auroc']:.3f} "
                f"wid_MAE={val_metrics['wid_mae_m']:.3f}"
            )

    return history


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_history(history: dict) -> None:
    """Plot train/val total loss and per-task losses."""
    plt.figure()
    plt.plot(history["epoch"], history["train_total"], label="Train")
    plt.plot(history["epoch"], history["val_total"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Total Loss (Train vs Val)")
    plt.legend()
    plt.show()

    task_titles = {
        "hwy": "Highway CE", "lan": "Lanes CE", "onw": "Oneway BCE",
        "wid": "Width Huber", "max": "Max Speed Huber", "min": "Min Speed Huber",
    }
    for t, title in task_titles.items():
        plt.figure()
        plt.plot(history["epoch"], history["train_losses"][t], label="Train")
        plt.plot(history["epoch"], history["val_losses"][t], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title} (masked-only)")
        plt.legend()
        plt.show()

    metric_titles = {
        "hwy_macro_f1": "Highway Macro-F1",
        "lan_macro_f1": "Lanes Macro-F1",
        "onw_auroc": "Oneway AUROC",
        "wid_mae_m": "Width MAE (m)",
        "max_mae": "Max Speed MAE",
        "min_mae": "Min Speed MAE",
    }
    for m, title in metric_titles.items():
        plt.figure()
        plt.plot(history["metric_epoch"], history["train_metrics"][m], label="Train")
        plt.plot(history["metric_epoch"], history["val_metrics"][m], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title(title)
        plt.legend()
        plt.show()


# ---------------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------------

def evaluate_test(
    model,
    data_test: Data,
    num_highway: int,
    highway_mask_id: int,
    lanes_mask_id: int,
    oneway_mask_id: int,
    device: torch.device,
    p_mask: float = 0.30,
    test_seed: int = 2025,
) -> tuple:
    """Evaluate on the held-out test split with fixed masks."""
    test_masks = make_fixed_masks(data_test, p_mask=p_mask, seed=test_seed)
    total, losses, metrics = evaluate_with_masks(
        model, data_test, test_masks, num_highway, device,
        highway_mask_id, lanes_mask_id, oneway_mask_id,
    )
    print("TEST metrics:", metrics)
    print("TEST losses: ", losses)
    return total, losses, metrics


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    hwy2id: dict,
    id2hwy: dict,
    scalers: dict,
    token_ids: dict,
    meta: dict,
    save_path: str,
) -> None:
    """
    Save model weights and all preprocessing artefacts needed for cross-city inference.

    Parameters
    ----------
    scalers : dict with keys 'length', 'width', 'max', 'min' (ZScaler instances).
    token_ids : dict with HIGHWAY_MASK_ID, LANES_MASK_ID, LANES_MISS_ID,
                ONEWAY_MASK_ID, ONEWAY_MISS_ID.
    meta : dict with seed, grid_size_m, p_mask, etc.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_cfg": {
            "num_highway": len(hwy2id),
            "hwy_emb_dim": 16,
            "lanes_emb_dim": 8,
            "oneway_emb_dim": 4,
            "cont_dim": 12,
            "hidden": 32,
            "heads": 2,
            "dropout": 0.1,
        },
        "hwy2id": hwy2id,
        "id2hwy": id2hwy,
        "token_ids": {k: int(v) for k, v in token_ids.items()},
        "scalers": {
            name: {"mu": float(s.mu), "sd": float(s.sd)}
            for name, s in scalers.items()
        },
        "meta": meta,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")
