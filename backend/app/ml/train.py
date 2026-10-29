"""Transfer-learning training CLI.  Example:
    python -m app.ml.train --arch densenet121 --epochs 12 --head-epochs 4 --run-dir ../data/artifacts/runs/my_run
Writes status.json (live), progress.jsonl (per epoch), model.pt (best val-loss checkpoint), config.json,
history.json, split_summary.json, evaluation.json and evaluation.html.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.core.config import get_settings
from app.ml import evaluate as ev
from app.ml.data import DatasetError, eval_transform, make_loader, make_splits, train_transform
from app.ml.model import (
    ARCHS,
    CLASS_NAMES,
    CLASS_TO_IDX,
    PretrainedWeightsError,
    apply_freeze,
    build_model,
    count_parameters,
    model_size_mb,
    pick_device,
    set_train_mode,
)

_CANCEL = False


def _on_term(_sig, _frm):
    global _CANCEL
    _CANCEL = True


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic: readers never see half-written files


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a pretrained CNN for NORMAL vs PNEUMONIA chest X-rays.")
    p.add_argument("--arch", default="densenet121", choices=sorted(ARCHS))
    p.add_argument("--epochs", type=int, default=12, help="total epochs (head phase + fine-tune phase)")
    p.add_argument("--head-epochs", type=int, default=4, help="epochs with the backbone frozen")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--fine-tune-mode", default="last_block", choices=["head_only", "last_block", "full"])
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--horizontal-flip", action="store_true")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--weights-path", type=Path, default=None, help="local torchvision ImageNet state_dict (offline use)")
    p.add_argument("--max-images-per-class", type=int, default=None, help="smoke-test subset")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, required=True, help="checkpoint directory (must be new/empty)")
    p.add_argument("--version", default=None)
    return p.parse_args(argv)


class Status:
    def __init__(self, run_dir: Path, total_epochs: int):
        self.path, self.data = (
            run_dir / "status.json",
            {
                "state": "starting",
                "phase": "setup",
                "epoch": 0,
                "total_epochs": total_epochs,
                "batch": 0,
                "batches": 0,
                "best_val_loss": None,
                "message": "",
                "started_at": datetime.now(UTC).isoformat(),
            },
        )
        self.flush()

    def update(self, **kw) -> None:
        self.data.update(kw, updated_at=datetime.now(UTC).isoformat())
        self.flush()

    def flush(self) -> None:
        write_json(self.path, self.data)


def run_epoch(model, loader, device, criterion, optimizer, status: Status | None, train: bool):
    set_train_mode(model) if train else model.eval()
    total, correct, loss_sum, ys, ss = 0, 0, 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b, (x, y) in enumerate(loader):
            if _CANCEL:
                raise KeyboardInterrupt
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if status and b % 5 == 0:
                    status.update(batch=b + 1, batches=len(loader))
            loss_sum += loss.detach().item() * len(y)
            total += len(y)
            correct += int((logits.argmax(1) == y).sum())
            ys.append(y.cpu().numpy())
            ss.append(torch.softmax(logits.detach(), 1)[:, 1].cpu().numpy())
    y_all, s_all = np.concatenate(ys), np.concatenate(ss)
    auc = float(ev.roc_auc_score(y_all, s_all)) if len(np.unique(y_all)) == 2 else None
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1), "auc": auc}


def main(argv=None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _on_term)
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "model.pt").exists():
        print(f"Refusing to overwrite existing checkpoint in {run_dir}", file=sys.stderr)
        return 2
    status = Status(run_dir, args.epochs)
    t_start = time.time()
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = pick_device(get_settings().inference_device)
        if args.head_epochs > args.epochs:
            raise ValueError("head-epochs cannot exceed epochs")
        if args.fine_tune_mode == "head_only":
            args.head_epochs = args.epochs
        root = args.dataset_dir or get_settings().dataset_dir
        status.update(state="running", phase="loading data", message="Scanning dataset and building splits")
        splits = make_splits(Path(root), args.val_fraction, args.seed, args.max_images_per_class)
        write_json(run_dir / "split_summary.json", splits["summary"])
        size = get_settings().image_size
        tl = make_loader(
            splits["train"], train_transform(size, args.horizontal_flip), args.batch_size, True, args.num_workers, args.seed
        )
        vl = make_loader(splits["val"], eval_transform(size), args.batch_size, False, args.num_workers)
        te = make_loader(splits["test"], eval_transform(size), args.batch_size, False, args.num_workers)

        status.update(phase="loading model", message="Loading pretrained weights" if args.pretrained else "Random init")
        model, source = build_model(args.arch, args.pretrained, args.weights_path)
        model.to(device)
        total_params, _ = count_parameters(model)

        counts = splits["summary"]["counts"]["train"]
        n = sum(counts.values())
        weights = torch.tensor([n / (2 * max(counts[c], 1)) for c in CLASS_NAMES], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        history, best_loss, best_epoch, bad = [], float("inf"), -1, 0
        t_train = time.time()
        phases = [("head", 0, args.head_epochs, "head_only", args.lr)]
        if args.epochs > args.head_epochs:
            phases.append(("fine-tune", args.head_epochs, args.epochs, args.fine_tune_mode, args.finetune_lr))
        for phase, start, end, mode, lr in phases:
            if start >= end:
                continue
            apply_freeze(model, args.arch, mode)
            _, trainable = count_parameters(model)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=end - start)
            bad = 0  # patience restarts for each phase
            for epoch in range(start, end):
                status.update(
                    phase=phase, epoch=epoch + 1, batch=0, batches=len(tl), message=f"{phase}: {trainable:,} trainable params"
                )
                t0 = time.time()
                tr = run_epoch(model, tl, device, criterion, opt, status, True)
                va = run_epoch(model, vl, device, criterion, None, None, False)
                sched.step()
                rec = {
                    "epoch": epoch + 1,
                    "phase": phase,
                    "lr": opt.param_groups[0]["lr"],
                    "train_loss": tr["loss"],
                    "train_acc": tr["acc"],
                    "val_loss": va["loss"],
                    "val_acc": va["acc"],
                    "val_auc": va["auc"],
                    "seconds": round(time.time() - t0, 1),
                }
                history.append(rec)
                with open(run_dir / "progress.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if va["loss"] < best_loss - 1e-4:
                    best_loss, best_epoch, bad = va["loss"], epoch + 1, 0
                    tmp = run_dir / "model.pt.tmp"
                    torch.save(
                        {
                            "arch": args.arch,
                            "state_dict": model.state_dict(),
                            "class_to_idx": CLASS_TO_IDX,
                            "image_size": size,
                            "epoch": epoch + 1,
                        },
                        tmp,
                    )
                    os.replace(tmp, run_dir / "model.pt")
                else:
                    bad += 1
                status.update(best_val_loss=best_loss, message=f"epoch {epoch + 1} done")
                if bad >= args.patience:
                    status.update(message=f"early stopping in {phase} phase (patience {args.patience})")
                    break
        train_seconds = time.time() - t_train
        if best_epoch < 0:
            raise RuntimeError("No checkpoint was produced.")
        write_json(run_dir / "history.json", history)

        # ---- threshold from validation, evaluation on the untouched test split ----
        status.update(phase="evaluating", message="Selecting threshold on validation and evaluating on test")
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        yv, sv = ev.predict_scores(model, vl, device)
        threshold, method = ev.select_threshold(yv, sv)
        yt, st = ev.predict_scores(model, te, device)
        report = ev.build_report(
            model,
            yt,
            st,
            threshold,
            method,
            device,
            "test",
            args.seed,
            {"split_summary": splits["summary"], "best_epoch": best_epoch},
        )
        write_json(run_dir / "evaluation.json", report)
        (run_dir / "evaluation.html").write_text(ev.render_html(report))
        cfg = {
            "version": args.version or run_dir.name,
            "architecture": args.arch,
            "pretrained": args.pretrained or bool(args.weights_path),
            "pretrained_source": source,
            "class_to_idx": CLASS_TO_IDX,
            "positive_class": "PNEUMONIA",
            "threshold": threshold,
            "threshold_method": method,
            "cam_layer": ARCHS[args.arch].cam_layer,
            "image_size": size,
            "parameter_count": total_params,
            "model_size_mb": round(model_size_mb(model), 2),
            "best_epoch": best_epoch,
            "training_seconds": round(train_seconds, 1),
            "total_seconds": round(time.time() - t_start, 1),
            "seed": args.seed,
            "device": device.type,
            "torch": torch.__version__,
            "training_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "run_dir"},
            "created_at": datetime.now(UTC).isoformat(),
        }
        write_json(run_dir / "config.json", cfg)
        status.update(state="completed", phase="done", message="Training and evaluation finished")
        return 0
    except KeyboardInterrupt:
        status.update(state="cancelled", message="Cancelled by user")
        return 130
    except (PretrainedWeightsError, DatasetError, ValueError) as exc:
        status.update(state="failed", message=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        status.update(state="failed", message=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
