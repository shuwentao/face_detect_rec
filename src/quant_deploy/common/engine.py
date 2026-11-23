"""训练引擎：AMP + AdamW + cosine，记录 history，保存 checkpoint。"""
import os
import time
import json
import math
import torch
from .utils import AverageMeter, save_json, ensure_dir


class Trainer:
    def __init__(self, model, train_loader, val_loader, out_dir,
                 epochs=20, lr=5e-4, wd=5e-4, device="cuda", amp=True,
                 log_every=50, warmup_epochs=1, min_lr_ratio=0.01,
                 val_every=1, val_batches=None, clip_grad=5.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = ensure_dir(out_dir)
        self.ckpt_dir = ensure_dir(os.path.join(out_dir, "ckpt"))
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"
        self.log_every = log_every
        self.warmup = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.val_every = val_every
        self.val_batches = val_batches

        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.history = {"epochs": [], "steps": [], "config": {
            "epochs": epochs, "lr": lr, "wd": wd, "amp": self.amp}}
        self.best = float("inf")
        self.clip_grad = clip_grad
        self.nan_batches = 0
        self.val_nan_batches = 0

    def lr_at(self, epoch, step_in_epoch, steps_per_epoch):
        """warmup + cosine。"""
        if epoch < self.warmup:
            t = (step_in_epoch + epoch * steps_per_epoch) / max(
                1, self.warmup * steps_per_epoch)
            return self.lr * (0.1 + 0.9 * t)
        total = max(1, self.epochs - self.warmup)
        prog = (epoch - self.warmup) + step_in_epoch / max(1, steps_per_epoch)
        prog = min(prog / total, 1.0)
        return self.lr * (self.min_lr_ratio +
                          (1 - self.min_lr_ratio) * 0.5 *
                          (1 + math.cos(math.pi * prog)))

    def _to_device(self, tgts):
        return [(t[0].to(self.device, non_blocking=True),
                 t[1].to(self.device, non_blocking=True),
                 t[2].to(self.device, non_blocking=True),
                 t[3].to(self.device, non_blocking=True)) for t in tgts]

    def train_epoch(self, epoch, criterion):
        self.model.train()
        m = {k: AverageMeter() for k in ("loss", "obj", "box", "lm")}
        spe = len(self.train_loader)
        t0 = time.time()
        for i, (imgs, tgts, _) in enumerate(self.train_loader):
            lr = self.lr_at(epoch, i, spe)
            for g in self.opt.param_groups:
                g["lr"] = lr
            imgs = imgs.to(self.device, non_blocking=True)
            tgts = self._to_device(tgts)
            with torch.amp.autocast("cuda", enabled=self.amp):
                preds = self.model(imgs)
                out = criterion(preds, tgts)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(out["loss"]).backward()
            # 先反缩放再裁剪：AMP 下 fp16 溢出会让 GradScaler 整步丢弃，
            # 裁剪能保住这些 batch 的更新
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.opt)
            self.scaler.update()
            loss_val = out["loss"].detach().item()
            if math.isfinite(loss_val):
                for k in ("loss", "obj", "box", "lm"):
                    m[k].update(out[k].detach().item(), 1)
            else:
                self.nan_batches += 1
            if i % self.log_every == 0:
                print(f"  ep{epoch+1} step {i}/{spe} loss={m['loss'].val:.3f} "
                      f"obj={m['obj'].val:.3f} box={m['box'].val:.3f} "
                      f"lm={m['lm'].val:.3f} lr={lr:.2e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            self.history["steps"].append(
                {"epoch": epoch + 1, "step": i, "loss": loss_val,
                 "obj": out["obj"].detach().item(),
                 "box": out["box"].detach().item(),
                 "lm": out["lm"].detach().item()})
        if self.nan_batches:
            print(f"  [warn] 非有限 loss 的 batch 累计 {self.nan_batches} 个"
                  f"（已排除出均值）", flush=True)
        return {k: m[k].avg for k in m}, time.time() - t0

    @torch.no_grad()
    def validate(self, criterion):
        self.model.eval()
        m = {k: AverageMeter() for k in ("loss", "obj", "box", "lm")}
        n = self.val_batches or len(self.val_loader)
        for i, (imgs, tgts, _) in enumerate(self.val_loader):
            if i >= n:
                break
            imgs = imgs.to(self.device, non_blocking=True)
            tgts = self._to_device(tgts)
            with torch.amp.autocast("cuda", enabled=self.amp):
                out = criterion(self.model(imgs), tgts)
            # 训练侧有 GradScaler 兜底，验证侧没有：非有限 batch 必须排除，
            # 否则 val_loss 变 nan，best 权重的选择逻辑会失效
            if math.isfinite(float(out["loss"])):
                for k in ("loss", "obj", "box", "lm"):
                    m[k].update(float(out[k]), 1)
            else:
                self.val_nan_batches += 1
        return {k: m[k].avg for k in m}

    def fit(self, criterion):
        for ep in range(self.epochs):
            tr, dt = self.train_epoch(ep, criterion)
            rec = {"epoch": ep + 1, "train_loss": tr["loss"], "obj": tr["obj"],
                   "box": tr["box"], "lm": tr["lm"], "time_sec": round(dt, 1)}
            if (ep + 1) % self.val_every == 0 and self.val_loader is not None:
                va = self.validate(criterion)
                rec.update({f"val_{k}": v for k, v in va.items()})
            self.history["epochs"].append(rec)
            print(f"[epoch {ep+1}/{self.epochs}] " +
                  " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in rec.items()), flush=True)
            save_json(self.history, os.path.join(self.out_dir, "history.json"))
            torch.save({"model": self.model.state_dict(), "epoch": ep + 1,
                        "history": self.history},
                       os.path.join(self.ckpt_dir, "last.pt"))
            vl = rec.get("val_loss", tr["loss"])
            if vl < self.best:
                self.best = vl
                torch.save({"model": self.model.state_dict(), "epoch": ep + 1},
                           os.path.join(self.ckpt_dir, "best.pt"))
        torch.save({"model": self.model.state_dict(), "epoch": self.epochs},
                   os.path.join(self.ckpt_dir, "final.pt"))
        return self.history
