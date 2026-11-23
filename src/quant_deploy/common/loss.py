"""检测损失：focal(obj) + smoothL1(box) + smoothL1(landmark, 带可见 mask)。"""
import torch
import torch.nn.functional as F


def focal_bce(pred, target, alpha=0.25, gamma=2.0, reduction="mean"):
    """pred: 已 sigmoid 的概率；target: 0/1。"""
    p = pred.clamp(1e-6, 1 - 1e-6)
    bce = -(target * p.log() + (1 - target) * (1 - p).log())
    pt = target * p + (1 - target) * (1 - p)
    loss = alpha * (1 - pt).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def smooth_l1(pred, target, reduction="mean"):
    return F.smooth_l1_loss(pred, target, reduction=reduction, beta=1 / 9)


class DetectionLoss:
    def __init__(self, cls_w=1.0, box_w=2.0, lm_w=0.5):
        self.cls_w, self.box_w, self.lm_w = cls_w, box_w, lm_w

    def __call__(self, preds, tgts):
        """preds: list[(B,15,H,W)]；tgts: list[(cls,box,lm,lmm)] 每项 (B,H,W..)。"""
        loss_obj = preds[0].new_zeros(())
        loss_box = preds[0].new_zeros(())
        loss_lm = preds[0].new_zeros(())
        n_pos = 0
        n_lm = 0
        for p, (tcls, tbox, tlm, tlmm) in zip(preds, tgts):
            obj = p[:, 0].sigmoid()
            loss_obj = loss_obj + focal_bce(obj, tcls)
            pos = tcls > 0.5
            n_pos += int(pos.sum())
            pb = p[:, 1:5].permute(0, 2, 3, 1)[pos]
            tb = tbox[pos]
            if pb.numel() > 0:
                loss_box = loss_box + smooth_l1(pb, tb, reduction="sum")
            m = (tlmm > 0.5) & pos
            n_lm += int(m.sum())
            if m.any():
                pl = p[:, 5:15].permute(0, 2, 3, 1)[m]
                tl = tlm[m]
                loss_lm = loss_lm + smooth_l1(pl, tl, reduction="sum")
        n_pos = max(n_pos, 1)
        n_lm = max(n_lm, 1)
        box = loss_box / n_pos
        lm = loss_lm / n_lm
        total = self.cls_w * loss_obj + self.box_w * box + self.lm_w * lm
        return {"loss": total, "obj": loss_obj, "box": box, "lm": lm,
                "n_pos": n_pos, "n_lm": n_lm}
