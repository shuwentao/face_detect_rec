"""报告生成：训练曲线、可视化样例、单算法 Markdown 报告、四算法对比汇总。"""
import os
import json
import base64
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _b64(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def plot_curves(history, out_png):
    ep = history.get("epochs", [])
    steps = history.get("steps", [])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if steps:
        xs = [s["step"] + (s["epoch"] - 1) * 10000 for s in steps]
        ax[0].plot(xs, [s["loss"] for s in steps], lw=0.6, alpha=0.6)
        # 平滑
        k = max(1, len(steps) // 200)
        sm = [sum(s["loss"] for s in steps[i:i + k]) / k
              for i in range(0, len(steps) - k + 1, k)]
        ax[0].plot(xs[:len(sm) * k:k], sm, color="r", lw=1.5)
        ax[0].set_title("train loss (step)")
        ax[0].set_xlabel("step")
        ax[0].grid(alpha=0.3)
    if ep:
        x = [e["epoch"] for e in ep]
        ax[1].plot(x, [e["train_loss"] for e in ep], marker="o", label="train")
        if "val_loss" in ep[0]:
            ax[1].plot(x, [e.get("val_loss", float("nan")) for e in ep],
                       marker="s", label="val")
        ax[1].set_title("loss per epoch")
        ax[1].set_xlabel("epoch")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        keys = ["obj", "box", "lm"]
        for k in keys:
            ax[2].plot(x, [e.get(k, 0) for e in ep], marker=".", label=k)
        ax[2].set_title("loss components")
        ax[2].set_xlabel("epoch")
        ax[2].legend()
        ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def save_vis(model, ann_json, img_dir, out_png, device="cuda", n=4,
             input_size=640, seed=0):
    """画若干张验证图的检测框 + 关键点。"""
    import cv2
    import numpy as np
    import torch
    from .eval_api import predict_image
    ann = json.load(open(ann_json))
    imgs = ann["images"]
    step = max(1, len(imgs) // n)
    picks = imgs[seed::step][:n]
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5))
    axes = np.atleast_1d(axes)
    model.eval()
    for ax, info in zip(axes, picks):
        img = cv2.imread(os.path.join(img_dir, info["file_name"]))
        if img is None:
            ax.axis("off")
            continue
        boxes, scores, lms = predict_image(model, img, input_size, device,
                                           conf_thr=0.3)
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for b, lm in zip(boxes, lms):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (px, py) in lm:
                cv2.circle(show, (int(px), int(py)), 2, (255, 0, 0), -1)
        ax.imshow(show)
        ax.set_title(f"{info['file_name'][:24]}  ({len(boxes)} faces)")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    plt.close(fig)
    return out_png


def _fmt(v, nd=4):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def gen_algo_report(algo_dir, algo, cfg, metrics, history=None,
                    curves_png=None, vis_png=None):
    """生成单个算法的 Markdown 报告，返回路径。"""
    os.makedirs(algo_dir, exist_ok=True)
    rows = [("算法", algo), ("backbone", cfg.get("backbone", algo)),
            ("输入尺寸", f"{cfg.get('input_size', 640)}x{cfg.get('input_size', 640)}"),
            ("epochs", cfg.get("epochs", "-")),
            ("batch", cfg.get("batch", "-")),
            ("参数量(M)", _fmt(metrics.get("params_m", -1), 3)),
            ("模型体积(MB)", _fmt(metrics.get("size_mb", -1), 2)),
            ("GFLOPs", _fmt(metrics.get("gflops", -1), 3)),
            ("GPU 延迟(ms)", _fmt(metrics.get("latency_gpu_ms", -1), 2)),
            ("CPU 延迟(ms)", _fmt(metrics.get("latency_cpu_ms", -1), 2)),
            ("AP easy", _fmt(metrics.get("ap_easy", -1))),
            ("AP medium", _fmt(metrics.get("ap_medium", -1))),
            ("AP hard", _fmt(metrics.get("ap_hard", -1))),
            ("NME(mean)", _fmt(metrics.get("nme_mean", -1))),
            ("NME(median)", _fmt(metrics.get("nme_median", -1))),
            ("FR@0.1", _fmt(metrics.get("fr_0.1", -1))),
            ("关键点匹配率", _fmt(metrics.get("match_rate", -1))),
            ("ONNX 体积(MB)", _fmt(metrics.get("onnx_size_mb", -1), 2)),
            ("ONNX 最大误差", _fmt(metrics.get("onnx_max_diff", -1), 6)),
            ("训练耗时(min)", _fmt(metrics.get("train_minutes", -1), 1)),
            ]
    md = [f"# {algo} 人脸检测 + 关键点 训练报告", "",
          f"生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "",
          "## 1. 配置与结果", "",
          "| 项目 | 值 |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in rows]
    if history and history.get("epochs"):
        md += ["", "## 2. 每轮训练记录", "",
               "| epoch | train_loss | obj | box | lm | val_loss | time(s) |",
               "|---|---|---|---|---|---|---|"]
        for e in history["epochs"]:
            md.append(f"| {e['epoch']} | {_fmt(e.get('train_loss', 0))} | "
                      f"{_fmt(e.get('obj', 0))} | {_fmt(e.get('box', 0))} | "
                      f"{_fmt(e.get('lm', 0))} | {_fmt(e.get('val_loss', float('nan')))} "
                      f"| {e.get('time_sec', '-')} |")
    if curves_png and os.path.exists(curves_png):
        md += ["", "## 3. 训练曲线", "", f"![]({os.path.basename(curves_png)})"]
    if vis_png and os.path.exists(vis_png):
        md += ["", "## 4. 验证集可视化（绿框=检测，蓝点=5 关键点）", "",
               f"![]({os.path.basename(vis_png)})"]
    path = os.path.join(algo_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
COLS = [
    ("backbone", "Backbone"), ("params_m", "参数量(M)"), ("size_mb", "模型(MB)"),
    ("gflops", "GFLOPs"), ("latency_gpu_ms", "GPU(ms)"),
    ("latency_cpu_ms", "CPU(ms)"), ("ap_easy", "AP-easy"),
    ("ap_medium", "AP-medium"), ("ap_hard", "AP-hard"),
    ("nme_mean", "NME"), ("fr_0.1", "FR@0.1"), ("match_rate", "匹配率"),
    ("onnx_size_mb", "ONNX(MB)"), ("train_minutes", "训练(min)"),
]


def gen_summary(root, algos, out_dir):
    """algos: list[(name, metrics_dict, curves_png, vis_png)] -> (md, html)"""
    os.makedirs(out_dir, exist_ok=True)
    md = ["# 人脸检测算法对比报告（YuNet / MobileNetV1 / V2 / V3）", "",
          f"生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "",
          "数据：WIDER Face 子集（train 8000 / val 2000 / lmval 800），"
          "关键点标注来自 retinaface labelv2(5 点)。", "",
          "## 1. 总览", "",
          "| 算法 | " + " | ".join(c for _, c in COLS) + " |",
          "|---|" + "---|" * len(COLS)]
    for name, m, _, _ in algos:
        md.append("| " + name + " | " + " | ".join(
            _fmt(m.get(k, -1), 3 if "M" in c or "MB" in c else 4)
            for k, c in COLS) + " |")

    md += ["", "## 2. 各算法训练曲线"]
    for name, _, cp, _ in algos:
        if cp and os.path.exists(cp):
            md += ["", f"### {name}", "", f"![]({os.path.basename(cp)})"]
    md += ["", "## 3. 各算法检测可视化"]
    for name, _, _, vp in algos:
        if vp and os.path.exists(vp):
            md += ["", f"### {name}", "", f"![]({os.path.basename(vp)})"]
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    # ---- 自包含 HTML ----
    head = ["<html><head><meta charset='utf-8'><title>人脸检测算法对比</title>",
            "<style>",
            "body{font-family:-apple-system,'Segoe UI',Arial,sans-serif;"
            "margin:24px;color:#222;background:#fafafa}",
            "h1{border-bottom:2px solid #333;padding-bottom:8px}",
            "table{border-collapse:collapse;width:100%;background:#fff;"
            "box-shadow:0 1px 3px rgba(0,0,0,.12)}",
            "th,td{border:1px solid #ddd;padding:8px 10px;text-align:center}",
            "th{background:#f0f3f7}", "tr:nth-child(even){background:#fbfbfd}",
            "img{max-width:100%;border:1px solid #ddd;margin:8px 0}",
            ".card{background:#fff;padding:16px;margin:16px 0;"
            "box-shadow:0 1px 3px rgba(0,0,0,.12)}",
            "</style></head><body>",
            "<h1>人脸检测算法对比报告</h1>",
            f"<p>生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S}　"
            "数据集：WIDER Face 子集（train 8000 / val 2000 / lmval 800）</p>"]
    tbl = ["<div class='card'><h2>1. 指标对比</h2><table>",
           "<tr><th>算法</th>" + "".join(f"<th>{c}</th>" for _, c in COLS) + "</tr>"]
    for name, m, _, _ in algos:
        tbl.append("<tr><td><b>" + name + "</b></td>" + "".join(
            f"<td>{_fmt(m.get(k, -1), 3 if 'M' in c or 'MB' in c else 4)}</td>"
            for k, c in COLS) + "</tr>")
    tbl.append("</table></div>")
    sec = ["<div class='card'><h2>2. 训练曲线</h2>"]
    for name, _, cp, _ in algos:
        b = _b64(cp)
        if b:
            sec.append(f"<h3>{name}</h3><img src='{b}'>")
    sec.append("</div>")
    sec += ["<div class='card'><h2>3. 可视化（绿框检测 / 蓝点关键点）</h2>"]
    for name, _, _, vp in algos:
        b = _b64(vp)
        if b:
            sec.append(f"<h3>{name}</h3><img src='{b}'>")
    sec.append("</div></body></html>")
    html_path = os.path.join(out_dir, "summary.html")
    with open(html_path, "w") as f:
        f.write("\n".join(head + tbl + sec))
    return md_path, html_path
