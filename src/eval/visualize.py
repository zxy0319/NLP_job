# -*- coding: utf-8 -*-
"""混淆矩阵可视化模块（成员 C 负责）。

将 metrics.confusion_matrix 输出的嵌套字典绘制为热力图并保存为 PNG，
供报告 §5（评估）引用。matplotlib 采用懒加载 + Agg 无界面后端，仅在实际
调用出图时才 import，未安装 matplotlib 时不影响评测主流程。

标签（positive/negative）为英文，无需中文字体即可正常渲染；标题可传入，
默认使用英文，避免无中文字体环境下出现乱码。
"""

from __future__ import annotations

from pathlib import Path


def plot_confusion_matrix(
    cm: dict,
    labels: list,
    out_path,
    title: str | None = None,
    dpi: int = 150,
) -> str:
    """绘制混淆矩阵热力图并保存为 PNG，返回输出路径。

    cm: cm[true][pred] -> count（与 metrics.confusion_matrix 输出一致）
    labels: 类别标签列表（如 ["positive", "negative"]）
    out_path: 输出 PNG 路径（str 或 Path）
    title: 图标题，默认 "Confusion Matrix"
    """
    import matplotlib

    matplotlib.use("Agg")  # 无 GUI 后端，适配服务器 / CI / 命令行
    import matplotlib.pyplot as plt

    labels = list(labels)
    n = len(labels)
    matrix = [[cm[t].get(p, 0) for p in labels] for t in labels]
    total = sum(sum(row) for row in matrix)
    max_v = max(max(row) for row in matrix) if total else 1

    fig, ax = plt.subplots(figsize=(max(5, n + 2), max(4, n + 1)))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title or "Confusion Matrix")

    # 每格标注：绝对数量 + 行占比（行=真实标签，行内占比更能反映召回）
    for i in range(n):
        row_sum = sum(matrix[i])
        for j in range(n):
            v = matrix[i][j]
            pct = v / row_sum if row_sum else 0.0
            color = "white" if v > 0.5 * max_v else "black"
            ax.text(j, i, f"{v}\n({pct:.1%})", ha="center", va="center",
                    color=color, fontsize=11)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
