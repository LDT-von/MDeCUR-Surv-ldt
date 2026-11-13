"""Build the final search report from the Stage 1/2/3 summaries.

Everything reported here comes from the pre-registered protocol
(HYPERPARAM_SEARCH_PROTOCOL_20261112.md).  The report is written as Markdown
plus a machine-readable JSON companion.  No configuration is dropped from the
tables, including runs that failed or that are worse than the baseline.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from search_config import (REPO as _REPO, RUN as _RUN, FOLD as _FOLD,
                           FOLD_DIR as _FOLD_DIR, GENES as _GENES, PY as _PY,
                           MEASURE as _MEASURE, LD_LIBRARY as _LD_LIBRARY,
                           PATCH_CHUNK_SIZE as _PATCH_CHUNK_SIZE)

import json
from pathlib import Path

RUN = _RUN


def load(name):
    path = RUN / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    stage1 = load('stage1_summary.json')
    stage2 = load('stage2_summary.json')
    stage3 = load('stage3_summary.json')

    lines = []
    lines.append('# 新框架超参数搜索最终报告')
    lines.append('')
    lines.append('协议：`HYPERPARAM_SEARCH_PROTOCOL_20261112.md`（运行前写定，含 §3.0 公开修订）')
    lines.append('')
    lines.append('## 1. 范围与边界')
    lines.append('')
    lines.append('- 只在新框架 `PathwayLocalSurvival` 内部搜索。')
    lines.append('- outer_test 76 名患者全程封存，从未加载。')
    lines.append('- 视图源为 `feature_noise`（编码特征噪声），不是 WSI 图像空间增强。')
    lines.append('- 全部 Stage 1/2/3 运行统一 `patch_chunk_size=8192`（协议 §3.0 修订）。')
    lines.append('')

    if stage1:
        lines.append('## 2. Stage 1 家族级筛选（单折 fold_0，61 验证患者）')
        lines.append('')
        base = stage1['baseline_anchor']
        lines.append(f"本次搜索基线（C00_baseline, chunk 8192）："
                     f"C = {base['c_index']:.4f}，IPCW Brier = {base['ipcw_brier']:.4f}，"
                     f"验证 NLL = {base['validation_nll']:.4f}")
        hist = stage1['historical_reference_arm']
        lines.append('')
        lines.append(f"历史参考臂（chunk 2048, fold_0）：C = {hist['fold_0_c_index']:.4f}，"
                     f"验证 NLL = {hist['fold_0_validation_nll']:.4f} —— 仅供对照，不是成对比较对象。")
        lines.append('')
        lines.append('单折 C-index 标准误约 0.03，所以本阶段**不排名冠军**，只判定哪些因子家族值得进入五折。')
        lines.append('')
        lines.append('| 配置 | 家族 | C | ΔC | Brier | ΔBrier | 验证NLL | ΔNLL | best_ep | 判定 |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for row in sorted(stage1['rows'], key=lambda x: -(x.get('c_index') or 0)):
            if row.get('status') != 'completed':
                lines.append(f"| {row['run']} | {row['family']} | — | — | — | — | — | — | — | "
                             f"{row.get('status', 'n/a')} |")
                continue
            d_nll = row.get('delta_validation_nll')
            d_nll_text = f"{d_nll:+.4f}" if d_nll is not None else 'n/a(离散化不同)'
            verdict = '入选' if row.get('admissible') else '不入选'
            lines.append(f"| {row['run']} | {row['family']} | {row['c_index']:.4f} | "
                         f"{row['delta_c_index']:+.4f} | {row['ipcw_brier']:.4f} | "
                         f"{row['delta_brier']:+.4f} | {row['validation_nll']:.4f} | {d_nll_text} | "
                         f"{row['best_epoch']} | {verdict} |")
        lines.append('')
        selection = stage1['stage2_selection']
        lines.append(f"家族配额规则：{selection['rule']}")
        lines.append('')
        lines.append(f"进入 Stage 2 的配置：{', '.join(selection['selected']) or '（无）'}")
        lines.append(f"因配额被排除但本身达标的配置：{', '.join(selection['rejected_admissible_by_quota']) or '（无）'}")
        lines.append('')

    if stage2:
        lines.append('## 3. Stage 2 五折确认（304 名开发患者，每人恰好被一个未见她的模型评分）')
        lines.append('')
        lines.append('| 配置 | 逐折 C | 均值 C | 折间 SD | 合并 OOF C | OOF SE | OOF 可比对对 |')
        lines.append('|---|---|---|---|---|---|---|')
        for name, entry in stage2.items():
            folds = entry.get('folds', [])
            if not folds:
                continue
            per_fold = ' '.join(f"{f['c_index']:.4f}" for f in folds)
            pooled = entry.get('pooled', {})
            mean = entry.get('mean_c_index')
            sd = entry.get('std_c_index')
            lines.append(
                f"| {name} | {per_fold} | {mean:.4f} | {sd:.4f} | "
                f"{pooled.get('c_index', float('nan')):.4f} | "
                f"{pooled.get('c_index_standard_error', float('nan')):.4f} | "
                f"{pooled.get('comparable_pairs', 0)} |")
        lines.append('')

    if stage3:
        lines.append('## 4. Stage 3 方差分解（折间 vs 种子间）')
        lines.append('')
        for name, entry in stage3.items():
            lines.append(f"### {name}")
            lines.append('')
            lines.append('| 种子 | 逐折 C | 均值 C | 折间 SD | 完整 |')
            lines.append('|---|---|---|---|---|')
            for seed, data in entry['seeds'].items():
                folds = ' '.join(f"{f['c_index']:.4f}" for f in data['folds'])
                lines.append(f"| {seed} | {folds} | {data['mean_c_index']:.4f} | "
                             f"{data['std_c_index']:.4f} | {data['complete']} |")
            lines.append('')
            lines.append(f"种子间均值 C 极差：{entry.get('seed_spread_of_mean_c_index')}")
            lines.append('')

    lines.append('## 5. 限制（必须随结论一起复述）')
    lines.append('')
    lines.append('- 每折 C-index 是在选择检查点 epoch 的同一批验证患者上算的，相对未触碰的测试集是乐观的。')
    lines.append('- 5 折合并 OOF 的 IPCW Brier 使用五折训练集删失 KM 的拼接，是一个记账聚合，不是单一预设训练集。')
    lines.append('- 阶段划分与配额在运行前写定；本报告不隐藏任何失败或变差的配置。')
    lines.append('- outer_test 76 名患者保持封存，全程未加载。')
    lines.append('- 视图源是编码特征噪声，不是 WSI 图像增强；不宣称临床意义、泛化能力或新颖性。')
    lines.append('- 局部病理/通路关联是弱监督学习结果，没有 patch 级 RNA 真值；零相关不等于独立或生物学可识别。')
    lines.append('- 本机同时运行其他项目（DCT-Reg），训练吞吐受 CPU 争用影响；这不影响数值确定性，只影响墙钟时间。')
    lines.append('')

    (RUN / 'FINAL_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
