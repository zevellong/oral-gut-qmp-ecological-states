#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 Subplot: Multi-Disease Microbiome Comparison
Layout: 4 rows x n_cols
Clean Subplot Facecolor Edition (16-inch width locked)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ranksums, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from typing import List, Dict, Optional, Tuple

# Local imports
from proj_config import proj_base_path, set_journal_style
from proj_plot_main_func import prepare_project_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

class BasePlotConfig:
    DPI = 300
    SCATTER_ALPHA = 0.7  # 适中的透明度，显色高级
    LINE_WIDTH = 1.0     # 边框精致细线

class FigComp3Config:
    DISEASE_ORDER = ['Control', 'LC', 'PPI', 'UC', 'CD', 'ABX-Exposed']
    DISEASE_COLORS = {
        'Control':              '#7FB3D5',
        'PPI':                  '#F4A582',
        'LC':                   '#B19CD9',
        'UC':                   '#F8B739',
        'CD':                   '#D4AC0D',
        'ABX-Exposed':          '#E57373',
    }
    MECHANISM_COLORS = {
        'Barrier Loss':       '#FEF9E7',  # Sand
        'Gut Depletion':      '#F5EEF8',  # Lavender
        'Systemic Depletion': '#FADBD8',  # Soft Red
    }
    STUDY_LABEL_RENAME = {
        'PPI': 'PPI (Zhu et al., 2024)',
        'LC': 'LC (PRJEB6337)',
        'IBD': 'IBD (PRJNA893901)',
        'Antibiotic': 'Antibiotic (PRJNA664754)',
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_significance_label(pval: float) -> str:
    if pval < 0.001:  return '***'
    elif pval < 0.01: return '**'
    elif pval < 0.05: return '*'
    else:             return 'ns'

def wilcoxon_test(data1: np.ndarray, data2: np.ndarray, test_method: str = 'mannwhitneyu') -> float:
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    if len(data1) < 2 or len(data2) < 2: return np.nan
    try:
        if test_method == 'mannwhitneyu':
            _, pval = mannwhitneyu(data1, data2, alternative='two-sided')
        elif test_method == 'ranksums':
            _, pval = ranksums(data1, data2)
        else:
            _, pval = ttest_ind(data1, data2, equal_var=False)
        return pval
    except Exception:
        return np.nan

def add_significance_bar(ax, x1: float, x2: float, y: float, pval: float, fontsize: int = 8):
    sig_text = get_significance_label(pval)
    if sig_text == 'ns': return
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = y_range * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='#333333', lw=1.0)
    ax.text((x1 + x2) / 2, y + h, sig_text, ha='center', va='bottom', fontsize=fontsize, color='black', fontweight='bold')

def style_axis(ax, labelsize: int = 8):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(BasePlotConfig.LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(BasePlotConfig.LINE_WIDTH)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, width=BasePlotConfig.LINE_WIDTH)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

def log_formatter(x, pos):
    if x == 0:   return '1'
    elif x == 1: return '10'
    elif x == -1:return '0.1'
    else:        return f'$10^{{{int(x)}}}$'

# ============================================================================
# ANTIBIOTIC DATA PREPARATION
# ============================================================================

def prepare_antibiotic_cohorts_merged(
        vdata2: pd.DataFrame, anti_proj_ids: List[str],
        pre_interval: Tuple[int, int] = (-15, 0), post_interval: Tuple[int, int] = (1, 15)
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    if not anti_proj_ids: return pd.DataFrame(), [], {}
    anti_data = vdata2[vdata2['proj_id'].isin(anti_proj_ids)].copy()
    if anti_data.empty: return pd.DataFrame(), [], {}

    def categorize(row):
        a_type = str(row.get('Antibiotic_Type', '')).lower()
        tp = row.get('Antibiotic_Timepoint', np.nan)
        if pd.isna(tp): return None
        if a_type == 'control' or (pre_interval[0] <= tp <= pre_interval[1]): return 'Control'
        if post_interval[0] <= tp <= post_interval[1]: return 'ABX-Exposed'
        return None

    anti_data['Disease_label'] = anti_data.apply(categorize, axis=1)
    anti_data = anti_data[anti_data['Disease_label'].notna()]
    cohort_list = sorted(anti_proj_ids)
    cohort_study_map = {pid: 'Antibiotic' for pid in anti_proj_ids}
    return anti_data, cohort_list, cohort_study_map

# ============================================================================
# CORE PLOT FUNCTION
# ============================================================================

def plot_panel(ax, row_idx: int, disease_data: pd.DataFrame, antibiotic_data: Optional[pd.DataFrame],
               metric_col: str, ylabel: str, disease_cohort_order: List[str], anti_cohort_order: List[str],
               cohort_study_map: Dict[str, str], config, show_study_labels: bool = False,
               test_method: str = 'mannwhitneyu', use_log_scale: bool = False):
    
    mech_map = {
        'PPI':        'Barrier Loss',
        'LC':         'Gut Depletion',
        'IBD':        'Gut Depletion',
        'Antibiotic': 'Systemic Depletion',
    }

    positions, colors, box_data, labels_info = [], [], [], []
    cohort_positions_map = {}

    current_pos = 0
    box_width   = 0.30      
    box_spacing = 0.50      

    all_cohorts_ordered = disease_cohort_order + anti_cohort_order
    all_diseases = disease_data['Disease_label'].unique()
    disease_order = [d for d in config.DISEASE_ORDER if d in all_diseases]

    # ---- disease cohorts ----
    for cohort in disease_cohort_order:
        cohort_data = disease_data[disease_data['proj_id'] == cohort]
        valid_diseases = [d for d in disease_order if len(cohort_data[cohort_data['Disease_label'] == d]) > 0]
        cohort_start = current_pos
        coh_dis_data = {}

        for li, disease in enumerate(valid_diseases):
            raw = cohort_data[cohort_data['Disease_label'] == disease][metric_col].dropna().values
            if use_log_scale:
                disp = raw.copy()
                mp = disp[disp > 0].min() if (disp > 0).any() else 1e-10
                disp[disp <= 0] = mp / 10
                disp = np.log10(disp)
            else:
                disp = raw

            pos = current_pos + li * box_spacing
            positions.append(pos)
            colors.append(config.DISEASE_COLORS.get(disease, '#888888'))
            box_data.append(disp)
            labels_info.append({'cohort': cohort, 'label': disease})
            coh_dis_data[disease] = raw

        cohort_positions_map[cohort] = {'start': cohort_start, 'diseases': valid_diseases, 'disease_data': coh_dis_data}

    # ---- antibiotic cohorts ----
    if antibiotic_data is not None and len(anti_cohort_order) > 0:
        anti_dis_order = ['Control', 'ABX-Exposed']
        for cohort in anti_cohort_order:
            cohort_data = antibiotic_data[antibiotic_data['proj_id'] == cohort]
            if cohort_data.empty: continue
            avail_labels = [lb for lb in anti_dis_order if lb in cohort_data['Disease_label'].values]
            cohort_start = current_pos
            coh_dis_data = {}

            for li, label in enumerate(avail_labels):
                raw = cohort_data[cohort_data['Disease_label'] == label][metric_col].dropna().values
                if len(raw) == 0: continue
                if use_log_scale:
                    disp = raw.copy()
                    mp = disp[disp > 0].min() if (disp > 0).any() else 1e-10
                    disp[disp <= 0] = mp / 10
                    disp = np.log10(disp)
                else:
                    disp = raw

                pos = current_pos + li * box_spacing
                positions.append(pos)
                colors.append(config.DISEASE_COLORS.get(label, '#888888'))
                box_data.append(disp)
                labels_info.append({'cohort': cohort, 'label': label})
                coh_dis_data[label] = raw

            cohort_positions_map[cohort] = {'start': cohort_start, 'diseases': list(coh_dis_data.keys()), 'disease_data': coh_dis_data}

    if not box_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    # ---- draw boxplots ----
    bp = ax.boxplot(
        box_data, positions=positions, widths=box_width, patch_artist=True, showfliers=True,
        medianprops=dict(linewidth=1.5, color='#1A1A1A', solid_capstyle='butt'),
        flierprops=dict(marker='o', markersize=2.5, alpha=0.4, markeredgecolor='none', markerfacecolor='#555555'),
        boxprops=dict(linewidth=1.0, edgecolor='#1A1A1A'),
        whiskerprops=dict(linewidth=1.0, color='#1A1A1A'),
        capprops=dict(linewidth=1.0, color='#1A1A1A'),
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(BasePlotConfig.SCATTER_ALPHA)

    # ---- significance bars ----
    all_vals = np.concatenate([d for d in box_data if len(d) > 0])
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    y_range = y_max - y_min if y_max != y_min else 1.0

    max_comp = 0
    for cohort in all_cohorts_ordered:
        if cohort not in cohort_positions_map: continue
        info = cohort_positions_map[cohort]
        valid_diseases = info['diseases']
        if 'Control' not in valid_diseases: continue

        ctrl_data = info['disease_data']['Control']
        ctrl_idx = valid_diseases.index('Control')
        ctrl_pos = info['start'] + ctrl_idx * box_spacing

        comp_count = 0
        for disease in valid_diseases:
            if disease == 'Control': continue
            dis_data = info['disease_data'][disease]
            dis_idx = valid_diseases.index(disease)
            dis_pos = info['start'] + dis_idx * box_spacing

            pval = wilcoxon_test(ctrl_data, dis_data, test_method)
            if not np.isnan(pval) and pval < 0.05:
                y_off = y_max + y_range * (0.05 + 0.15 * comp_count)
                add_significance_bar(ax, ctrl_pos, dis_pos, y_off, pval, fontsize=9)
                comp_count += 1
        max_comp = max(max_comp, comp_count)

    top_buf = 0.05 + 0.18 * max(max_comp, 1)
    ax.set_ylim(y_min - y_range * 0.04, y_max + y_range * top_buf)

    # ---- x-tick labels ----
    xtick_labs = []
    for info in labels_info:
        lbl = info['label']
        if info['cohort'] in anti_cohort_order:
            if lbl == 'Control': xtick_labs.append('Pre-ABX')
            elif lbl == 'ABX-Exposed': xtick_labs.append('Post-ABX')
            else: xtick_labs.append(lbl)
        else:
            if lbl == 'Control': xtick_labs.append('HC')
            else: xtick_labs.append(lbl)

    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_labs, fontsize=10, fontweight='bold')

    if use_log_scale: ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', labelpad=6)
    style_axis(ax, labelsize=10)

    if show_study_labels and all_cohorts_ordered:
        study = cohort_study_map.get(all_cohorts_ordered[0], '')
        label_text = config.STUDY_LABEL_RENAME.get(study, study)
        ax.set_title(label_text, fontsize=12, fontweight='bold', pad=12)

    # ---- 👑 完美的独立子图背景填充 (The Natural Subplot Fill) ----
    if all_cohorts_ordered:
        cohort = all_cohorts_ordered[0]
        study = cohort_study_map.get(cohort, '')
        mech = mech_map.get(study, None)
        if mech:
            color = config.MECHANISM_COLORS.get(mech, '#FFFFFF')
            # 完整填充子图本身的背景区域，不溢出边缘，不干扰 Y 轴标签
            ax.set_facecolor(mcolors.to_rgba(color, alpha=0.5))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    set_journal_style()

    mp4ver      = "mp4v2025"
    fig_ppath   = "fig1"
    oral_def    = "zhucomb"
    test_method = 'mannwhitneyu'

    mask_projid = []
    mask_dise_labels = ["Adenoma", "CD_surgery", "Indeterminate_colitis", "polyp", "adenoma", "Small adenoma", "Large adenoma"]
    cache_path = proj_base_path / "data/cache/antibiotic_analysis_cache_v1.pkl.gz"

    print("=== Loading Data ===")
    proj_data = prepare_project_dataset(proj_base_path, fig_ppath, mask_projids=mask_projid, 
                                        cache_path=cache_path, mp4ver=mp4ver, mask_disease_labels=mask_dise_labels)
    vdata2 = proj_data['vdata2']
    _, _, fig_pre = proj_data['paths']

    disease_cohort_selection = {"PPI": ['PPI_01'], "LC": ['PRJEB6337'], "IBD": ['PRJNA893901']}
    anti_proj_ids = ["PRJNA664754"]
    
    disease_proj_ids = [pid for cohorts in disease_cohort_selection.values() for pid in cohorts]
    vdata2_disease = vdata2[vdata2['proj_id'].isin(disease_proj_ids)].copy()

    disease_cohort_order, cohort_study_map = [], {}
    for study, cohorts in disease_cohort_selection.items():
        for c in cohorts:
            disease_cohort_order.append(c)
            cohort_study_map[c] = study

    anti_data, anti_cohort_order, anti_map = prepare_antibiotic_cohorts_merged(vdata2, anti_proj_ids)
    cohort_study_map.update(anti_map)

    metrics = [
        (f'oral_qmp_{oral_def}', 'Oral QMP', True),
        (f'gut_qmp_{oral_def}',  'Gut QMP',  True),
        (f'oral_rmp_{oral_def}', 'Oral RMP', False),
        (f'gut_rmp_{oral_def}',  'Gut RMP',  False),
    ]

    all_cohorts_ordered = disease_cohort_order + anti_cohort_order
    n_cols = len(all_cohorts_ordered)

    w_ratios = [2.5 if c == 'PRJNA893901' else 2.0 for c in all_cohorts_ordered]

    # 👑 画布尺寸：恢复为大气的 16x6.5
    fig, axes = plt.subplots(4, n_cols, figsize=(14.16, 6), dpi=300, squeeze=False, gridspec_kw={'width_ratios': w_ratios})

    for row_idx, (metric_col, ylabel, use_log) in enumerate(metrics):
        for col_idx, cohort in enumerate(all_cohorts_ordered):
            ax = axes[row_idx, col_idx]
            show_labels = (row_idx == 0)

            if cohort in disease_cohort_order:
                cohort_dis_data = vdata2_disease[vdata2_disease['proj_id'] == cohort].copy()
                if cohort == 'PRJNA893901':
                    def map_ibd(row):
                        if row.get('Disease_label') == 'Control': return 'Control'
                        status = str(row.get('Disease_status', '')).upper()
                        if 'UC' in status: return 'UC'
                        if 'CD' in status: return 'CD'
                        return 'IBD'
                    cohort_dis_data['Disease_label'] = cohort_dis_data.apply(map_ibd, axis=1)
                cohort_anti_data, do_dis, do_anti = None, [cohort], []
            else:
                cohort_dis_data = vdata2_disease.iloc[0:0]
                cohort_anti_data = anti_data[anti_data['proj_id'] == cohort]
                do_dis, do_anti = [], [cohort]

            plot_panel(ax, row_idx, cohort_dis_data, cohort_anti_data, metric_col, ylabel if col_idx == 0 else "",
                       do_dis, do_anti, cohort_study_map, FigComp3Config, show_labels, test_method, use_log)

            if row_idx < 3:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('')

            # 动态调整 margin 防止箱体太宽
            if len(ax.get_xticks()) > 0:
                x_min, x_max = min(ax.get_xticks()), max(ax.get_xticks())
                margin = 0.6 
                ax.set_xlim(x_min - margin, x_max + margin)

    # 👑 完美的物理隔离间距：wspace=0.35 保证了各个列之间有充裕的空间，Y 轴数字再也不会碰到背景色
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08, hspace=0.30, wspace=0.35)

    save_path = f"{fig_pre}/fig1_data_panel_16inch.pdf"
    plt.savefig(save_path, dpi=300, facecolor='white')
    print(f"\n✅ Final Natural Fig 1 saved: {save_path}")
    plt.close()