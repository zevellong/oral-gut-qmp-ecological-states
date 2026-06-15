#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transposed Heatmap V20 - Reordered Blocks (RMP -> QMP -> Total Load)
Separated blocks reordered from top to bottom according to user request.
Width: 14.25 inches (~1026 pt)
Height: 4.8 inches 

@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

from proj_config import proj_base_path, set_journal_style
from proj_plot_main_func import prepare_project_dataset

set_journal_style()

# ============================================================================
# Config & Cohort Selection
# ============================================================================
mp4ver = "mp4v2025"
fig_ppath = "fig4"
mask_projid =  [] 
mask_dise_labels = ["Adenoma", "CD_surgery","Indeterminate_colitis","polyp"]

COHORT_SELECTION = {
    "Cancer": {
        "PDAC": ["PRJNA832909", "PRJEB42013", "PRJNA665854","PRJEB38625"],
        "CRC": ['PRJDB4176', 'PRJEB10878', 'PRJEB27928', 'PRJEB53891', 'PRJEB6070', 
                    'PRJNA1050885', 'PRJNA1138893', 'PRJNA1167935', 'PRJNA389927', 'PRJNA429097', 
                    'PRJNA447983', 'PRJNA731589', 'PRJNA763023', 'PRJNA888860', 'PRJNA961076'],
        "CCA": ["PRJNA932948"], 
        "BRCA": ["PRJNA453965"],
    },
    "IBD": {
        "IBD": [ 'PRJEB76677', 'PRJNA385949', 'PRJNA511372', 'PRJNA793776', 'PRJNA893901']
    },
    "Neurological": {
        "PD": ['PRJNA433459', 'PRJEB53401', 'PRJNA762484', 'PRJNA834801', 'PRJEB53403'],
        "ASD": ['PRJNA1037036', 'PRJNA686821', "PRJEB23052"],
        "AD": ['PRJNA665245'],
    },
    "Other": {
        "PPI": ['PPI_01'],
        "LC": ['PRJEB6337'],
    }
}
    
GROUP_COLORS = {
    'Neurological': '#00BBF9',     
    'Cancer': '#F15BB5',           
    'CRC': '#FEE440',              
    'IBD': '#00F5D4',              
    'Other': '#9B5DE5'             
}

# ============================================================================
# 1. DATA PREPARATION 
# ============================================================================
def prepare_ordered_heatmap_data(vdata2, rmp, qmp, proj_cache_path, oral_def="zhucomb", sort_by_metric=None):
    rows = []
    metrics = [
        ('Total_QMP', 'Total Load QMP'),  
        (f'oral_qmp_{oral_def}', 'Oral QMP'),
        (f'gut_qmp_{oral_def}', 'Gut QMP'),
        (f'oral_rmp_{oral_def}', 'Oral RMP'),
        (f'gut_rmp_{oral_def}', 'Gut RMP'),
    ]
    
    for group, diseases in COHORT_SELECTION.items():
        for disease_type, projects in diseases.items():
            disease_rows = [] 
            
            def process_subset(subset_df, ctrl_df, cohort_label, real_label, dis_type_label):
                if len(subset_df) < 3: return
                row_data = {
                    'Group': group, 'Disease_Type': dis_type_label,
                    'Cohort': cohort_label, 'Real_Disease_Label': real_label
                }
                
                for col, name in metrics:
                    if name == 'Total Load QMP':
                        subset_vals = qmp.loc[subset_df.index].sum(axis=1)
                        ctrl_vals = qmp.loc[ctrl_df.index].sum(axis=1)
                    else:
                        subset_vals = subset_df[col].dropna()
                        ctrl_vals = ctrl_df[col].dropna()
                    
                    try:
                        _, p = mannwhitneyu(subset_vals, ctrl_vals, alternative='two-sided')
                    except: p = 1.0
                    
                    direction = 1 if subset_vals.median() > ctrl_vals.median() else -1
                    score = -np.log10(max(p, 1e-100)) * direction
                    row_data[name] = score
                    row_data[f"{name}_p"] = p
                
                disease_rows.append(row_data)

            for proj in projects:
                if proj not in vdata2['proj_id'].values: continue
                proj_data = vdata2[vdata2['proj_id'] == proj]
                if 'Control' not in proj_data['Disease_label'].values: continue
                ctrl = proj_data[proj_data['Disease_label'] == 'Control']
                
                if proj == "PRJNA932948" and disease_type == "CCA":
                    avail = [d for d in proj_data['Disease_label'].unique() if d != 'Control']
                    for target in ['HCC', 'CCA']:
                        if target in avail:
                            dis = proj_data[proj_data['Disease_label'] == target]
                            process_subset(dis, ctrl, f"{proj}_{target}", target, target)
                    continue
                
                if disease_type == "IBD":
                    has_disease_status = False
                    if 'Disease_status' in proj_data.columns:
                        avail_statuses = [s for s in proj_data['Disease_status'].unique() if s in ['UC', 'CD']]
                        if avail_statuses:
                            has_disease_status = True
                            for status in avail_statuses:
                                dis = proj_data[proj_data['Disease_status'] == status]
                                process_subset(dis, ctrl, f"{proj}_{status}", status, status)
                    
                    if not has_disease_status:
                        avail = [d for d in proj_data['Disease_label'].unique() if d != 'Control']
                        subtypes = [d for d in avail if d in ['UC', 'CD', 'IBD']]
                        if subtypes:
                            for sub in subtypes:
                                dis = proj_data[proj_data['Disease_label'] == sub]
                                lbl = sub if sub in ['UC', 'CD'] else 'IBD'
                                process_subset(dis, ctrl, f"{proj}" if sub == 'IBD' else f"{proj}_{sub}", sub, lbl)
                        else:
                            if avail:
                                dis = proj_data[proj_data['Disease_label'] == avail[0]]
                                process_subset(dis, ctrl, proj, avail[0], 'IBD')
                    continue
                
                avail = [d for d in proj_data['Disease_label'].unique() if d != 'Control']
                if not avail: continue
                dis = proj_data[proj_data['Disease_label'] == avail[0]]
                process_subset(dis, ctrl, proj, avail[0], disease_type)
            
            if disease_type == "IBD" and sort_by_metric and sort_by_metric in [m[1] for m in metrics]:
                from itertools import groupby
                disease_rows.sort(key=lambda x: x['Disease_Type'])
                sorted_rows = []
                for subtype, group in groupby(disease_rows, key=lambda x: x['Disease_Type']):
                    subtype_rows = list(group)
                    subtype_rows.sort(key=lambda x: x[sort_by_metric], reverse=True)
                    sorted_rows.extend(subtype_rows)
                disease_rows = sorted_rows
            elif sort_by_metric and sort_by_metric in [m[1] for m in metrics] and disease_rows:
                disease_rows.sort(key=lambda x: x[sort_by_metric], reverse=True)
            
            rows.extend(disease_rows)
    
    df = pd.DataFrame(rows)
    
    metric_cols = [m[1] for m in metrics]
    plot_df_stats = df[metric_cols]
    
    p_df = df[[m + '_p' for m in metric_cols]]
    p_df.columns = metric_cols
    annot_df = p_df.map(lambda x: '***' if x<0.001 else ('**' if x<0.01 else ('*' if x<0.05 else '')))
    
    row_meta = df[['Group', 'Disease_Type', 'Cohort', 'Real_Disease_Label']]
    
    return plot_df_stats, annot_df, row_meta


# ============================================================================
# 2. PLOTTING FUNCTION (V20: RMP -> QMP -> Total Load)
# ============================================================================
def plot_hierarchical_heatmap_transposed(plot_df, annot_df, row_meta, save_path=None):
    n_cols = len(plot_df)
    
    fig_width = 14.25  
    fig_height = 4.8   
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    width_ratios = [0.055, 0.945]
    
    # 🌟 重新排列高度顺序：Group, Disease, RMP(1.2), Gap, QMP(1.2), Gap, TotalLoad(0.6), Bottom(2.2)
    height_ratios = [0.2, 0.2, 1.2, 0.04, 1.2, 0.04, 0.6, 2.2] 
    
    gs = gridspec.GridSpec(8, 2, 
                           width_ratios=width_ratios, 
                           height_ratios=height_ratios, 
                           wspace=0.01, hspace=0.04)
    
    ax_group = fig.add_subplot(gs[0, 1])
    ax_disease = fig.add_subplot(gs[1, 1])
    
    # RMP 移到了最上面（数据区第一块）
    ax_rmp = fig.add_subplot(gs[2, 1])
    ax_rmp_lbl = fig.add_subplot(gs[2, 0])
    
    # QMP 在中间
    ax_qmp = fig.add_subplot(gs[4, 1])
    ax_qmp_lbl = fig.add_subplot(gs[4, 0])
    
    # Total Load 移到了最下面
    ax_tot = fig.add_subplot(gs[6, 1])
    ax_tot_lbl = fig.add_subplot(gs[6, 0])
    
    ax_cbar = fig.add_subplot(gs[7, 1]) 
    
    for ax in [ax_group, ax_disease, ax_tot_lbl, ax_qmp_lbl, ax_rmp_lbl, ax_cbar]:
        ax.set_axis_off()

    # ── 数据切片与转置 ──
    df_rmp = plot_df[['Oral RMP', 'Gut RMP']].T
    df_qmp = plot_df[['Oral QMP', 'Gut QMP']].T
    df_tot = plot_df[['Total Load QMP']].T
    
    ann_rmp = annot_df[['Oral RMP', 'Gut RMP']].T
    ann_qmp = annot_df[['Oral QMP', 'Gut QMP']].T
    ann_tot = annot_df[['Total Load QMP']].T
    
    cmap_stats = sns.diverging_palette(240, 10, as_cmap=True)
    
    # ── 分块绘制热图 ──
    sns.heatmap(df_rmp, ax=ax_rmp, cmap=cmap_stats, center=0, vmin=-5, vmax=5,
                annot=ann_rmp, fmt='', annot_kws={"size": 9, "va": "center", "weight": "bold"},
                cbar=False, yticklabels=False, xticklabels=False,
                linewidths=0.5, linecolor='white')
                
    sns.heatmap(df_qmp, ax=ax_qmp, cmap=cmap_stats, center=0, vmin=-5, vmax=5,
                annot=ann_qmp, fmt='', annot_kws={"size": 9, "va": "center", "weight": "bold"},
                cbar=False, yticklabels=False, xticklabels=False,
                linewidths=0.5, linecolor='white')
    
    # 最底部的 Total Load 显示 xticklabels=True，为下方的坐标系绑定刻度
    sns.heatmap(df_tot, ax=ax_tot, cmap=cmap_stats, center=0, vmin=-5, vmax=5,
                annot=ann_tot, fmt='', annot_kws={"size": 9, "va": "center", "weight": "bold"},
                cbar=False, yticklabels=False, xticklabels=True,
                linewidths=0.5, linecolor='white')
    
    # ── 顶部 Annotation Bars ──
    group_rgb = np.array([mcolors.to_rgb(GROUP_COLORS.get(g, '#888888')) for g in row_meta['Group']])
    ax_group.imshow(group_rgb.reshape(1, n_cols, 3), aspect='auto', extent=[0, n_cols, 0, 1])
    
    disease_rgb = []
    current_grp = None; current_dis = None; shade_toggle = False
    
    for g, d in zip(row_meta['Group'], row_meta['Disease_Type']):
        if g != current_grp: shade_toggle = False; current_grp = g
        if d != current_dis: shade_toggle = not shade_toggle; current_dis = d
        base_c = mcolors.to_rgb(GROUP_COLORS.get(g, '#888888'))
        factor = 0.45 if shade_toggle else 0.20
        disease_rgb.append(tuple(c * factor + 1.0 * (1 - factor) for c in base_c))
        
    ax_disease.imshow(np.array(disease_rgb).reshape(1, n_cols, 3), aspect='auto', extent=[0, n_cols, 0, 1])
    
    # 分割线与文字 
    current = 0
    from itertools import groupby
    for g, grp in groupby(row_meta['Group']):
        count = len(list(grp))
        center = current + count / 2.0
        
        ax_group.text(center, 0.5, g, ha='center', va='center', fontweight='bold', fontsize=9.5, color='white',
                      path_effects=[pe.withStroke(linewidth=1.0, foreground='#444')])
        
        if current + count < n_cols:
            line_x = current + count
            ax_group.axvline(line_x, color='white', linewidth=2.5)
            ax_disease.axvline(line_x, color='white', linewidth=2.5)
            ax_rmp.axvline(line_x, color='white', linewidth=2.5)
            ax_qmp.axvline(line_x, color='white', linewidth=2.5)
            ax_tot.axvline(line_x, color='white', linewidth=2.5)
            
        current += count
        
    current = 0
    vals = list(zip(row_meta['Group'], row_meta['Disease_Type']))
    for k, grp in groupby(vals):
        count = len(list(grp))
        center = current + count / 2.0
        
        if count >= 1:
            ax_disease.text(center, 0.5, k[1], ha='center', va='center', 
                            fontweight='bold', fontsize=6.0, color='#111', rotation=0)
            
        if current + count < n_cols:
            line_x = current + count
            ax_disease.axvline(line_x, color='white', linewidth=1.0)
        current += count

    # ── Cohort X-Axis Labels (现在绑定到位于最下方的 ax_tot 上) ──
    DISPLAY_NAMES = {'PPI_01': 'PPI (Zhu et al., 2024)'}
    pretty_labels = [DISPLAY_NAMES.get(c, c) for c in row_meta['Cohort']]
    
    ax_tot.set_xticks(np.arange(n_cols) + 0.5)
    ax_tot.set_xticklabels(pretty_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=8.5)
    ax_tot.tick_params(bottom=False)
    
    # ── 4. 独立的三段式左侧标签排版 ──
    
    # Block 1: RMP 标签 
    ax_rmp_lbl.set_xlim(0, 1)
    ax_rmp_lbl.set_ylim(2, 0)
    ax_rmp_lbl.text(0.15, 1.0, 'RMP', ha='center', va='center', rotation=90, fontweight='bold', fontsize=12, color='#111')
    ax_rmp_lbl.text(0.95, 0.5, 'Oral', ha='right', va='center', fontweight='bold', fontsize=9.5, color='#444')
    ax_rmp_lbl.text(0.95, 1.5, 'Gut', ha='right', va='center', fontweight='bold', fontsize=9.5, color='#444')

    # Block 2: QMP 标签
    ax_qmp_lbl.set_xlim(0, 1)
    ax_qmp_lbl.set_ylim(2, 0) 
    ax_qmp_lbl.text(0.15, 1.0, 'QMP', ha='center', va='center', rotation=90, fontweight='bold', fontsize=12, color='#111')
    ax_qmp_lbl.text(0.95, 0.5, 'Oral', ha='right', va='center', fontweight='bold', fontsize=9.5, color='#444')
    ax_qmp_lbl.text(0.95, 1.5, 'Gut', ha='right', va='center', fontweight='bold', fontsize=9.5, color='#444')
    
    # Block 3: Total Load 标签 
    ax_tot_lbl.set_xlim(0, 1)
    ax_tot_lbl.set_ylim(1, 0)
    ax_tot_lbl.text(0.95, 0.5, 'Total\nLoad', ha='right', va='center', fontweight='bold', fontsize=9.0, color='#444')

    # ── Colorbar & Legend ──
    cbar_ax = ax_cbar.inset_axes([0.30, 0.05, 0.25, 0.15]) 
    
    norm = Normalize(vmin=-5, vmax=5)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_stats)
    cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Signed -Log10(P-value)', fontsize=9.5, fontweight='bold', labelpad=2)
    cbar.ax.tick_params(labelsize=8)
    
    cbar_ax.text(0, 1.25, 'Depleted', ha='left', va='bottom', fontsize=9, fontweight='bold', transform=cbar_ax.transAxes)
    cbar_ax.text(1, 1.25, 'Enriched', ha='right', va='bottom', fontsize=9, fontweight='bold', transform=cbar_ax.transAxes)
    
    legend_text = "* p < 0.05      ** p < 0.01      *** p < 0.001"
    ax_cbar.text(0.62, 0.12, legend_text, ha='left', va='center', fontsize=9, fontweight='bold', color='#333')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Panel A Reordered Heatmap saved to: {save_path}")
    return fig

if __name__ == "__main__":
    print("Generating Reordered Heatmap (V20: RMP -> QMP -> Total Load)...")
    
    proj_data = prepare_project_dataset(proj_base_path, fig_ppath, mask_projids=mask_projid, 
                                    mp4ver=mp4ver, mask_disease_labels=mask_dise_labels,
                                    use_v2=False,v2_strategy='oral_priority')
    
    vdata2 = proj_data['vdata2']
    rmp = proj_data['rmp']
    qmp = proj_data['qmp']
    proj_cache_path = proj_data['paths'][1]
    fig_pre = proj_data['paths'][2]
    
    plot_df, annot_df, row_meta = prepare_ordered_heatmap_data(
        vdata2, rmp, qmp, proj_cache_path, "ehomd", sort_by_metric="Oral QMP"
    )
    
    save_path = fig_pre / "fig2_panelA_transposed_v20.pdf"
    
    if not os.path.exists(fig_pre):
        os.makedirs(fig_pre)
        
    plot_hierarchical_heatmap_transposed(plot_df, annot_df, row_meta, save_path)
    print("Done!")