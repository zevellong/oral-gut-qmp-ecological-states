#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified to include Antibiotic cohorts in Panel A
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
from scipy import stats
from scipy.stats import linregress, spearmanr
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
from pathlib import Path

# Local imports
from proj_plot_main_func import prepare_project_dataset
from proj_config import proj_base_path

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION & STYLE
# ============================================================================
FIG_CONFIG = {
    'oral_def': 'zhucomb',
    'figsize': (16, 7.5), 
    'dpi': 300,
    'colors': {
        'slope_point': '#c0392b',
        'Oral': '#FF6B6B',
        'Gut': '#4ECDC4',
        'Control': '#95a5a6',
        'IBD': '#e67e22',
        'CD': '#d35400',
        'UC': '#c0392b',
    },
    'groups': {                
        'Cancer': '#EF233C', 
        'IBD': '#FFD166', 
        'Neuro': '#7209B7',
        'Metabolic': '#06D6A0',
        'Antibiotic': '#118AB2', 
        'Other': '#8D99AE'
    }
}

DISEASE_GROUP_MAP = {
    'CRC': 'Cancer', 'PDAC': 'Cancer', 'HCC': 'Cancer', 'GC': 'Cancer', 
    'CCA': 'Cancer', 'HNC': 'Cancer', 'BRCA': 'Cancer', 'LC': 'Cancer', 'CC': 'Cancer',
    'IBD': 'IBD', 'CD': 'IBD', 'UC': 'IBD', 
    'AD': 'Neuro', 'ASD': 'Neuro', 'PD': 'Neuro', 'MS': 'Neuro',
    'T2D': 'Metabolic', 'Obesity': 'Metabolic', 'LiverCirrhosis': 'Metabolic', 
    'Antibiotic': 'Antibiotic', # 新增映射
    'PPI': 'Other', 'Control': 'Other'
}

# ============================================================================
# 2. DATA PREPARATION FUNCTIONS
# ============================================================================

def get_project_slopes(vdata2, oral_def):
    """Panel A: Calculate slopes for each individual project, including Antibiotics"""
    stats_list = []
    x_col = f'oral_rmp_{oral_def}'
    y_col = 'load'
    
    # --- 1. 处理非抗生素队列 ---
    non_ab_vdata = vdata2[vdata2['study_label'] != 'Antibiotic']
    for proj_id in non_ab_vdata['proj_id'].unique():
        df = non_ab_vdata[non_ab_vdata['proj_id'] == proj_id].dropna(subset=[x_col, y_col])
        df = df[(df[x_col] > 0) & (df[y_col] > 0)]
        
        if len(df) < 15:
            continue
        
        study = df['study_label'].iloc[0]
        if study == 'Control': # 排除纯对照队列
            continue
        
        slope, intercept, r_value, p_value, std_err = linregress(
            np.log10(df[x_col]), np.log10(df[y_col])
        )
        group = DISEASE_GROUP_MAP.get(study, 'Other')
        stats_list.append({
            'proj_id': proj_id,
            'study_label': study,
            'group': group,
            'slope': slope,
            'slope_stderr': std_err,
            'n_samples': len(df)
        })

    # --- 2. 处理抗生素队列 (纵向) ---
    ab_vdata = vdata2[vdata2['study_label'] == 'Antibiotic']
    if not ab_vdata.empty:
        # 按项目分组，与其他队列保持一致
        for proj_id, df_sub in ab_vdata.groupby('proj_id'):
            # 时间窗口过滤 (-15, 15)，排除 Antibiotic_Type 为 Control 的数据点
            df = df_sub[
                (df_sub['Antibiotic_Type'] != 'Control') &
                (df_sub['Antibiotic_Timepoint'] > -15) & 
                (df_sub['Antibiotic_Timepoint'] < 15)
            ].dropna(subset=[x_col, y_col])
            
            df = df[(df[x_col] > 0) & (df[y_col] > 0)]
            
            if len(df) < 15: 
                continue
            
            slope, intercept, r_value, p_value, std_err = linregress(
                np.log10(df[x_col]), np.log10(df[y_col])
            )
            
            stats_list.append({
                'proj_id': proj_id,
                'study_label': 'Antibiotic',
                'group': 'Antibiotic',
                'slope': slope,
                'slope_stderr': std_err,
                'n_samples': len(df)
            })
    
    return pd.DataFrame(stats_list)

# (其他函数 prepare_hierarchical_data, aggregate_to_level, plot_sunburst 等保持不变)
# [此处省略与原脚本一致的 prepare_hierarchical_data, calculate_cv, aggregate_to_level 函数]

def calculate_cv(df: pd.DataFrame) -> pd.Series:
    means = df.mean()
    stds = df.std()
    cv = stds / means.replace(0, np.nan)
    return cv

def prepare_hierarchical_data(rmp_oral: pd.DataFrame, 
                             rmp_oral_g: pd.DataFrame, 
                             rmp_oral_s: pd.DataFrame,
                             oral_bact_list: list):
    genus_means = rmp_oral_g.mean().sort_values(ascending=False)
    genus_cv = calculate_cv(rmp_oral_g)
    species_means = rmp_oral_s.mean().sort_values(ascending=False)
    species_cv = calculate_cv(rmp_oral_s)
    
    genus_species_mapping = {}
    for col in rmp_oral.columns:
        taxonomy_parts = col.split('|')
        parsed_taxonomy = {}
        for part in taxonomy_parts:
            if '__' in part:
                level_key = part.split('__')[0]
                level_value = part.split('__')[1]
                parsed_taxonomy[level_key] = level_value
        genus_name = f"g__{parsed_taxonomy.get('g', 'unknown')}"
        species_name = f"s__{parsed_taxonomy.get('s', 'unknown')}"
        if genus_name not in genus_species_mapping:
            genus_species_mapping[genus_name] = []
        genus_species_mapping[genus_name].append({
            'species_name': species_name,
            'original_abundance': rmp_oral[col].mean(),
            'original_column': col
        })
    
    hierarchical_data = []
    cumulative_angle = 0
    total_genus_sum = genus_means.sum()
    for genus in genus_means.index:
        if genus not in genus_species_mapping: continue
        genus_name_clean = genus.replace('g__', '')
        genus_proportion = genus_means[genus] / total_genus_sum
        genus_angle = genus_proportion * 2 * np.pi
        species_info_sorted = sorted(genus_species_mapping[genus], key=lambda x: x['original_abundance'], reverse=True)
        species_data = []
        species_cv_data = {}
        species_sources = {}
        for sp_info in species_info_sorted:
            sp_name = sp_info['species_name']
            if sp_name in species_means.index:
                species_data.append((sp_name, species_means[sp_name]))
                species_cv_data[sp_name] = species_cv.get(sp_name, 0)
            else:
                species_data.append((sp_name, sp_info['original_abundance']))
                species_cv_data[sp_name] = 0
            species_sources[sp_name] = 'Oral' if sp_info['original_column'] in oral_bact_list else 'Gut'
        hierarchical_data.append({
            'genus_name': genus_name_clean, 'genus_cv': genus_cv.get(genus, 0),
            'start_angle': cumulative_angle, 'end_angle': cumulative_angle + genus_angle,
            'species': species_data, 'species_cv': species_cv_data, 'species_sources': species_sources
        })
        cumulative_angle += genus_angle
    return hierarchical_data

def aggregate_to_level(df, level_prefix):
    agg_map = {}
    for col in df.columns:
        parts = col.split('|')
        level = next((p for p in parts if level_prefix in p), None)
        if level:
            if level not in agg_map: agg_map[level] = df[col].copy()
            else: agg_map[level] += df[col]
    return pd.DataFrame(agg_map)

# ============================================================================
# 3. PLOTTING FUNCTIONS
# ============================================================================

def plot_panel_a_project_slopes(ax, project_slopes_df):
    """Panel A: Slopes by Cohort"""
    if len(project_slopes_df) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    cohort_counts = project_slopes_df['study_label'].value_counts()
    all_cohorts = sorted(cohort_counts.index)
    cohort_medians = {c: project_slopes_df[project_slopes_df['study_label'] == c]['slope'].median() for c in all_cohorts}
    cohorts_ordered = sorted(all_cohorts, key=lambda x: cohort_medians[x])
    
    for i, cohort in enumerate(cohorts_ordered):
        cohort_data = project_slopes_df[project_slopes_df['study_label'] == cohort]
        slopes = cohort_data['slope'].values
        color = FIG_CONFIG['groups'].get(DISEASE_GROUP_MAP.get(cohort, 'Other'), '#888888')
        pos = i + 1
        
        if len(slopes) >= 2:
            ax.boxplot([slopes], positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=color, alpha=0.4, linewidth=1.5),
                       medianprops=dict(color='darkred', linewidth=2.5))
            x_jitter = np.random.normal(pos, 0.05, size=len(slopes))
            ax.scatter(x_jitter, slopes, alpha=0.8, s=60, color=color, edgecolors='black', linewidth=1, zorder=3)
        else:
            ax.errorbar(pos, slopes[0], yerr=cohort_data['slope_stderr'].values[0] if 'slope_stderr' in cohort_data.columns else 0,
                       fmt='o', markersize=11, capsize=5, color=color, ecolor='gray', 
                       markeredgecolor='black', markeredgewidth=1.2, alpha=0.9, zorder=3)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.3)
    overall_median = project_slopes_df['slope'].median()
    ax.axhline(y=overall_median, color='darkred', linestyle='--', linewidth=2, alpha=0.8, 
               label=f'Global Median: {overall_median:.3f}')
    
    ax.set_xticks(range(1, len(cohorts_ordered) + 1))
    ax.set_xticklabels(cohorts_ordered, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel('Slope ($log_{10}$ Load ~ $log_{10}$ Oral RMP)', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=9, frameon=True)

def plot_sunburst(ax, hierarchical_data):
    # [此处省略与原脚本一致的 plot_sunburst 函数代码]
    all_genus_cv = [item['genus_cv'] for item in hierarchical_data if not np.isnan(item['genus_cv'])]
    all_species_cv = [cv for item in hierarchical_data for cv in item['species_cv'].values() if not np.isnan(cv)]
    if not all_genus_cv or not all_species_cv: return
    cv_min, cv_max = min(all_genus_cv + all_species_cv), max(all_genus_cv + all_species_cv)
    norm = Normalize(vmin=cv_min, vmax=cv_max)
    cmap = plt.cm.RdYlBu_r
    r1, r2 = (0.30, 0.60), (0.63, 0.95) 
    for item in hierarchical_data:
        if not np.isnan(item['genus_cv']):
            theta = np.linspace(item['start_angle'], item['end_angle'], 50)
            ax.fill(np.concatenate([r1[0]*np.cos(theta), r1[1]*np.cos(theta[::-1])]),
                    np.concatenate([r1[0]*np.sin(theta), r1[1]*np.sin(theta[::-1])]),
                    facecolor=cmap(norm(item['genus_cv'])), edgecolor='white', linewidth=0.5)
            if (item['end_angle'] - item['start_angle']) > 0.25:
                mid = (item['start_angle'] + item['end_angle']) / 2
                rad = (r1[0] + r1[1]) / 2
                ax.text(rad*np.cos(mid), rad*np.sin(mid), item['genus_name'][:8], 
                        ha='center', va='center', fontsize=7, color='white', weight='bold',
                        path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])
        genus_width = item['end_angle'] - item['start_angle']
        species_total = sum([ab for _, ab in item['species'] if ab > 0])
        if species_total == 0: continue
        curr_ang = item['start_angle']
        for sp_name, ab in item['species']:
            if ab <= 0: continue
            sp_width = genus_width * (ab / species_total)
            sp_end = curr_ang + sp_width
            theta = np.linspace(curr_ang, sp_end, 10)
            ax.fill(np.concatenate([r2[0]*np.cos(theta), r2[1]*np.cos(theta[::-1])]),
                    np.concatenate([r2[0]*np.sin(theta), r2[1]*np.sin(theta[::-1])]),
                    facecolor=cmap(norm(item['species_cv'].get(sp_name, cv_min))), 
                    edgecolor='white', linewidth=0.3)
            if item['species_sources'].get(sp_name) == 'Oral' and sp_width > 0.005:
                mid_ang = (curr_ang + sp_end) / 2
                ax.plot(0.99*np.cos(mid_ang), 0.99*np.sin(mid_ang), 'o', markersize=2.5, 
                        color='#E63946', markeredgecolor='white', markeredgewidth=0.3)
            curr_ang = sp_end
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect('equal'); ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.05, aspect=20).set_label('CV', weight='bold')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946', markersize=6, label='Oral Species'),
        Patch(facecolor='gray', label='Inner: Genus', alpha=0.6),
        Patch(facecolor='lightgray', label='Outer: Species', alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1.05), fontsize=8, frameon=True)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    REMOVE_TITLES = True
    mp4ver = "mp4v2025"
    fig_ppath = "fig5_AC_with_AB" 
    # mask_projid =  ["PRJEB8094", 'PRJEB38625', "PRJEB21933","PRJNA1111312", 'PRJEB42155',
    #                 "PRJNA493153","PRJNA702617","PRJNA531273","PRJNA692325",
    #                 "PRJNA936589", 'PRJNA851554'] 
    mask_projid =  [] 
    
    mask_dise_labels = ["Adenoma", "CD_surgery", "Indeterminate_colitis", "polyp", "adenoma",
                        "Small adenoma", "Large adenoma"]
    
    cache_path = proj_base_path / "data/cache/antibiotic_analysis_cache_v1.pkl.gz"
    
    print("=== Loading Data ===")
    proj_data = prepare_project_dataset(
        proj_base_path, fig_ppath, mask_projids=mask_projid, 
        cache_path=cache_path, mp4ver=mp4ver, mask_disease_labels=mask_dise_labels
    )
    vdata2 = proj_data['vdata2']
    rmp = proj_data["rmp"]
    oral_bact_dict = proj_data['oral_bact_dict']
    _, _, fig_pre = proj_data['paths']

    print("=== Preparing Data for Panels ===")
    project_slopes_df = get_project_slopes(vdata2, FIG_CONFIG['oral_def'])
    
    top_n = 60
    rmp_all_g = aggregate_to_level(rmp, 'g__')
    top_genera_names = rmp_all_g.mean().sort_values(ascending=False).head(top_n).index.tolist()
    
    cols_top = [c for c in rmp.columns if any(g in c for g in top_genera_names)]
    cols_other = [c for c in rmp.columns if c not in cols_top]
    rmp_top = rmp[cols_top].copy()
    if cols_other:
        rmp_top['k__Bacteria|g__Other|s__Other'] = rmp[cols_other].sum(axis=1)
    
    rmp_top_g = aggregate_to_level(rmp_top, 'g__')
    rmp_top_s = aggregate_to_level(rmp_top, 's__')
    hierarchical_data = prepare_hierarchical_data(rmp_top, rmp_top_g, rmp_top_s, oral_bact_dict[FIG_CONFIG['oral_def']])

    print("=== Plotting ===")
    fig = plt.figure(figsize=FIG_CONFIG['figsize'], dpi=FIG_CONFIG['dpi'])
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.92, 1], wspace=0.05, left=0.06, right=0.96)
    
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a_project_slopes(ax_a, project_slopes_df)
    
    ax_b = fig.add_subplot(gs[0, 1])
    plot_sunburst(ax_b, hierarchical_data)
    
    save_path = f"{fig_pre}/Fig5_A_B_with_Antibiotics.pdf"
    Path(fig_pre).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Main Figure saved to: {save_path}")
    
    # --- 分开保存 Panel A 和 Panel B ---
    # Panel A
    fig_a, ax_a_sep = plt.subplots(figsize=(8, 5), dpi=FIG_CONFIG['dpi'])
    plot_panel_a_project_slopes(ax_a_sep, project_slopes_df)
    plt.savefig(f"{fig_pre}/Fig5_A_Slopes.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig_a)
    print(f"✓ Panel A saved to: {fig_pre}/Fig5_A_Slopes.pdf")
    
    # Panel B
    fig_b, ax_b_sep = plt.subplots(figsize=(8, 8), dpi=FIG_CONFIG['dpi'])
    plot_sunburst(ax_b_sep, hierarchical_data)
    plt.savefig(f"{fig_pre}/Fig5_B_Sunburst.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig_b)
    print(f"✓ Panel B saved to: {fig_pre}/Fig5_B_Sunburst.pdf")
    
    plt.show()