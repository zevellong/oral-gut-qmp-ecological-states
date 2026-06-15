#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Antibiotic Landscape & Aggregated Boxplots (4-Stage Analysis)
Features:
1. Unified Classification Map (Scatter) & Trajectories (Line)
2. Metric: Cohen's d (replacing Hedges' g)
3. Flexible Time Stages: Parameterized n, m, and stage count (3 or 4)
4. Aggregated Boxplots by Antibiotic Category
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from scipy.stats import mannwhitneyu, wilcoxon
from pathlib import Path
import os
import warnings

# Local imports
from proj_config import proj_base_path, set_journal_style, PANEL_LABEL_STYLE
from proj_plot_main_func import prepare_project_dataset
from matplotlib.ticker import SymmetricalLogLocator, LogFormatterSciNotation, FuncFormatter

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class PlotConfig:
    DPI = 300
    PALETTE = {
        'Control': '#2C3E50',          # Black/Dark Grey
        'Broad-Spectrum': '#D9534F',   # Red
        'Gut-Targeting': '#4A90E2',    # Blue
        'Oral-Targeting': '#50E3C2',   # Teal/Green
        'Mild-Impact': '#9B9B9B'       # Light Grey
    }
    
    METRIC_LABEL = "Cohen's d"  # Set to "Cohen's d" or "Hedges' g"
    
    # Medication intervals for specific projects [start, end]
    ABX_INTERVALS = {
        'PRJEB20800': (1, 4), 
        'PRJEB28058': (1, 6), 
        'PRJNA588313': (1, 2), 
        'PRJNA664754': (1, 5)
    }
    
    # --- STAGE PARAMETERS ---
    EXTEND_N = 2      # n: days to extend "During" phase (Matched with micro-view)
    FOLLOWUP_M = 7    # m: days for "Short-term Recovery" phase
    # MIXED STAGES: Gut-Targeting uses 3 stages, others use 4.
    
    # ------------------------

    ORAL_DEF = "ehomd"
    BIOMASS_COLS = {'Oral': f'oral_qmp_{ORAL_DEF}', 'Gut': f'gut_qmp_{ORAL_DEF}'}
    DIV_COLS = {'Oral': f'oral_shannon_{ORAL_DEF}', 'Gut': f'gut_shannon_{ORAL_DEF}'}
    
    # Trajectory Styles
    STYLE_BIOMASS = {'ls': '-', 'lw': 3.0, 'alpha': 0.9} 
    STYLE_DIV = {'ls': '--', 'lw': 2.5, 'alpha': 0.8}

    @staticmethod
    def get_stage_order(category):
        if category == 'Gut-Targeting':
            return ['Before', 'During (Ext)', 'Recovery']
        return ['Before', 'During (Ext)', 'Recovery (S)', 'Recovery (L)']

    @staticmethod
    def get_stage_palette(category):
        # Dynamic palette: "During" uses the orange color
        # Recovers use consistent grays/yellows
        return {
            'Before': '#BDC3C7', 
            'During': '#E67E22', 
            'Recovery': '#95A5A6', 
            'Recovery (S)': '#F1C40F', 
            'Recovery (L)': '#95A5A6'
        }

config = PlotConfig()
def set_pltrc(obj):
    """恢复原版清晰、纤细的高级质感 (1x标准)，适配 15x15 画板"""
    obj.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.0,     # 恢复精致细线
        'patch.linewidth': 1.0,
        'lines.linewidth': 1.5     # 恢复折线图正常粗细
    })


# ============================================================================
# 2. CORE CALCULATIONS
# ============================================================================

def calc_effect_size(x, y, metric="cohen_d"):
    """Calculates Cohen's d or Hedges' g."""
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2: return np.nan
    
    n1, n2 = len(x), len(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if s_pooled == 0: return 0.0
    
    d = (np.mean(x) - np.mean(y)) / s_pooled
    
    if metric == "hedges_g":
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        return d * correction
    return d

def get_baseline(df_c, col):
    """Gets baseline values for a cohort/column."""
    ctrl = df_c[df_c['Antibiotic_Type'].astype(str).str.lower().isin(['control', 'none'])]
    vals = ctrl[col].dropna().values
    if len(vals) < 3: 
        vals = df_c[df_c['Antibiotic_Timepoint'] <= 0][col].dropna().values
    return vals

set_journal_style()
# ============================================================================
# 3. PROCESSING LOGIC
# ============================================================================
def process_unified_data(vdata):
    """
    Processes data for Trajectories and Classification Map.
    Reused from anti3 but updated with Cohen's d.
    """
    df = vdata[vdata["study_label"] == "Antibiotic"].copy()
    # Filter by time window ([-25, 30]) as requested to include baseline while excluding long-term
    df = df[(df['Antibiotic_Timepoint'] >= -25) & (df['Antibiotic_Timepoint'] <= 30)]
    print(df["Antibiotic_Timepoint"].unique())
    
    traj_rows = []
    print(f"Calculating trajectories using {config.METRIC_LABEL}...")
    
    # 1. Trajectories
    for cohort in df['proj_id'].unique():
        df_c = df[df['proj_id'] == cohort].copy()
        for site, col in config.BIOMASS_COLS.items():
            if col not in df_c.columns: continue
            df_c['Log_Val'] = np.log10(df_c[col].replace(0, 1e-9))
            base_vals = get_baseline(df_c, 'Log_Val')
            if len(base_vals) < 3: continue
            
            for abx in df_c['Antibiotic_Type'].unique():
                if str(abx).lower() in ['nan', 'none']: continue
                df_abx = df_c[df_c['Antibiotic_Type'] == abx]
                for t in sorted(df_abx['Antibiotic_Timepoint'].unique()):
                    vals = df_abx[df_abx['Antibiotic_Timepoint'] == t]['Log_Val'].dropna().values
                    if len(vals) >= 2:
                        es = calc_effect_size(vals, base_vals, metric="cohen_d" if "Cohen" in config.METRIC_LABEL else "hedges_g")
                        traj_rows.append({'Antibiotic': abx, 'Site': site, 'Timepoint': t, 'Value': es, 
                                          'Metric_Type': f'Biomass ({config.METRIC_LABEL})', 'Cohort': cohort})

    traj_df = pd.DataFrame(traj_rows)
    if traj_df.empty: return None, None

    # 2. Classification
    class_map = {}
    thresh = -1.0  # 🎯 修改点：将分类阈值同步修改为 -1.0
    for abx in traj_df['Antibiotic'].unique():
        if str(abx).lower() == 'control':
            class_map[abx] = {'Category': 'Control', 'Oral_Peak': 0.0, 'Gut_Peak': 0.0}
            continue
        sub = traj_df[(traj_df['Antibiotic'] == abx) & (traj_df['Timepoint'].between(0, 15))]
        oral_peak = sub[sub['Site'] == 'Oral']['Value'].min() if not sub.empty else 0
        gut_peak = sub[sub['Site'] == 'Gut']['Value'].min() if not sub.empty else 0
        
        if pd.isna(oral_peak): oral_peak = 0
        if pd.isna(gut_peak): gut_peak = 0
        
        if oral_peak < thresh and gut_peak < thresh: cat = 'Broad-Spectrum'
        elif gut_peak < thresh: cat = 'Gut-Targeting'
        elif oral_peak < thresh: cat = 'Oral-Targeting'
        else: cat = 'Mild-Impact'
        class_map[abx] = {'Category': cat, 'Oral_Peak': oral_peak, 'Gut_Peak': gut_peak}

    traj_df['Category'] = traj_df['Antibiotic'].map(lambda x: class_map.get(x, {}).get('Category', 'Mild-Impact'))

    # 3. Diversity
    div_rows = []
    for _, row in df.iterrows():
        abx = row['Antibiotic_Type']
        if str(abx).lower() in ['nan', 'none']: continue
        cat = class_map.get(abx, {}).get('Category', 'Mild-Impact')
        for site, col in config.DIV_COLS.items():
            if col in row and pd.notna(row[col]):
                div_rows.append({'Antibiotic': abx, 'Site': site, 'Timepoint': row['Antibiotic_Timepoint'], 
                                 'Value': row[col], 'Metric_Type': 'Diversity (Shannon)', 'Category': cat, 'Cohort': row['proj_id']})
    
    full_df = pd.concat([traj_df, pd.DataFrame(div_rows)], ignore_index=True)
    class_df = pd.DataFrame([{'Antibiotic': k, **v} for k, v in class_map.items()])
    
    return full_df, class_df


def classify_timepoint_mixed(row, abx_cat_map):
    """Category-aware time stage classification."""
    proj = row['proj_id']
    t = row['Antibiotic_Timepoint']
    abx = row['Antibiotic_Type']
    cat = abx_cat_map.get(abx)
    
    if proj not in config.ABX_INTERVALS or not cat: return None
    start, end = config.ABX_INTERVALS[proj]
    
    # 1. Before
    if t < start: return 'Before'
    
    # 2. During (Extended)
    during_end = end + config.EXTEND_N
    if start <= t <= during_end: return 'During'
    
    # 3. Recovery Phases (Unified Logic)
    # Generic logic for all: Split into Short (<=7d after) and Long (>7d after)
    # If a category only has S, plotting logic can rename it to "Recovery"
    if t > during_end:
        recovery_s_end = during_end + config.FOLLOWUP_M
        if t <= recovery_s_end: return 'Recovery (S)'
        return 'Recovery (L)'
        
    return None

def prepare_aggregated_data(vdata, abx_cat_map):
    """Prepares data for aggregated boxplots with mixed stages."""
    df = vdata[vdata["study_label"] == "Antibiotic"].copy()
    df = df[df['proj_id'].isin(config.ABX_INTERVALS.keys())]
    
    # Filter by time window ([-25, 30]) as requested to include baseline while excluding long-term
    df = df[(df['Antibiotic_Timepoint'] >= -25) & (df['Antibiotic_Timepoint'] <= 30)]

    df['Category'] = df['Antibiotic_Type'].map(abx_cat_map)
    df['Time_Stage'] = df.apply(lambda r: classify_timepoint_mixed(r, abx_cat_map), axis=1)
    df = df[df['Time_Stage'].notna()]
    df = df[df['Category'].fillna('').astype(str).str.lower() != 'control']
    
    # We don't strictly cast to categorical order here because it varies by category
    return df

def calculate_avg_exposure(traj_df):
    """
    Dynamically calculates average exposure duration per category 
    based on the cohorts present in that category.
    """
    cat_exposure = {}
    
    # Defaults
    default_exposure = 7
    
    # Categories to process
    categories = traj_df['Category'].unique()
    
    for cat in categories:
        if cat == 'Control': 
            cat_exposure[cat] = 0
            continue
            
        # Get cohorts associated with this category
        cohorts = traj_df[traj_df['Category'] == cat]['Cohort'].unique()
        
        durations = []
        for proj in cohorts:
            if proj in config.ABX_INTERVALS:
                start, end = config.ABX_INTERVALS[proj]
                # Duration is the 'end' day (assuming start is typically Day 0/1)
                # User preference: "0-4" -> 4 days
                durations.append(end)
        
        if durations:
            # Logic Change: Use MAX duration (Union of time windows) as requested
            max_dur = np.max(durations)
            cat_exposure[cat] = max_dur
            print(f"Category '{cat}' exposure calculated from {cohorts}: {max_dur:.1f} days (MAX)")
        else:
            cat_exposure[cat] = default_exposure
            print(f"Category '{cat}' exposure defaulted to {default_exposure} days (no interval data)")
            
    # Add Mild-Impact default if missing (sometimes it has diverse cohorts)
    if 'Mild-Impact' not in cat_exposure:
        cat_exposure['Mild-Impact'] = default_exposure

    return cat_exposure

# ============================================================================
# 4. PLOTTING FUNCTIONS
# ============================================================================
# ============================================================================
# 4. PLOTTING FUNCTIONS 
# ============================================================================

def plot_scatter(df, ax):
    """Classification Scatter Map."""
    # 设置整个画板的极限范围
    X_MIN, X_MAX = -3.0, 3.0
    Y_MIN, Y_MAX = -3.0, 1.0
    
    # 设置规范边界
    T = -1.0         # 核心划分阈值 (-1)
    T_RIGHT = 1.0    # Oral-Targeting 的右边界 (1)
    
    # 重新定义四大象限的填充坐标 (x, y, width, height, Category)
    zones = [
        (X_MIN, Y_MIN, T - X_MIN, T - Y_MIN, 'Broad-Spectrum'),  # 左下角: x[-3, -1], y[-3, -1]
        (X_MIN, T, T - X_MIN, Y_MAX - T, 'Gut-Targeting'),       # 左上角: x[-3, -1], y[-1, 1]
        (T, Y_MIN, T_RIGHT - T, T - Y_MIN, 'Oral-Targeting')     # 🎯 右下角修改: x[-1, 1], y[-3, -1]
    ]
    
    for x, y, w, h, cat in zones:
        ax.add_patch(Rectangle((x, y), w, h, color=config.PALETTE[cat], alpha=0.08, zorder=0))
    
    # 加上 -1 边界的虚线
    ax.axvline(T, c='grey', ls='--', alpha=0.6, zorder=1)
    ax.axhline(T, c='grey', ls='--', alpha=0.6, zorder=1)
    
    # 0 刻度的实线（基准线）
    ax.axvline(0, c='black', lw=1, zorder=1)
    ax.axhline(0, c='black', lw=1, zorder=1)
    
    sns.scatterplot(data=df, x='Gut_Peak', y='Oral_Peak', hue='Category', palette=config.PALETTE, s=180, alpha=0.9, edgecolor='k', ax=ax, legend=False, zorder=10)
    
    for _, row in df.iterrows():
        lbl = str(row['Antibiotic']).replace('MEM500+VAN500+GEN40', 'MEM+VAN+GEN')
        off = 0.08
        x_pos = row['Gut_Peak'] - off if row['Gut_Peak'] > 0.3 else row['Gut_Peak'] + off
        txt = ax.text(x_pos, row['Oral_Peak'] + off, lbl, fontsize=9, fontweight='bold', zorder=15)
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white', alpha=0.7)])

    ax.set_xlabel(f"Gut Peak Depletion ({config.METRIC_LABEL})", fontweight='bold')
    ax.set_ylabel(f"Oral Peak Depletion \n ({config.METRIC_LABEL})", fontweight='bold')
    
    # 显式设定 XY 轴极限
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    

def plot_dual_axis(df, category, site, ax, exposure_map, is_left_col=False):
    """Biomass & Diversity Trajectories with Phase Highlights."""
    color = config.PALETTE.get(category, '#333')
    sub_bio = df[(df['Category'] == category) & (df['Site'] == site) & (df['Metric_Type'].str.contains('Biomass'))]
    sub_div = df[(df['Category'] == category) & (df['Site'] == site) & (df['Metric_Type'].str.contains('Diversity'))]
    
    if sub_bio.empty: 
        ax.set_axis_off()
        return
    
    
    # --- Trajectory Lines ---
    sns.lineplot(data=sub_bio, x='Timepoint', y='Value', color=color, ax=ax, errorbar=('ci', 68), **config.STYLE_BIOMASS, zorder=10)
    sns.lineplot(data=sub_bio, x='Timepoint', y='Value', units='Antibiotic', estimator=None, color=color, alpha=0.1, lw=0.6, ax=ax, zorder=5)
    
    ax.axhline(0, c='black', lw=0.8, alpha=0.5, zorder=1)
    # Set Y-axis limits as requested [-3, 1]
    ax.set_ylim(-3.0, 1.0)
    
    # --- Diversity (Secondary Axis) ---
    ax2 = ax.twinx()
    sns.lineplot(data=sub_div, x='Timepoint', y='Value', color='#444444', ax=ax2, errorbar=None, **config.STYLE_DIV, zorder=8)
    ax2.set_ylim(0, 2.8 if site == 'Oral' else 6.5)
    
    # --- Styling ---
    ax.set_xlabel(""); ax.set_ylabel(""); ax2.set_ylabel("")
    ax.grid(True, axis='y', alpha=0.15, ls=':')
    
    # RESTORE RIGHT SPINE (Better for dual-axis plots)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    
    if site == 'Oral': 
        ax.set_title(category, color=color, fontweight='bold', fontsize=12, pad=8)
    
    if is_left_col:
        ax.set_ylabel(f"{site}\nImpact", fontweight='bold', color=color, fontsize=10)
        ax.tick_params(axis='y', colors=color, labelsize=8)
        ax2.set_yticklabels([]); ax2.tick_params(right=False)
    elif category == 'Mild-Impact':
        ax.set_yticklabels([]); ax.tick_params(left=False)
        ax2.set_ylabel("Diversity", fontweight='bold', rotation=270, labelpad=15, color='#444444', fontsize=10)
        ax2.tick_params(axis='y', labelsize=8)
    else:
        ax.set_yticklabels([]); ax.tick_params(left=False)
        ax2.set_yticklabels([]); ax2.tick_params(right=False)
    
    # Optimize X-axis - Set fixed limits [-3, 30] as requested
    ax.set_xlim(-15, 30)
    ax.tick_params(axis='x', labelsize=8)
    
    # Adjust shading based on Category specific exposure duration
    xmin, xmax = ax.get_xlim()
    if xmin < 0.5:
        ax.axvspan(max(xmin, -20), 0.5, color='#f0f0f0', alpha=0.3, zorder=0)
    
    # Dynamic Exposure Fill
    exposure_days = exposure_map.get(category, 7)
    # Start at 0.5 (Day 0 is baseline/start), end at 0.5 + duration
    # This visually represents the treatment window starting right after baseline
    fill_end = 0.5 + exposure_days
    
    if fill_end > 0.5:
        ax.axvspan(0.5, min(xmax, fill_end), color='#ffd8a8', alpha=0.4, zorder=0)

def plot_aggregated_boxplots(df, axes_row, site, hide_xlabel=False):
    """Aggregated Boxplots row with mixed stage support."""
    cats = ['Broad-Spectrum', 'Oral-Targeting', 'Gut-Targeting', 'Mild-Impact']
    col_name = config.BIOMASS_COLS[site]
    
    for i, cat in enumerate(cats):
        ax = axes_row[i]
        plot_data = df[df['Category'] == cat].dropna(subset=[col_name])
        if plot_data.empty: 
            ax.set_axis_off(); continue
            
        # Dynamic Stage Detection
        unique_stages = set(plot_data['Time_Stage'].unique())
        
        # Rename logic: If Recovery (S) exists but Recovery (L) does not, rename S -> Recovery
        if 'Recovery (S)' in unique_stages and 'Recovery (L)' not in unique_stages:
            plot_data = plot_data.copy() # Ensure specific copy
            plot_data['Time_Stage'] = plot_data['Time_Stage'].replace({'Recovery (S)': 'Recovery'})
            unique_stages = set(plot_data['Time_Stage'].unique()) # Update set
            
        # Define logical sort order
        full_order = ['Before', 'During', 'Recovery (S)', 'Recovery', 'Recovery (L)']
        stage_order = [s for s in full_order if s in unique_stages]
        
        stage_palette = config.get_stage_palette(cat)
        
        sns.boxplot(data=plot_data, x='Time_Stage', y=col_name, order=stage_order, palette=stage_palette, ax=ax, width=0.6, showfliers=False, boxprops=dict(alpha=0.6))
        sns.stripplot(data=plot_data, x='Time_Stage', y=col_name, order=stage_order, color='black', alpha=0.3, size=2.5, jitter=0.2, ax=ax)
        
        
        # Stats vs Before (Unpaired Mann-Whitney / Wilcoxon Rank-Sum)
        d_before = plot_data[plot_data['Time_Stage'] == 'Before']
        
        sig_brackets = []  # Store bracket info: (x1, x2, sig_text, bracket_index)
        
        if not d_before.empty:
            for k, stage in enumerate(stage_order):
                if stage == 'Before': continue
                d_stage = plot_data[plot_data['Time_Stage'] == stage]
                
                # Unpaired data extraction
                val_before = d_before[col_name].dropna()
                val_stage = d_stage[col_name].dropna()
                if len(val_before) < 2 or len(val_stage) < 2: continue
                
                # Mann-Whitney U test (Wilcoxon Rank-Sum)
                try:
                    _, p_val = mannwhitneyu(val_before, val_stage, alternative='two-sided')
                except ValueError:
                    p_val = 1.0

                if p_val < 0.05:
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                    # Store: x positions (0 to k), significance text, bracket level
                    sig_brackets.append((0, k, sig, len(sig_brackets)))
            
        # 1. Apply SymLog scale and initial limits FIRST
        y_data_min = plot_data[col_name].replace(0, np.nan).min()
        y_data_max = plot_data[col_name].max()
        
        if site == 'Oral':
            linthresh = 1e5
            ax.set_yscale('symlog', linthresh=linthresh)
            display_min = max(1e5, y_data_min * 0.5) if not np.isnan(y_data_min) else 1e5
            display_max = y_data_max * 2 if y_data_max > 0 else 1e10
            # Ensure display_max is at least 10x display_min for log range
            display_max = max(display_max, display_min * 10)
        else:  # Gut
            linthresh = 1e10
            ax.set_yscale('symlog', linthresh=linthresh)
            display_min = max(1e10, y_data_min * 0.5) if not np.isnan(y_data_min) else 1e10
            display_max = y_data_max * 2 if y_data_max > 0 else 1e12
            display_max = max(display_max, display_min * 5) # Gut is tighter

        ax.set_ylim(display_min, display_max)
        
        # 2. Draw brackets using log-space offsets for visual consistency
        if sig_brackets:
            log_min = np.log10(display_min)
            log_max = np.log10(display_max)
            log_range = log_max - log_min
            
            # Start brackets slightly above the highest data point (in log units)
            # Use data max if it's within display limits, else use a fixed point
            current_top_log = np.log10(max(y_data_max, display_min))
            start_y_log = current_top_log + 0.05 * log_range
            
            h_log = 0.02 * log_range      # Vertical tick height (reduced)
            step_log = 0.12 * log_range   # Vertical spacing between brackets (increased)
            text_off_log = 0.04 * log_range # Text offset above bracket (increased)
            
            max_bracket_y_log = start_y_log
            
            for x1, x2, sig_text, bracket_idx in sig_brackets:
                y_base_log = start_y_log + bracket_idx * step_log
                y_top_log = y_base_log + h_log
                
                y_base = 10**y_base_log
                y_top = 10**y_top_log
                y_text = 10**(y_top_log + text_off_log)
                
                # Draw bracket
                ax.plot([x1, x1, x2, x2], [y_base, y_top, y_top, y_base], 
                       lw=1.2, c='k', clip_on=False)
                ax.text((x1 + x2)/2, y_text, sig_text, ha='center', va='bottom', 
                       fontsize=8, fontweight='bold', clip_on=False)
                
                max_bracket_y_log = max(max_bracket_y_log, y_top_log + 0.1 * log_range)

            # Update ylim to accommodate the highest bracket
            ax.set_ylim(display_min, 10**max_bracket_y_log)
        else:
            # Padding for no-bracket case
            log_min = np.log10(display_min)
            log_max = np.log10(display_max)
            ax.set_ylim(display_min, 10**(log_max + 0.1 * (log_max - log_min)))

        # 3. Ticks and Labels
        if site == 'Oral':
            tick_values = [10**i for i in range(5, 12)]
            tick_values = [t for t in tick_values if display_min <= t <= ax.get_ylim()[1]]
            ax.set_yticks(tick_values)
        else:
            tick_values = []
            for exp in range(10, 13):
                base = 10**exp
                tick_values.extend([base, 2*base, 5*base])
            tick_values = [t for t in tick_values if display_min <= t <= ax.get_ylim()[1]]
            ax.set_yticks(tick_values)
        
        # Custom formatter to display as 10^N
        def log_formatter(x, pos):
            if x == 0: return '0'
            abs_x = abs(x)
            exp = np.log10(abs_x)
            if abs(exp - round(exp)) < 0.01:
                return f'$10^{{{int(round(exp))}}}$'
            else:
                mantissa = abs_x / (10 ** int(np.floor(exp)))
                if abs(mantissa - round(mantissa)) < 0.01:
                    return f'${int(round(mantissa))}×10^{{{int(np.floor(exp))}}}$'
                return f'{x:.1e}'
        
        ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
        
        if i == 0: ax.set_ylabel(f"{site} QMP", fontweight='bold')
        else: ax.set_ylabel("")
        ax.set_xlabel("")
        
        if hide_xlabel:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)
        else:
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        sns.despine(ax=ax)

# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == "__main__":
    cache_file = proj_base_path / "data/cache/antibiotic_analysis_cache_v1.pkl.gz"
    data = prepare_project_dataset(proj_base_path, "fig4", mask_projids=["PRJEB8094"], cache_path=str(cache_file), use_v2=True)
    vdata = data['vdata2']; _, _, fig_pre = data['paths']
    
    # --- 1. Processing ---
    full_traj_df, class_df = process_unified_data(vdata)
    abx_cat_map = dict(zip(class_df['Antibiotic'], class_df['Category']))
    df_agg = prepare_aggregated_data(vdata, abx_cat_map)
    
    # Calculate Dynamic Exposure Map
    exposure_map = calculate_avg_exposure(full_traj_df)
    
    # --- 2. Plotting (Restored Aesthetics & Absolute Aligned ABC) ---
    # fig = plt.figure(figsize=(15, 15)) # 恢复 15x15 的精美比例
    fig = plt.figure(figsize=(16, 13.2)) # 统一宽度为 16
    
    # Outer GS
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[0.20, 0.80], hspace=0.28)
    
    # --- 区域 A. Classification Scatter (Top) ---
    ax_scatter = plt.subplot(gs_main[0])
    plot_scatter(class_df, ax_scatter)
    
    # --- 区域 B & C. Detailed Grid ---
    gs_sub = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs_main[1], 
                                              height_ratios=[1, 1, 1.3, 1.3],
                                              wspace=0.3, hspace=0.45)
    
    cats = ['Broad-Spectrum', 'Oral-Targeting', 'Gut-Targeting', 'Mild-Impact']
    ax_traj_first = None
    ax_box_first = None

    # --- 区域 B. Trajectories (Rows 0-1) ---
    for i, cat in enumerate(cats):
        for j, site in enumerate(['Oral', 'Gut']):
            ax = plt.subplot(gs_sub[j, i])
            if i == 0 and j == 0: ax_traj_first = ax  # 捕获B区域基准点
            
            plot_dual_axis(full_traj_df, cat, site, ax, exposure_map=exposure_map, is_left_col=(i==0))
            if j == 0: 
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Time (Days)", fontsize=9)
    
    # --- 区域 C. Aggregated Boxplots (Rows 2-3) ---
    for i, site in enumerate(['Oral', 'Gut']):
        axes_row = [plt.subplot(gs_sub[i+2, j]) for j in range(4)]
        if i == 0: ax_box_first = axes_row[0]  # 捕获C区域基准点
        
        plot_aggregated_boxplots(df_agg, axes_row, site, hide_xlabel=(site=='Oral'))

    # --- 👑 终极绝招：绝对全局坐标打标 A, B, C ---
    # 强制重绘一次画板，以获取各个坐标轴真实的物理 bounding box
    fig.canvas.draw()
    bbox_A = ax_scatter.get_position()
    bbox_B = ax_traj_first.get_position()
    bbox_C = ax_box_first.get_position()

    # 取全图最靠左的 X 轴起点，再往左偏移 4% 页面宽度，作为统一的打标垂直线！
    align_x = min(bbox_A.x0, bbox_B.x0, bbox_C.x0) - 0.04
    
    # 字体采用标准 Arial 加粗，大小 22 适配此画板恰到好处，不再突兀
    label_style = dict(fontsize=22, fontweight='bold', fontfamily='Arial', va='bottom', ha='right')

    # fig.text(align_x, bbox_A.y1, "A", **PANEL_LABEL_STYLE)
    # fig.text(align_x, bbox_B.y1 + 0.015, "B", **PANEL_LABEL_STYLE)
    # fig.text(align_x, bbox_C.y1 + 0.015, "C", **PANEL_LABEL_STYLE)
    

    # --- Legend: Trajectories ---
    legend_handles = [
        Line2D([0], [0], color='#555555', lw=2.5, label=f"Biomass ({config.METRIC_LABEL})"),
        Line2D([0], [0], color='#555555', lw=1.5, ls='--', label="Shannon Diversity"),
        Patch(facecolor='#ff922b', alpha=0.6, edgecolor='none', label='Antibiotic Exposure')
    ]
    # 图例悬浮于 A 和 B 之间
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.70), 
               frameon=True, ncol=3, fontsize=10, facecolor='white', framealpha=0.9)
    
    save_file = f"{fig_pre}/fig4_Unified_Final_Fixed_anti5_box_agg.pdf"
    plt.savefig(save_file, bbox_inches='tight', dpi=config.DPI)
    print(f"Final PDF saved to: {save_file}")
    plt.close()