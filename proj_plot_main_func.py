# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:57:46 2025

Target:
1. metaphlan 版本的不一致
2. 绘图的排列，左边为同一种标签，按显著性排列
3. 绘图：boxplot

@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""

from pathlib import Path
import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Set, List, Tuple, Optional, Dict
import os
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats
import re
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, ranksums
from scipy.stats import linregress
from matplotlib.ticker import FuncFormatter
import pickle
import gzip

#local
from dftools.df_extractor import bioProjsDataloader
from dftools.Metaphlan_df_extractor import trim_bact_names
from proj_config import proj_base_path

# proj_base_path = Path(r"C:\Users\longz\OneDrive\HUST_Working\ChenWorking\testproj2") #win
# proj_base_path = Path(r"/home/zw/zwLearn/zwProjHust/testproj2") #linux
                      
def filter_project_data(vdata, qmp, rmp, mask_list, col="proj_id"):
    """
    过滤掉指定项目ID的数据
    
    Parameters:
    vdata, qmp, rmp: 要过滤的数据框
    mask_projids: 要屏蔽的项目ID列表
    
    Returns:
    过滤后的数据框元组
    """
    if not mask_list:
        return vdata, qmp, rmp
    keep_mask = ~vdata[col].isin(mask_list)
    return vdata[keep_mask], qmp[keep_mask], rmp[keep_mask]

def extract_data(vdata: pd.DataFrame, 
                qmp: pd.DataFrame, 
                rmp: pd.DataFrame,
                col_selector: Union[Set, List, np.ndarray, None] = None,
                row_selector: Union[Set, List, np.ndarray, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    灵活提取数据的函数，支持行列同时筛选
    
    Parameters:
    -----------
    vdata, qmp, rmp : pd.DataFrame
        要提取的数据框，假设它们有相同的index
    col_selector : Union[Set, List, np.ndarray, None], optional
        列选择器，支持以下类型：
        - Set/List: 列名集合或列表，用于匹配rmp.columns
        - np.ndarray (bool): 布尔索引数组，直接用于列选择
        - None: 选择所有列
    row_selector : Union[Set, List, np.ndarray, None], optional
        行选择器，支持以下类型：
        - Set/List: 行索引集合或列表
        - np.ndarray (bool): 布尔索引数组，直接用于行选择
        - None: 选择所有行
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        过滤后的 (vdata, qmp, rmp)
    """
    
    # 处理列选择
    if col_selector is None:
        col_mask = slice(None)  # 选择所有列
    elif isinstance(col_selector, (set, list)):
        # 将set/list转换为布尔索引
        col_mask = np.isin(np.array(rmp.columns), list(col_selector))
    elif isinstance(col_selector, np.ndarray):
        if col_selector.dtype == bool:
            col_mask = col_selector
        else:
            raise ValueError("如果col_selector是ndarray，必须是布尔类型")
    else:
        raise ValueError("col_selector必须是set, list, bool ndarray或None")
    
    # 处理行选择
    if row_selector is None:
        row_mask = slice(None)  # 选择所有行
    elif isinstance(row_selector, (set, list)):
        # 将set/list转换为布尔索引
        row_mask = np.isin(np.array(vdata.index), list(row_selector))
    elif isinstance(row_selector, np.ndarray):
        if row_selector.dtype == bool:
            row_mask = row_selector
        else:
            raise ValueError("如果row_selector是ndarray，必须是布尔类型")
    else:
        raise ValueError("row_selector必须是set, list, bool ndarray或None")
    
    # 应用筛选
    if isinstance(row_mask, slice) and isinstance(col_mask, slice):
        # 都是全选
        return vdata.copy(), qmp.copy(), rmp.copy()
    elif isinstance(row_mask, slice):
        # 只选择列
        return (vdata.copy(), 
                qmp.loc[:, col_mask] if hasattr(qmp, 'loc') else qmp[:, col_mask],
                rmp.loc[:, col_mask] if hasattr(rmp, 'loc') else rmp[:, col_mask])
    elif isinstance(col_mask, slice):
        # 只选择行
        return (vdata.loc[row_mask, :] if hasattr(vdata, 'loc') else vdata[row_mask, :],
                qmp.loc[row_mask, :] if hasattr(qmp, 'loc') else qmp[row_mask, :],
                rmp.loc[row_mask, :] if hasattr(rmp, 'loc') else rmp[row_mask, :])
    else:
        # 同时选择行和列
        return (vdata.loc[row_mask, :] if hasattr(vdata, 'loc') else vdata[row_mask, :],
                qmp.loc[row_mask, col_mask] if hasattr(qmp, 'loc') else qmp[row_mask, :][:, col_mask],
                rmp.loc[row_mask, col_mask] if hasattr(rmp, 'loc') else rmp[row_mask, :][:, col_mask])

def get_path_proj(proj_base_path, fig_ppath):
    proj_data_path = proj_base_path / "data"
    # file_path = proj_data_path / "cancer_data"
    # load_qmp_dir = proj_data_path / "load_qmp_data"
    proj_cache_path = proj_data_path / "cache"
    fig_pre = proj_base_path / f"fig/fig_main/{fig_ppath}"
    return proj_data_path, proj_cache_path, fig_pre

def load_data_proj(proj_cache_path, mp4ver="mp4v2025", mask_projid = ["PRJEB8094"]):
    proj_dataloader = bioProjsDataloader().load_local(proj_cache_path / f"cancerdata_20260124_43set_{mp4ver}.pkl.gz")
    ppi_dataloader =  bioProjsDataloader().load_local(proj_cache_path / "ppi02_data.pkl.gz")
    # ppi2_dataloader =  bioProjsDataloader().load_local(proj_cache_path / "ppi02_data.pkl.gz")
    lc_dataloader = bioProjsDataloader().load_local(proj_cache_path / "lc_data.pkl.gz")
    
    proj_dataloader = proj_dataloader  + ppi_dataloader # + lc_dataloader
    
    
    vdata = pd.merge(proj_dataloader.meta_data, proj_dataloader.load, left_index=True, right_index=True, how='left')
    vdata.loc[vdata.loc[:, "BMI"] < 10, "BMI"] =  np.NAN
    rmp = proj_dataloader.rmp
    qmp = proj_dataloader.get_qmp()
    
    if mask_projid:
        vdata, qmp, rmp = filter_project_data(vdata, qmp, rmp, mask_projid)

    return proj_dataloader, vdata, rmp, qmp


############## ASV def ###########
def extract_fusobacterium_columns(
    rmp: pd.DataFrame, 
    pattern: str = "Fusobacteri",
    similarity_threshold: int = 85,
    exclude_solobacterium: bool = True,
    extract_level: str = "s__",
    return_simple_name: bool = False
) -> Set[str]:
    """
    从RMP数据框中提取所有Fusobacterium相关的完整列名
    
    Parameters:
    -----------
    rmp : pd.DataFrame
        相对丰度数据框，列名为完整的分类路径
    pattern : str, default "Fusobacteri"
        搜索模式，默认"Fusobacteri"可以匹配Fusobacterium和Fusobacteriaceae
    similarity_threshold : int, default 85
        模糊匹配的相似度阈值（用于fuzzywuzzy）
    exclude_solobacterium : bool, default True
        是否排除Solobacterium（虽然名字相似但不是Fusobacterium）
    extract_level : str, default "s__"
        提取级别，可选: "s__"(种), "g__"(属), "f__"(科), "o__"(目), "c__"(纲), "p__"(门)
    return_simple_name : bool, default False
        是否只返回简化的名称（如 "s__Fusobacterium_nucleatum"）
        False: 返回完整路径 "k__Bacteria|p__...|s__Fusobacterium_nucleatum"
        True: 返回简化名称 "s__Fusobacterium_nucleatum"
    
    Returns:
    --------
    Set[str]
        如果return_simple_name=False: 包含完整分类路径的列名集合
        如果return_simple_name=True: 包含对应级别简化名称的集合
    """
    from thefuzz import fuzz
    
    # 验证extract_level参数
    valid_levels = ["s__", "g__", "f__", "o__", "c__", "p__"]
    if extract_level not in valid_levels:
        raise ValueError(f"extract_level必须是以下之一: {valid_levels}")
    
    fuso_columns = set()
    fuso_simple_names = set()
    
    for col in rmp.columns:
        # 根据指定级别提取名称
        pattern_regex = rf"{re.escape(extract_level)}([^|]+)"
        match = re.search(pattern_regex, col)
        if not match:
            continue
            
        taxon_name = match.group(1).strip()
        
        # 跳过 unclassified
        if not taxon_name or taxon_name.endswith('_unclassified'):
            continue
        
        # 模糊匹配
        score = fuzz.partial_ratio(pattern.lower(), taxon_name.lower())
        
        if score >= similarity_threshold:
            # 排除Solobacterium（如果需要）
            if exclude_solobacterium and 'solobacterium' in taxon_name.lower():
                continue
            
            fuso_columns.add(col)
            fuso_simple_names.add(f"{extract_level}{taxon_name}")
    
    return fuso_simple_names if return_simple_name else fuso_columns



def load_asv_proj(vdata, rmp, qmp, proj_cache_path, include_v2=True, strategy="oral_priority"):
    """
    Unified ASV project loading. Extracts bacterial sets, matches them to columns,
    and calculates all diversity and biomass metrics.
    """
    # 1. Load Raw Sets (CSV)
    raw_sets = {}
    try:
        def _load(f): return set(pd.read_csv(proj_cache_path / f).values.flatten())
        raw_sets["liao"] = _load("oral_liao.csv")
        raw_sets["zhu90"] = _load("oral_zhu_90.csv")
        raw_sets["zhu95"] = _load("oral_zhu_95.csv")
        raw_sets["ehomd"] = _load("oral_ehomd.csv")
        raw_sets["hrom_ectopic"] = _load("oral_hrom_ectopic.csv")
        if include_v2: raw_sets["hrom_shared"] = _load("oral_hrom_shared.csv")
    except Exception as e:
        print(f"[Warning] Failed to load some CSV sets: {e}")

    # 2. Extract Functional Sets (Regex/Special)
    raw_sets["fuso"] = extract_fusobacterium_columns(rmp, extract_level="s__", return_simple_name=True)
    raw_sets["zhucomb"] = raw_sets.get("ehomd", set()) | raw_sets.get("zhu95", set()) | raw_sets.get("hrom_ectopic", set())
    raw_sets["union"] = raw_sets.get("ehomd", set()) | raw_sets.get("zhu95", set()) | raw_sets.get("liao", set())

    # 3. Match Columns to Sets
    oral_bact_dict = {k: set() for k in raw_sets.keys()}
    v2_oral_raw, v2_shared_raw = set(), set()
    v2_target_oral = raw_sets.get("ehomd", set()) | raw_sets.get("zhu95", set()) | raw_sets.get("hrom_ectopic", set())
    v2_target_shared = raw_sets.get("hrom_shared", set())

    for col in rmp.columns:
        # V1 logic (Simple match)
        spe = col.split("|")[-1]
        for k in oral_bact_dict:
            if spe in raw_sets[k]: oral_bact_dict[k].add(col)
        
        # V2 logic (Cleaned match)
        if include_v2:
            clean_name = clean_bact_species_name(col)
            if clean_name:
                if clean_name in v2_target_oral: v2_oral_raw.add(col)
                if clean_name in v2_target_shared: v2_shared_raw.add(col)

    for k, v in oral_bact_dict.items():
        print(f"{k}: matched {len(v)} columns")

    # 4. Resolve V2 (Oral/Shared/Gut) Strategies
    if include_v2:
        suffix = "ehzhhr"
        if strategy == "shared_priority":
            cols_shared_final = v2_shared_raw
            cols_oral_final = v2_oral_raw - v2_shared_raw
        else: # oral_priority
            cols_oral_final = v2_oral_raw
            cols_shared_final = v2_shared_raw - v2_oral_raw
        
        cols_gut_final = set(rmp.columns) - (cols_oral_final | cols_shared_final)
        print(f"v2 ({suffix}) - Oral: {len(cols_oral_final)}, Shared: {len(cols_shared_final)}, Gut: {len(cols_gut_final)}")
        
        # Store in dict
        oral_bact_dict[f"{suffix}_oral_final"] = cols_oral_final
        oral_bact_dict[f"{suffix}_shared_final"] = cols_shared_final
        oral_bact_dict[f"{suffix}_gut_final"] = cols_gut_final
        oral_bact_dict[suffix] = cols_oral_final # alias

    # 5. Calculation Logic (Metrics)
    tmp_metrics = {}
    
    # a. Total Metrics
    tmp_metrics["total_richness"] = (rmp > 0).sum(axis=1)
    tmp_metrics["total_shannon"] = rmp.apply(lambda x: stats.entropy(x) if x.sum() > 0 else 0, axis=1)

    # b. V1-style Metrics (Oral/Gut pairs)
    for k in [k for k in oral_bact_dict if not k.startswith("ehzhhr")]:
        oral_cols, gut_cols = list(oral_bact_dict[k]), list(rmp.columns.difference(oral_bact_dict[k]))
        
        # QMP/RMP sums
        tmp_metrics[f"oral_qmp_{k}"] = qmp.loc[:, oral_cols].sum(1)
        tmp_metrics[f"gut_qmp_{k}"] = qmp.loc[:, gut_cols].sum(1)
        tmp_metrics[f"oral_rmp_{k}"] = rmp.loc[:, oral_cols].sum(1)
        tmp_metrics[f"gut_rmp_{k}"] = rmp.loc[:, gut_cols].sum(1)
        
        # Diversity
        tmp_metrics[f"oral_richness_{k}"] = (rmp[oral_cols] > 0).sum(1)
        tmp_metrics[f"gut_richness_{k}"] = (rmp[gut_cols] > 0).sum(1)
        tmp_metrics[f"oral_shannon_{k}"] = rmp[oral_cols].apply(lambda x: stats.entropy(x) if x.sum() > 0 else 0, axis=1)
        tmp_metrics[f"gut_shannon_{k}"] = rmp[gut_cols].apply(lambda x: stats.entropy(x) if x.sum() > 0 else 0, axis=1)

        # Fuso special
        fuso_cols = [c for c in oral_cols if re.search(r'\|s__Fusobacteri(um|aceae)([_|]|$)', c)]
        tmp_metrics[f"fuso_qmp_{k}"] = qmp.loc[:, fuso_cols].sum(1)
        tmp_metrics[f"nfuso_qmp_{k}"] = tmp_metrics[f"oral_qmp_{k}"] - tmp_metrics[f"fuso_qmp_{k}"]

    # c. V2-style Metrics (ehzhhr)
    if include_v2:
        suffix = "ehzhhr"
        for niche in ["oral", "shared", "gut"]:
            cols = list(oral_bact_dict[f"{suffix}_{niche}_final"])
            tmp_metrics[f"{niche}_qmp_{suffix}"] = qmp.loc[:, cols].sum(axis=1)
            tmp_metrics[f"{niche}_rmp_{suffix}"] = rmp.loc[:, cols].sum(axis=1)
            tmp_metrics[f"{niche}_shannon_{suffix}"] = rmp.loc[:, cols].apply(lambda x: stats.entropy(x) if x.sum() > 0 else 0, axis=1)

    # 6. Final Data Assembly
    df_metrics = pd.DataFrame(tmp_metrics)
    vdata2 = pd.concat([vdata, df_metrics], axis=1)
    
    # Antibiotic Label Fix
    vanti = vdata2[vdata2['study_label'] == "Antibiotic"].copy()
    if not vanti.empty and 'Antibiotic_Type' in vanti.columns:
        vanti['Disease_label'] = vanti['Antibiotic_Type'].astype(str) + "_" + vanti['Antibiotic_Timepoint'].astype(str)
        vdata2.loc[vdata2['study_label'] == "Antibiotic", 'Disease_label'] = vanti['Disease_label']
    
    return vdata2, vanti, oral_bact_dict



def clean_bact_species_name(col_str):
    """
    清洗函数：从 Metaphlan 复杂的列名中提取“核心物种名”
    用于在 load_asv_proj_v2 中匹配数据库
    """
    if '|s__' not in col_str:
        return None
        
    # A. 提取 s__ 部分
    raw = col_str.split('|s__')[1].split('|')[0]
    
    # B. 逐步清洗后缀
    clean = re.sub(r'_SGB\d+$', '', raw)
    clean = re.sub(r'_group_\d+$', '', clean)
    clean = re.sub(r'_unclassified$', '', clean)
    
    if re.search(r'_[A-Z]$', clean):
        clean = clean[:-2]
        
    # 特殊处理：sp_A12, SSPC 等如果是数据库里有的，应该保留
    # 这里我们只做了通用清洗，确保能匹配上数据库的核心名
    
    return "s__" + clean if not clean.startswith("s__") else clean

def handle_cache(cache_path, data_generator_func=None, force_reload=False):
    """
    通用缓存处理函数 (支持 gzip 压缩)
    
    Parameters:
    -----------
    cache_path : str or Path
        缓存文件的完整路径。如果以 '.gz' 结尾，将自动使用 gzip 压缩。
    data_generator_func : function
        如果缓存不存在，用于生成数据的回调函数 (无参数)
    force_reload : bool
        是否强制重新生成并覆盖缓存
        
    Returns:
    --------
    data : dict
        加载或生成的数据字典
    """
    # 转换为字符串以便处理 Path 对象
    path_str = str(cache_path)
    use_gzip = path_str.endswith('.gz')
    
    # --- 读取逻辑 ---
    if cache_path and os.path.exists(cache_path) and not force_reload:
        print(f"[Cache] Found cache file: {cache_path}")
        try:
            # 根据后缀选择打开方式
            opener = gzip.open if use_gzip else open
            
            with opener(cache_path, 'rb') as f:
                print("[Cache] Loading (this may take a while)...")
                data = pickle.load(f)
                print("[Cache] Loaded successfully.")
                return data
        except Exception as e:
            print(f"[Cache] Error loading cache: {e}. Will regenerate.")
    
    if data_generator_func is None:
        return None

    # --- 生成逻辑 ---
    print("[Cache] Generating new data...")
    data = data_generator_func()
    
    # --- 保存逻辑 ---
    if cache_path:
        # 确保存储目录存在
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        print(f"[Cache] Saving to {cache_path} (Compression: {'On' if use_gzip else 'Off'})...")
        try:
            # 根据后缀选择打开方式
            opener = gzip.open if use_gzip else open
            
            with opener(cache_path, 'wb') as f:
                # 使用最高协议版本，速度更快且体积通常更小
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("[Cache] Saved.")
        except Exception as e:
            print(f"[Cache] Warning: Failed to save cache: {e}")
            
    return data

def prepare_project_dataset(
        proj_base_path: Path,
        fig_ppath: str,
        mp4ver: str = "mp4v2025",
        mask_projids: Optional[List[str]] = None,
        mask_disease_labels: Optional[List[str]] = None,
        cache_path: Optional[str] = None,   
        force_reload: bool = False,
        use_v2: bool = True,
        v2_strategy: str = "oral_priority"
    ):
    # 1. 路径准备
    proj_data_path, proj_cache_path, fig_pre = get_path_proj(proj_base_path, fig_ppath)
    
    # --- 修改点 1: 核心生成逻辑只负责加载“全量”数据 ---
    def _generate_full_data():
        print(f"--- Loading Full Dataset for Cache (Ver: {mp4ver}) ---")
        
        # 加载全量数据，此处 mask_projid 传 None，保证缓存的是完整版
        proj_dataloader, vdata, rmp, qmp = load_data_proj(
            proj_cache_path,
            mp4ver=mp4ver,
            mask_projid=None 
        )
        
        # 计算 ASV 特征 (这一步最耗时，必须缓存)
        vdata2, vanti, oral_bact_dict = load_asv_proj(
            vdata, rmp, qmp, proj_cache_path, 
            include_v2=use_v2, 
            strategy=v2_strategy
        )
        
        return {
            'vdata2': vdata2,
            'vanti': vanti,
            'oral_bact_dict': oral_bact_dict,
            'rmp': rmp,
            'qmp': qmp
        }

    # 2. 获取数据 (从缓存读取或新生成)
    if cache_path:
        data = handle_cache(cache_path, _generate_full_data, force_reload)
    else:
        data = _generate_full_data()
    
    # --- 修改点 2: 在缓存外应用屏蔽逻辑 ---
    # 这样每次运行脚本，即使缓存存在，也会根据 mask 列表动态过滤
    vdata2 = data['vdata2']
    rmp = data['rmp']
    qmp = data['qmp']

    # 应用项目 ID 屏蔽
    if mask_projids:
        print(f"[Filter] Masking Project IDs: {mask_projids}")
        keep_mask = ~vdata2['proj_id'].isin(mask_projids)
        vdata2 = vdata2[keep_mask]
        rmp = rmp.loc[vdata2.index]
        qmp = qmp.loc[vdata2.index]

    # 应用疾病标签屏蔽
    if mask_disease_labels:
        print(f"[Filter] Masking Disease Labels: {mask_disease_labels}")
        # 注意：这里要确保 col 参数与你的数据匹配，通常是 Disease_status 或 Disease_label
        col_name = "Disease_status" if "Disease_status" in vdata2.columns else "Disease_label"
        keep_mask = ~vdata2[col_name].isin(mask_disease_labels)
        vdata2 = vdata2[keep_mask]
        rmp = rmp.loc[vdata2.index]
        qmp = qmp.loc[vdata2.index]

    # 更新结果字典
    result_dict = {
        'vdata2': vdata2,
        'rmp': rmp,
        'qmp': qmp,
        'vanti': data['vanti'], # vanti 通常也需要根据 vdata2 的 index 过滤，如下：
        'oral_bact_dict': data['oral_bact_dict'],
        'paths': (proj_data_path, proj_cache_path, fig_pre)
    }
    
    # 确保 vanti 也是过滤后的
    result_dict['vanti'] = vdata2[vdata2['study_label'] == "Antibiotic"]

    return result_dict

def get_subject_col(df):
    """Robustly detect subject identifier column."""
    candidates = ['host_subject_id', 'subject_id', 'individual', 'patient_id', 'Subject_ID']
    for c in candidates:
        if c in df.columns:
            return c
            
    # Check if Antibiotic_SampleID matches the pattern of a Subject ID (duplicates exist)
    if 'Antibiotic_SampleID' in df.columns:
        if df['Antibiotic_SampleID'].duplicated().any():
            return 'Antibiotic_SampleID'
            
    return None

