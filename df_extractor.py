# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:13:45 2025

@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import zipfile
from collections import defaultdict
import re
import pickle
import copy
import gzip

# 可选依赖：pyarrow
try:
    import pyarrow.feather as feather
    import pyarrow as pa
    # import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    feather = None
    pa = None
    HAS_PYARROW = False

# import pyarrow.parquet as pq
import tempfile
import shutil

from .Metaphlan_df_extractor import MetaplanDataExtractor, read_df_metaphlan4, trim_bact_names


def is_direct_subdirectory( path, base_dir):
    base_path = Path(base_dir)
    path = Path(path)
    return path.parent == base_path


def extract_prj_id(project_name):
    """
    从项目名称中提取PRJ编号
    
    Args:
        project_name: 项目名称，如 "PRJEB20800_Antibiotic" 或 "WGS_CCA_PRJNA932948"
        
    Returns:
        str: PRJ编号，如 "PRJEB20800" 或 "PRJNA932948"，找不到则返回None
    """
    # 匹配PRJ开头的编号模式
    match = re.search(r'(PRJ[A-Z]+\d+)', project_name)
    return match.group(1) if match else None

def find_qmp_in_external_dir(prj_id, qmp_base_dir, qmp_basenames, suffixes):
    """
    在外部QMP目录中根据PRJ编号查找对应的QMP文件
    """
    if not qmp_base_dir or not qmp_base_dir.exists():
        return None, float("inf")
    
    # 查找包含该PRJ编号的目录
    for item in qmp_base_dir.iterdir():
        if item.is_dir() and prj_id in item.name:
            # 在匹配的目录中查找QMP文件
            for ext in ["*.txt", "*.csv", "*.tsv"]:
                for f in item.glob(ext):
                    fname = f.name
                    fsuf = f.suffix
                    
                    # 检查是否匹配QMP文件名模式
                    for idx, sub in enumerate(qmp_basenames):
                        if sub in fname and fsuf in suffixes:
                            return f, idx
    
    return None, float("inf")

def data_group_by_proj(dir_list, mp4_basenames=["feat.txt", "merged_abundance"], base_dir="", 
                       suffixes=[".csv", ".txt", ".tsv"], 
                       meta_basenames=["meta_merge_man.csv", "meta_merge", "SraRunTable.csv", "SraRunTable.txt"],
                       qmp_basenames=["load_qmp.csv", "load_qmp.tsv", "load.csv", "load.tsv"],
                       qmp_base_dir=None):
    dct = defaultdict(lambda: {
        "mp4_data": None, "meta_data": None, "qmp_data": None,  # 新增qmp_data
        "mp4_priority": float("inf"), "meta_priority": float("inf"), "qmp_priority": float("inf")  # 记录优先级
    })
    
    base_dir = Path(base_dir)
    if qmp_base_dir:
        qmp_base_dir = Path(qmp_base_dir)
    dir_list = sorted(dir_list)
    
    for f in dir_list:
        f = Path(f)
        if f == base_dir:
            continue
        
        pdir = f.parent
        if pdir == Path("."):
            continue
        
        projk = pdir.relative_to(base_dir).as_posix()
        fname = f.name
        fsuf = f.suffix
        
        # 处理 mp4_basenames
        for idx, sub in enumerate(mp4_basenames):
            if sub in fname:
                # 检查文件名是否包含PRJ编号，如果包含则提升优先级
                prj_id = extract_prj_id(fname)
                effective_priority = idx
                if prj_id:  # 如果包含PRJ编号，优先级提升（数值减小）
                    effective_priority -= 0.5
                
                if effective_priority < dct[projk]["mp4_priority"]:  # 只有更高优先级才替换
                    if dct[projk]["mp4_data"] is not None:
                        reason = "包含项目特定标识符" if prj_id else "基础优先级更高"
                        print(f"⚠️ Warning: {projk} 的 mp4_data 从 {dct[projk]['mp4_data']} 替换为 {f}（{reason}）。")
                    dct[projk]["mp4_data"] = f
                    dct[projk]["mp4_priority"] = effective_priority
                break  # 只要匹配到了 mp4_basenames，就跳出循环
            
        # 处理 meta_basenames
        for idx, sub in enumerate(meta_basenames):
            if sub in fname and fsuf in suffixes:
                # 检查文件名是否包含PRJ编号，如果包含则提升优先级
                prj_id = extract_prj_id(fname)
                effective_priority = idx
                if prj_id:  # 如果包含PRJ编号，优先级提升（数值减小）
                    effective_priority -= 0.5
                
                if effective_priority < dct[projk]["meta_priority"]:  # 只有更高优先级才替换
                    if dct[projk]["meta_data"] is not None:
                        reason = "包含项目特定标识符" if prj_id else "基础优先级更高"
                        print(f"⚠️ Warning: {projk} 的 meta_data 从 {dct[projk]['meta_data']} 替换为 {f}（{reason}）。")
                    dct[projk]["meta_data"] = f
                    dct[projk]["meta_priority"] = effective_priority
                break  # 匹配到优先级高的 meta_basenames 就跳出
        
        # 处理 qmp_basenames
        for idx, sub in enumerate(qmp_basenames):
            if sub in fname and fsuf in suffixes:
                # 检查文件名是否包含PRJ编号，如果包含则提升优先级
                prj_id = extract_prj_id(fname)
                effective_priority = idx
                if prj_id:  # 如果包含PRJ编号，优先级提升（数值减小）
                    effective_priority -= 0.5
                
                if effective_priority < dct[projk]["qmp_priority"]:  # 只有更高优先级才替换
                    if dct[projk]["qmp_data"] is not None:
                        reason = "包含项目特定标识符" if prj_id else "基础优先级更高"
                        print(f"⚠️ Warning: {projk} 的 qmp_data 从 {dct[projk]['qmp_data']} 替换为 {f}（{reason}）。")
                    dct[projk]["qmp_data"] = f
                    dct[projk]["qmp_priority"] = effective_priority
                break  # 匹配到优先级高的 qmp_basenames 就跳出
    
    # 第二步：对于没有找到qmp_data的项目，去外部目录查找
    if qmp_base_dir:
        for projk in dct:
            if dct[projk]["qmp_data"] is None:
                prj_id = extract_prj_id(projk)
                if prj_id:
                    external_qmp_file, external_priority = find_qmp_in_external_dir(
                        prj_id, qmp_base_dir, qmp_basenames, suffixes
                    )
                    
                    if external_qmp_file:
                        print(f"✅ 在外部目录找到 {projk} 的QMP文件: {external_qmp_file}")
                        dct[projk]["qmp_data"] = external_qmp_file
                        dct[projk]["qmp_priority"] = external_priority
            
    # 移除优先级字段
    for projk in dct:
        del dct[projk]["mp4_priority"]
        del dct[projk]["meta_priority"]
        del dct[projk]["qmp_priority"]
    return dct




def get_clean_metadata(keys, cols, df_meta, cols_map):
    pass


class bioProjsDataloader:
    def __init__(self, rmp=None, metadata=None, load=None):
        self.rmp = rmp
        self.meta_data = metadata
        self.load = load
        
        self.rmp_full_feas = None
        
        # 初始化 rmp 为百分比
        if self.rmp is not None:
            self.rmp = self.rmp.fillna(0)
            self.rmp = self.rmp.div(self.rmp.sum(axis=1), axis=0)
            
    
    def trim_rmp_feas(self, strip_level=""):
        if self.rmp is not None:
            self.rmp_full_feas = self.rmp.columns
            self.rmp.columns = trim_bact_names(self.rmp.columns, strip_level)
        
    def check_index(self):
        """检查 rmp, meta_data 和 load 的索引是否一致，并调整顺序"""
        if self.rmp.index.equals(self.meta_data.index) and self.rmp.index.equals(self.load.index):
            print("✅ 索引一致")
            return self
        else:
            print("❌ 索引不一致，正在调整顺序...")
            
            # 根据 rmp 的索引调整 meta_data 和 load 的顺序
            self.meta_data = self.meta_data.reindex(self.rmp.index)
            self.load = self.load.reindex(self.rmp.index)
    
            # 再次检查索引是否一致
            if self.rmp.index.equals(self.meta_data.index) and self.rmp.index.equals(self.load.index):
                print("✅ 索引已调整一致")
                return self
            else:
                raise ValueError("rmp, meta_data 和 load 的索引仍然不一致，无法调整。")
        
    def merge(self, *others, check_index=True):
        """合并多个 bioProjsDataloader 实例"""
        all_rmp = [self.rmp] + [other.rmp for other in others]
        all_meta = [self.meta_data] + [other.meta_data for other in others]
        all_load = [self.load] + [other.load for other in others]

        # 合并 rmp（按行拼接，列取并集，缺失填0）
        rmp_merged = pd.concat(all_rmp, axis=0, join="outer").fillna(0)
        rmp_merged = rmp_merged.loc[:, sorted(rmp_merged.columns)]
        rmp_merged = rmp_merged.div(rmp_merged.sum(axis=1), axis=0)
        
      

        # 合并 meta_data（列取并集，按行拼接）
        meta_merged = pd.concat(all_meta, axis=0)

        # 合并 load
        load_merged = pd.concat(all_load, axis=0)

        merged_obj = bioProjsDataloader(rmp=rmp_merged, metadata=meta_merged, load=load_merged)
        
        print(rmp_merged.shape, meta_merged.shape, load_merged.shape)
        if check_index:
            merged_obj = merged_obj.check_index()

        return merged_obj

    def __add__(self, other):
        if not isinstance(other, bioProjsDataloader):
            return NotImplemented
        return self.merge(other)
    
    def save(self, file_path):
        """根据后缀保存为 pickle/xlsx/feather.zip"""
        file_path = Path(file_path)
        file_suffixes = file_path.suffixes
        data_dict = {
            "rmp": self.rmp,
            "meta_data": self.meta_data,
            "load": self.load
        }
    
        if '.gz' in file_suffixes and '.pkl' in file_suffixes:
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(data_dict, f, protocol=4)
        elif file_path.suffix == ".pkl":
            with open(file_path, 'wb') as f:
                pickle.dump(data_dict, f, protocol=4)
        elif file_path.suffix in [".xlsx", ".xls"]:
            with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
                self.rmp.to_excel(writer, sheet_name="rmp")
                self.meta_data.to_excel(writer, sheet_name="meta_data")
                self.load.to_excel(writer, sheet_name="load")
        elif ".feather.zip" in str(file_path):
            if not HAS_PYARROW:
                raise ImportError("pyarrow is required for feather format. Install with: pip install pyarrow")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                for name, df in data_dict.items():
                    table = pa.Table.from_pandas(df)
                    feather.write_feather(table, tmpdir / f"{name}.feather", compression="zstd")
                shutil.make_archive(str(file_path).replace(".zip", ""), 'zip', tmpdir)
        else:
            raise ValueError(f"不支持的文件类型: {file_path.suffix}")
    
        print(f"✅ 数据已保存至 {file_path}")
    
    def load_local(self, file_path):
        file_path = Path(file_path)
        file_suffixes = file_path.suffixes
    
        if '.gz' in file_suffixes and '.pkl' in file_suffixes:
            return self.load_from_pickle(file_path)
        elif file_path.suffix == ".pkl":
            return self.load_from_pickle(file_path)
        elif file_path.suffix in [".xlsx", ".xls"]:
            return self.load_from_xls(file_path)
        elif ".feather.zip" in str(file_path):
            if not HAS_PYARROW:
                raise ImportError("pyarrow is required for feather format. Install with: pip install pyarrow")  
            return self.load_from_feather_zip(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path.suffix}")
    
    def load_from_pickle(self, file_path):
        with (gzip.open(file_path, 'rb') if '.gz' in file_path.suffixes else open(file_path, 'rb')) as f:
            data_dict = pickle.load(f)
        self.rmp = data_dict["rmp"]
        self.meta_data = data_dict["meta_data"]
        self.load = data_dict["load"]
        return self
    
    def load_from_feather_zip(self, file_path):
        if not HAS_PYARROW:
            raise ImportError("pyarrow is required for feather format. Install with: pip install pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            shutil.unpack_archive(file_path, tmpdir)
            self.rmp = feather.read_feather(tmpdir / "rmp.feather")
            self.meta_data = feather.read_feather(tmpdir / "meta_data.feather")
            self.load = feather.read_feather(tmpdir / "load.feather")
        return self
    
    def load_from_xls(self, file_path):
        df_dict = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        self.rmp = df_dict["rmp"]
        self.meta_data = df_dict["meta_data"]
        self.load = df_dict["load"]
        return self
    
    
     # 统一切片方法：对三个数据框进行统一的切片
    def slice_data(self, share_slice=None):
         """切片操作，返回一个新的实例"""
         rmp_sliced = self.rmp[share_slice] if self.rmp is not None and share_slice else self.rmp
         meta_data_sliced = self.meta_data[share_slice] if self.meta_data is not None and share_slice else self.meta_data
         load_sliced = self.load[share_slice] if self.load is not None and share_slice else self.load
         return bioProjsDataloader(rmp=rmp_sliced, metadata=meta_data_sliced, load=load_sliced)
     
     # 打印数据的方法：打印 rmp, meta_data 和 load 的信息
    def print_data(self):
         """打印 rmp, meta_data 和 load 数据框的基本信息"""
         print("rmp 数据框概览:")
         print(self.rmp.info())
         print("\nmeta_data 数据框概览:")
         print(self.meta_data.info())
         print("\nload 数据框概览:")
         print(self.load.info())
     
     # 获取数据的方法：便于访问 rmp, meta_data 和 load
    def get_data(self):
         """返回三个数据框作为字典"""
         return {
             "rmp": self.rmp,
             "meta_data": self.meta_data,
             "load": self.load
         }
    def get_qmp(self):
         """计算 qmp = load * rmp"""
         if self.rmp is not None and self.load is not None:
             qmp = self.load.values * self.rmp.values
             return pd.DataFrame(qmp, index=self.rmp.index, columns=self.rmp.columns)
         else:
             raise ValueError("rmp 或 load 数据框为空，无法计算 qmp。")
     
    def clean_zero_rmp(self):
        """
        删除 self.rmp 中所有列和为 0 的特征列。
        返回被删除的列名列表。
        """
        if self.rmp is None:
            raise ValueError("self.rmp 为 None，无法清理。")
        
        col_sums = self.rmp.sum(axis=0)
        zero_sum_cols = col_sums[col_sums == 0].index.tolist()
        
        if zero_sum_cols:
            print(f"🧹 删除了 {len(zero_sum_cols)} 个列和为 0 的特征。")
            self.rmp = self.rmp.drop(columns=zero_sum_cols)
        else:
            print("✅ 所有特征列均有非零值，无需清理。")
        
        return zero_sum_cols  

        
    def filter_by_metadata(self, meta_col="proj_id", values=None, check_index=True):
        """
        根据指定的元数据列提取数据，返回一个新的 bioProjsDataloader 对象
        
        Parameters:
        -----------
        meta_col : str, default "proj_id"
            要基于的元数据列名
        values : list, str, or None, default None
            要提取的特定值。如果为 None，则返回所有唯一值的数据
            如果为 str，则提取该单个值对应的数据
            如果为 list，则提取列表中所有值对应的数据
        check_index : bool, default True
            是否检查返回对象的索引一致性
            
        Returns:
        --------
        bioProjsDataloader
            过滤后的新实例
        """
        if self.meta_data is None:
            raise ValueError("meta_data 为空，无法进行提取操作。")
        
        if meta_col not in self.meta_data.columns:
            raise ValueError(f"元数据中不存在列 '{meta_col}'。可用列: {list(self.meta_data.columns)}")
        
        # 如果没有指定 values，返回所有数据
        if values is None:
            mask = pd.Series(True, index=self.meta_data.index)
            print(f"✅ 提取所有数据，共 {mask.sum()} 个样本")
        else:
            # 确保 values 是列表格式
            if isinstance(values, str):
                values = [values]
            elif not isinstance(values, (list, tuple, set)):
                values = [values]
            
            # 创建布尔掩码
            mask = self.meta_data[meta_col].isin(values)
            
            if mask.sum() == 0:
                print(f"⚠️  警告: 没有找到匹配 {meta_col}={values} 的数据")
                return bioProjsDataloader()
            
            print(f"✅ 基于 {meta_col}={values} 提取了 {mask.sum()} 个样本")
        
        # 提取对应的行
        filtered_indices = self.meta_data[mask].index
        
        # 过滤三个数据框
        rmp_filtered = self.rmp.loc[filtered_indices] if self.rmp is not None else None
        meta_filtered = self.meta_data.loc[filtered_indices] if self.meta_data is not None else None
        load_filtered = self.load.loc[filtered_indices] if self.load is not None else None
        
        # 创建新对象
        new_obj = bioProjsDataloader(rmp=rmp_filtered, metadata=meta_filtered, load=load_filtered)
        
        # 检查索引一致性
        if check_index and new_obj.rmp is not None:
            new_obj = new_obj.check_index()
        
        return new_obj
    
    def get_unique_meta_values(self, meta_col):
        """
        获取指定元数据列的所有唯一值
        
        Parameters:
        -----------
        meta_col : str
            元数据列名
            
        Returns:
        --------
        list
            该列的所有唯一值
        """
        if self.meta_data is None:
            raise ValueError("meta_data 为空。")
        
        if meta_col not in self.meta_data.columns:
            raise ValueError(f"元数据中不存在列 '{meta_col}'。可用列: {list(self.meta_data.columns)}")
        
        unique_vals = self.meta_data[meta_col].unique().tolist()
        print(f"📊 {meta_col} 列的唯一值 ({len(unique_vals)} 个): {unique_vals}")
        return unique_vals
    

        

def find_valid_index(df_index, df1, possible_indexes, print_info=""):
    """
    在 df1 中找到一个有效索引列，并确保该列的值在 df_index 中匹配。

    参数：
        df_index: 主要索引（来自 df.index）。
        df1: 需要查找索引的 DataFrame。
        possible_indexes: 可能的索引列名列表。
        print_info: 额外的提示信息，便于调试。

    返回：
        找到的索引列名（字符串），或者 None（未找到有效索引）。
    """
    for index in possible_indexes:
        if index in df1.columns:
            valid_mask = df1[index].astype(str).isin(df_index)
            num_valid = valid_mask.sum()
            num_total = len(df1)

            if num_valid > 0:
                print(f"✅ {print_info} 发现有效索引列: '{index}' ({num_valid}/{num_total} 个匹配)")

                if num_valid < num_total:
                    missing_values = df1.loc[~valid_mask, index].tolist()
                    print(f"⚠️ {print_info} 警告: {num_total - num_valid} 个值不在 df_index 中")
                    print(f"❌ {print_info} 这些值未匹配:", missing_values[:10], "...")  # 只打印前10个

                return index  # 只返回找到的索引列名

    print(f"❌ {print_info}: 未找到有效的索引列，可能需要手动检查 df1")
    return None

def find_meta_prior1(df, rules, print_info="", ignore_up_low=True): 
    """
    - **优先匹配 `rules` 里靠前的字段**
    - **完整匹配优先，失败后再部分匹配**
    - **排除全 NaN/None 的列**
    - **返回匹配的列名及统计信息**
    """
    results = {}

    for std_name, possible_cols in rules.items():
        matched_col = None

        # **（1）按规则优先级排序 df.columns**
        col_priority = sorted(df.columns, key=lambda x: (
            possible_cols.index(x) if x in possible_cols else len(possible_cols)
        ))

        # **（2）完整匹配**
        for col in col_priority:
            col_cmp = col.lower() if ignore_up_low else col
            if col_cmp in [p.lower() for p in possible_cols]:
                matched_col = col
                break  # **匹配到就立刻退出**

        # **（3）如果完整匹配失败，尝试部分匹配**
        if not matched_col:
            for col in col_priority:
                col_cmp = col.lower() if ignore_up_low else col
                col_parts = re.split(r"[_\-\s]+", col_cmp)
                if any(keyword.lower() in col_parts for keyword in possible_cols):
                    matched_col = col
                    break  # **匹配到就立刻退出**

        # **（4）检查列是否全是 NaN/None，如果是则视为未找到**
        if matched_col and df[matched_col].dropna().empty:
            print(f"❌ {print_info} 找到 '{std_name}' 但全是 NaN/None，不算有效匹配")
            matched_col = None

        # **（5）记录匹配结果**
        if matched_col:
            print(f"✅ {print_info} 找到 '{std_name}' 列: '{matched_col}'")

            # **（5.1）数值型列**
            if df[matched_col].dtype.kind in "if":  # `i` = 整数，`f` = 浮点数
                abs_info = {
                    "mean": round(df[matched_col].mean(), 2),
                    "len": len(df[matched_col])
                }
                print(f"📊 {print_info} {std_name} 平均值: {abs_info['mean']}, 总数: {abs_info['len']}")

            # **（5.2）字符型列**
            else:
                category_counts = df[matched_col].astype(str).str.lower().value_counts()
                unique_count = category_counts.shape[0]  # 统计 unique 值的总数

                # 只保存前 5 个类别，避免输出过长
                abs_info = {
                    "unique_count": unique_count,
                    "top_categories": category_counts.head(5).to_dict()
                }

                print(f"📊 {print_info} {std_name} 统计: {abs_info['top_categories']} (共 {unique_count} 类)")

        else:
            print(f"❌ {print_info} 未找到 '{std_name}' 列")
            abs_info = None  # 没有找到时，返回 None

        results[std_name] = (matched_col if matched_col else "None", abs_info)

    return results


def get_interested_df(df_rmp_raw, df_qmp_raw, df_metadata_raw, valid_index_name, selected_meta_columns):
    """
    获取三个数据框的交集并提取感兴趣的元数据列
    
    参数：
    - df_rmp_raw: 相对丰度数据框
    - df_qmp_raw: 绝对丰度数据框  
    - df_metadata_raw: 元数据框
    - valid_index_name: 元数据中有效的索引列名
    - selected_meta_columns: find_meta_prior1 返回的列映射字典
    
    返回：
    - tuple: (df_metadata_filtered, df_rmp_filtered, df_qmp_filtered, common_samples)
    """
    
    # 获取各数据框的样本ID
    rmp_samples = set(df_rmp_raw.index.astype(str))
    qmp_samples = set(df_qmp_raw.index.astype(str))
    
    # 将元数据的指定列设为索引，并转为字符串
    df_metadata_indexed = df_metadata_raw.set_index(valid_index_name)
    df_metadata_indexed.index = df_metadata_indexed.index.astype(str)
    meta_samples = set(df_metadata_indexed.index)
    
    # 计算交集
    common_samples = rmp_samples & qmp_samples & meta_samples
    common_samples = sorted(list(common_samples))
    
    print(f"样本数量统计:")
    print(f"   - df_rmp: {len(rmp_samples)}, df_qmp: {len(qmp_samples)}, df_meta: {len(meta_samples)}")
    print(f"   - 交集: {len(common_samples)} 个样本")
    
    if len(common_samples) == 0:
        print("警告: 没有找到共同的样本!")
        return None, None, None, []
    
    # 按交集筛选数据
    df_rmp_filtered = df_rmp_raw.loc[common_samples]
    df_qmp_filtered = df_qmp_raw.loc[common_samples]  
    df_metadata_intersected = df_metadata_indexed.loc[common_samples]
    
    # 提取感兴趣的元数据列
    existing_cols = {}
    for std_name, (col_name, _) in selected_meta_columns.items():
        if col_name != "None" and col_name in df_metadata_intersected.columns:
            existing_cols[col_name] = std_name
    
    # 创建结果DataFrame
    df_metadata_filtered = df_metadata_intersected[list(existing_cols.keys())].copy()
    
    # 重命名列为标准名称
    df_metadata_filtered = df_metadata_filtered.rename(columns=existing_cols)
    
    # 添加缺失的列（填充为None）
    for std_name, (col_name, _) in selected_meta_columns.items():
        if std_name not in df_metadata_filtered.columns:
            df_metadata_filtered[std_name] = None
    
    return df_metadata_filtered, df_rmp_filtered, df_qmp_filtered, common_samples



def print_unique_values(df, n=5):
    """
    打印 DataFrame 每列的唯一值（最多 n 个）。
    
    参数:
    df: pd.DataFrame - 输入的 DataFrame
    n: int - 每列最多显示的唯一值数量（默认 5）
    """
    for col in df.columns:
        unique_values = df[col].dropna().unique()  # 去除 NaN 并获取唯一值
        n_unique = len(unique_values)
        sample_values = unique_values[:n]  # 最多显示 n 个 
        print(f"🔹 列 '{col}' 有 {n_unique} 个唯一值: {sample_values}")
        
        






