#!/usr/bin/env python3
"""
MetaPhlAn & Kraken2 & Sylph & HUMAnN3 LSF-Chunk 工业级流水线 (v5.2)
包含 HUMAnN3 2025+ 版数据库表头欺骗补丁 (无损拷贝版)
"""

import os
import sys
import yaml
import re
import shutil
import subprocess
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 👑 自动版本提取函数 =================
def extract_mp4_version(db_path_str):
    """
    从 /share/.../metaphlan4_database_v202503 中自动提取 v202503
    支持匹配 v2023, vOct22, vJan21 等各种格式
    """
    if not db_path_str: return "v_unknown"
    dir_name = Path(db_path_str).name
    match = re.search(r'v[0-9a-zA-Z]+', dir_name)
    return match.group(0) if match else "v_custom"

class ChunkPipeline:
    def __init__(self, config):
        self.conf = config
        self.work_dir = Path(self.conf['work_dir'])
        self.project_name = self.work_dir.parent.name 
        
        self.active_profilers = self.conf.get('run_profilers', [])

        self.dirs = {
            'qc': self.work_dir / '01_qc',
            'host': self.work_dir / '02_rmhost',
            'tmp': self.work_dir / 'tmp'
        }
        
        if 'metaphlan' in self.active_profilers: self.dirs['metaphlan'] = self.work_dir / '03_metaphlan'
        if 'kraken2' in self.active_profilers: 
            self.dirs['kraken2'] = self.work_dir / '04_kraken2'
            self.dirs['bracken'] = self.work_dir / '05_bracken'
        if 'sylph' in self.active_profilers: self.dirs['sylph'] = self.work_dir / '07_sylph'
        if 'humann' in self.active_profilers: self.dirs['humann'] = self.work_dir / '09_humann'

        for d in self.dirs.values(): d.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.work_dir / "chunk_pipeline.log"
        self.logger = logging.getLogger(self.project_name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def log(self, sample_id, step, status, msg=""):
        content = f"[{sample_id} - {step}] {status}"
        if msg: content += f" | {msg}"
        if status == "ERROR": self.logger.error(content)
        elif status == "WARN": self.logger.warning(content)
        else: self.logger.info(content)

    def run_cmd(self, cmd, env='main'):
        env_map = {
            'main': self.conf.get('env_main', 'zwskinbact'),
            'kraken': self.conf.get('env_kraken', 'kraken2'),
            'sylph': self.conf.get('env_sylph', 'sylph'),
            'metaphlan': self.conf.get('env_metaphlan', 'humann4_env'), # 👑 新增映射
            'humann': self.conf.get('env_humann', 'humann4_env')
        }
        target_env = env_map.get(env, env_map['main'])
        conda_sh = self.conf.get('conda_sh_path', '/mnt/raid6/longzhengwu/anaconda3/etc/profile.d/conda.sh')

        script_content = f"#!/bin/bash\nsource {conda_sh}\nconda activate {target_env}\nset -e\n{cmd}\n"
        tmp_sh = self.dirs['tmp'] / f"run_{os.getpid()}_{np.random.randint(100000)}.sh"
        
        with open(tmp_sh, 'w') as f: f.write(script_content)

        try:
            subprocess.run(f"bash {tmp_sh}", shell=True, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            raise RuntimeError(f"\n[FAILED COMMAND] => {cmd}\n[STDERR] => {error_msg}")
        finally:
            if tmp_sh.exists(): tmp_sh.unlink()

    # ================= 任务单元 =================

    def task_qc(self, sample_id, r1, r2):
        step = "QC"
        qc_r1, qc_r2 = self.dirs['qc'] / f"{sample_id}_clean_1.fq.gz", self.dirs['qc'] / f"{sample_id}_clean_2.fq.gz"
        try:
            if qc_r1.exists() and qc_r1.stat().st_size > 1024: return True
            th = self.conf.get('threads', {}).get('qc', 6)
            self.run_cmd(f"fastp -i {r1} -I {r2} -o {qc_r1} -O {qc_r2} -w {th} --json {self.dirs['qc']}/{sample_id}.json --html {self.dirs['qc']}/{sample_id}.html", env='main')
            self.log(sample_id, step, "DONE")
            return True
        except Exception as e:
            self.log(sample_id, step, "ERROR", str(e))
            return False

    def task_host(self, sample_id):
        step = "RMHost"
        nh_r1, nh_r2 = self.dirs['host'] / f"{sample_id}_nohost_1.fq.gz", self.dirs['host'] / f"{sample_id}_nohost_2.fq.gz"
        qc_r1, qc_r2 = self.dirs['qc'] / f"{sample_id}_clean_1.fq.gz", self.dirs['qc'] / f"{sample_id}_clean_2.fq.gz"
        try:
            if nh_r1.exists() and nh_r1.stat().st_size > 1024: return True
            if not qc_r1.exists(): return False
            th = self.conf.get('threads', {}).get('bowtie', 12)
            temp_prefix = self.dirs['host'] / f"{sample_id}_temp"
            self.run_cmd(f"bowtie2 -p {th} -x {self.conf['host_index']} -1 {qc_r1} -2 {qc_r2} --un-conc-gz {temp_prefix} -S /dev/null", env='main')
            if Path(f"{temp_prefix}.1").exists():
                shutil.move(f"{temp_prefix}.1", nh_r1)
                shutil.move(f"{temp_prefix}.2", nh_r2)
            self.log(sample_id, step, "DONE")
            return True
        except Exception as e:
            self.log(sample_id, step, "ERROR", str(e))
            return False

    def task_metaphlan(self, sample_id):
        step, th = "MetaPhlAn", self.conf.get('threads', {}).get('metaphlan', 8)
        
        # 👑 提取版本标签
        v_tag = extract_mp4_version(self.conf.get('mp4db_dir', ''))
        
        out = self.dirs['metaphlan'] / f"{sample_id}_profile_{v_tag}.tsv"
        bow = self.dirs['metaphlan'] / f"{sample_id}_bowtie2.bz2"
        nh_r1 = self.dirs['host'] / f"{sample_id}_nohost_1.fq.gz"
        nh_r2 = self.dirs['host'] / f"{sample_id}_nohost_2.fq.gz"
        
        try:
            if out.exists() and out.stat().st_size > 0: return True
            if not nh_r1.exists(): return False
            
            # 核心控制逻辑
            mp_ver = str(self.conf.get('metaphlan_ver', '422')) 
            extra_params = self.conf.get('mp4_extra_params', '')
            
            # 👑 动态构建 -x 参数（如果 yaml 里写了 mp4_index 就加上）
            mp_index = self.conf.get('mp4_index', '')
            index_str = f"-x {mp_index} " if mp_index else ""
            
            if mp_ver.startswith('41'): 
                # MetaPhlAn 4.1.1 语法
                cmd = (
                    f"metaphlan {nh_r1},{nh_r2} --input_type fastq "
                    f"{index_str}"              # 👑 显式指定索引
                    f"-t rel_ab_w_read_stats "  
                    f"--bowtie2db {self.conf['mp4db_dir']} --bowtie2out {bow} "
                    f"--nproc {th} -o {out} {extra_params}"
                )
            else: 
                # MetaPhlAn 4.2.x 语法
                cmd = (
                    f"metaphlan {nh_r1},{nh_r2} --input_type fastq "
                    f"{index_str}"              # 👑 显式指定索引
                    f"-t rel_ab_w_read_stats "  
                    f"--db_dir {self.conf['mp4db_dir']} --mapout {bow} "
                    f"--nproc {th} -o {out} {extra_params}"
                )
            
            # 提交到专属的 humann4_env 环境运行
            self.run_cmd(cmd, env='metaphlan') 
            
            # 清理中间比对文件
            if bow.exists(): bow.unlink()
            
            self.log(sample_id, step, "DONE")
            return True
        except Exception as e:
            self.log(sample_id, step, "ERROR", str(e))
            return False

    def task_kraken(self, sample_id):
        step, th = "Kraken2", self.conf.get('threads', {}).get('kraken', 12)
        out, rep = self.dirs['kraken2'] / f"{sample_id}.kraken2.output", self.dirs['kraken2'] / f"{sample_id}.kraken2.report"
        nh_r1, nh_r2 = self.dirs['host'] / f"{sample_id}_nohost_1.fq.gz", self.dirs['host'] / f"{sample_id}_nohost_2.fq.gz"
        try:
            if rep.exists() and rep.stat().st_size > 0: return True
            if not nh_r1.exists(): return False
            self.run_cmd(f"kraken2 --db {self.conf['kraken2_db']} --threads {th} --paired {nh_r1} {nh_r2} --output {out} --report {rep} --use-names {self.conf.get('kraken_extra_params', '')}", env='kraken')
            self.log(sample_id, step, "DONE")
            return True
        except Exception as e:
            self.log(sample_id, step, "ERROR", str(e))
            return False

    def task_bracken(self, sample_id):
        step = "Bracken"
        out = self.dirs['bracken'] / f"{sample_id}_bracken_species.tsv"
        rep = self.dirs['kraken2'] / f"{sample_id}.kraken2.report"
        try:
            if out.exists() and out.stat().st_size > 0: return True
            if not rep.exists(): return False
            self.run_cmd(f"bracken -d {self.conf['kraken2_db']} -i {rep} -o {out} -l S {self.conf.get('bracken_extra_params', '')}", env='kraken')
            self.log(sample_id, step, "DONE")
            return True
        except Exception as e:
            self.log(sample_id, step, "ERROR", str(e))
            return False

    def task_sylph(self, sample_id):
        step, th = "Sylph", self.conf.get('threads', {}).get('sylph', 16)
        nh_r1 = self.dirs['host'] / f"{sample_id}_nohost_1.fq.gz"
        
        def _run(db_key, d_name, suffix):
            t_dir = self.work_dir / d_name
            t_dir.mkdir(parents=True, exist_ok=True)
            out = t_dir / f"{sample_id}_{suffix}.tsv"
            db = self.conf.get(db_key, '').replace('\n', ' ').strip()
            if not db or not nh_r1.exists() or (out.exists() and out.stat().st_size > 0): return
            self.run_cmd(f"sylph profile {db} {nh_r1} -t {th} -u -o {out}", env='sylph')

        try:
            _run('sylph_db', '07_sylph', 'sylph')
            if self.conf.get('run_sylph_fungi_only'): _run('sylph_db_fungi', '08_sylph_fungi', 'sylph_fungi')
            self.log(sample_id, step, "DONE")
            return True
        except Exception as e:
            self.log(sample_id, step, "ERROR", str(e))
            return False
    # ================= 👑 HUMAnN3 表头欺骗任务单元 =================
    def task_humann(self, sample_id):
        step = "HUMAnN"
        sample_out = self.dirs['humann'] / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)

        gf_out = sample_out / f"{sample_id}_humann_genefamilies.tsv"
        if gf_out.exists() and gf_out.stat().st_size > 0: return True

        nh_r1 = self.dirs['host'] / f"{sample_id}_nohost_1.fq.gz"
        nh_r2 = self.dirs['host'] / f"{sample_id}_nohost_2.fq.gz"
        
        # 精准捕获 MP4 的表
        v_tag = extract_mp4_version(self.conf.get('mp4db_dir', ''))
        mp4_profile = self.dirs['metaphlan'] / f"{sample_id}_profile_{v_tag}.tsv"

        if not nh_r1.exists() or not mp4_profile.exists(): return False

        merged_fq = sample_out / f"{sample_id}_merged.fq.gz"

        try:
            th = self.conf.get('threads', {}).get('humann', 12)
            subprocess.run(f"cat {nh_r1} {nh_r2} > {merged_fq}", shell=True, check=True)

            nuc_db = self.conf['humann_db_nuc']
            prot_db = self.conf['humann_db_prot']
            
            # 彻底原生调用！没有任何补丁和 Hack
            cmd = (
                f"humann --input {merged_fq} --output {sample_out} --threads {th} "
                f"--taxonomic-profile {mp4_profile} --nucleotide-database {nuc_db} "
                f"--protein-database {prot_db} --output-basename {sample_id}_humann"
            )
            
            self.run_cmd(cmd, env='humann')
            self.log(sample_id, step, "DONE")
            return True
        except Exception as e:
            self.log(sample_id, step, "ERROR", str(e))
            return False
        finally:
            if merged_fq.exists(): merged_fq.unlink()

    def run_stage(self, stage_name, task_func, sample_list, max_workers, *args):
        if not sample_list: return []
        self.logger.info(f"{'='*10} Stage Start: {stage_name} (Workers: {max_workers}, Total: {len(sample_list)}) {'='*10}")
        success_samples = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {}
            for s in sample_list:
                future = executor.submit(task_func, s[0], s[1], s[2]) if args else executor.submit(task_func, s)
                future_to_sample[future] = s[0] if args else s

            for future in as_completed(future_to_sample):
                sid = future_to_sample[future]
                try:
                    if future.result(): success_samples.append(sid)
                except Exception as e:
                    self.logger.error(f"System Error in {sid}: {e}")
        return success_samples

    def cleanup_space(self, valid_ids):
        if not self.conf.get('auto_cleanup_qc', True): return
        for sid in valid_ids:
            for f in [self.dirs['qc'] / f"{sid}_clean_1.fq.gz", self.dirs['qc'] / f"{sid}_clean_2.fq.gz"]:
                if f.exists(): f.unlink()
        if self.dirs['tmp'].exists(): shutil.rmtree(self.dirs['tmp'])

    def find_samples(self):
        samples = []
        pattern = re.compile(r"(.+?)(_R1|_1|_01)(\.fastq|\.fq)(\.gz)?$")
        for d_str in self.conf.get('input_dirs', []):
            d_path = Path(d_str)
            if not d_path.exists(): continue
            for f in d_path.iterdir():
                match = pattern.match(f.name)
                if match:
                    prefix, tag, ext, gz = match.groups()
                    r2_path = d_path / f"{prefix}{tag.replace('1', '2')}{ext}{gz or ''}"
                    if r2_path.exists(): samples.append((prefix, f, r2_path))
        return samples

    def run(self):
        all_samples = self.find_samples()
        if not all_samples: return

        total = self.conf.get('total_node_cores', 52)
        th = self.conf.get('threads', {})
        
        valid_ids = self.run_stage('QC', self.task_qc, all_samples, max(1, total // th.get('qc', 8)), True)
        valid_ids = self.run_stage('Host_Removal', self.task_host, valid_ids, max(1, total // th.get('bowtie', 12)))

        if 'metaphlan' in self.active_profilers:
            self.run_stage('MetaPhlAn', self.task_metaphlan, valid_ids, max(1, total // th.get('metaphlan', 16)))
        if 'kraken2' in self.active_profilers:
            kraken_ids = self.run_stage('Kraken2', self.task_kraken, valid_ids, max(1, total // th.get('kraken', 16)))
            self.run_stage('Bracken', self.task_bracken, kraken_ids, max(1, total // 4)) 
        if 'sylph' in self.active_profilers:
            self.run_stage('Sylph', self.task_sylph, valid_ids, max(1, total // th.get('sylph', 16)))

        if 'humann' in self.active_profilers:
            self.run_stage('HUMAnN3_Profiling', self.task_humann, valid_ids, max(1, total // th.get('humann', 12)))

        self.cleanup_space(valid_ids)


# ================= 👑 全局跨 Chunk 大表合龙与高级后处理 =================
def run_global_merge(conf, base_work_dir):
    base_dir = Path(base_work_dir)
    print(f"[*] 🚀 启动全局大表合并模式，扫描主根目录: {base_dir}")
    
    out_dir = base_dir / "00_Merged_Profiling_Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if 'metaphlan' in conf.get('pipeline_base_config', {}).get('run_profilers', []):
        mpa_files = list(base_dir.rglob("work/03_metaphlan/*_profile.tsv"))
        if mpa_files:
            merged_mpa = out_dir / f"Global_metaphlan_merged.tsv"
            cmd = f"merge_metaphlan_tables.py {' '.join([str(p) for p in mpa_files])} > {merged_mpa}"
            subprocess.run(f"conda run -n {conf['pipeline_base_config']['env_main']} {cmd}", shell=True, check=True)
            print(f"[+] ✅ MetaPhlAn 全局合并完成: {merged_mpa}")
            
    if 'kraken2' in conf.get('pipeline_base_config', {}).get('run_profilers', []):
        bra_files = list(base_dir.rglob("work/05_bracken/*_bracken_species.tsv"))
        if bra_files:
            dfs = []
            for tsv in bra_files:
                sample_name = tsv.name.replace("_bracken_species.tsv", "")
                try:
                    df = pd.read_csv(tsv, sep='\t')
                    df = df[['name', 'fraction_total_reads']].rename(columns={'fraction_total_reads': sample_name})
                    df.set_index('name', inplace=True)
                    dfs.append(df)
                except: pass
            if dfs:
                final_bracken = pd.concat(dfs, axis=1).fillna(0)
                merged_bracken = out_dir / "Global_bracken_merged.csv"
                final_bracken.to_csv(merged_bracken)
                print(f"[+] ✅ Bracken 全局合并完成: {merged_bracken}")

    if 'sylph' in conf.get('pipeline_base_config', {}).get('run_profilers', []):
        sylph_files = list(base_dir.rglob("work/07_sylph/*_sylph.tsv"))
        if sylph_files:
            tax_db = conf['pipeline_base_config'].get('sylph_tax_db', '').replace('\n', ' ').strip()
            if tax_db:
                mpa_dir = out_dir / "sylph_tmp_mpa"
                mpa_dir.mkdir(exist_ok=True)
                cmd_tax = f"sylph-tax taxprof {' '.join([str(p) for p in sylph_files])} -t {tax_db} -o {mpa_dir}/tax_ --overwrite"
                subprocess.run(f"conda run -n {conf['pipeline_base_config']['env_sylph']} {cmd_tax}", shell=True, check=True)
                mpa_files = list(mpa_dir.glob("*.sylphmpa"))
                if mpa_files:
                    merged_sylph = out_dir / "Global_sylph_merged.tsv"
                    cmd_merge = f"sylph-tax merge -o {merged_sylph} --column relative_abundance {' '.join([str(p) for p in mpa_files])}"
                    subprocess.run(f"conda run -n {conf['pipeline_base_config']['env_sylph']} {cmd_merge}", shell=True, check=True)
                    
                    df = pd.read_csv(merged_sylph, sep='\t')
                    df.columns = ['clade_name'] + [re.sub(r'_nohost_.*$', '', os.path.basename(col)) for col in df.columns if col != 'clade_name']
                    df.to_csv(merged_sylph, sep='\t', index=False)
                    shutil.rmtree(mpa_dir)
                    print(f"[+] ✅ Sylph 全局合并完成: {merged_sylph}")

    if 'humann' in conf.get('pipeline_base_config', {}).get('run_profilers', []):
        print("\n[*] ======= 开始提取和处理 HUMAnN3 全局结果 =======")
        hm_out_dir = out_dir / "05_humann_global_results"
        hm_out_dir.mkdir(parents=True, exist_ok=True)
        
        env_humann = conf['pipeline_base_config']['env_humann']
        prefix = hm_out_dir / "Global_humann"
        
        # [Step 1] 合并基础的三大表
        table_types = {"genefamilies": "humann_genefamilies.tsv", 
                       "pathabundance": "humann_pathabundance.tsv", 
                       "pathcoverage": "humann_pathcoverage.tsv"}
        
        for name, suffix in table_types.items():
            out_file = f"{prefix}_{name}.tsv"
            if not os.path.exists(out_file):
                print(f"[1/5] 正在合并 {name} ...")
                cmd_join = f"humann_join_tables --input {base_dir} --output {out_file} --file_name {suffix} --search-subdirectories"
                subprocess.run(f"conda run -n {env_humann} {cmd_join}", shell=True, check=True)

        # [Step 2] 转换为相对丰度 (Relab)
        print("[2/5] 正在转换为相对丰度 (Relab)...")
        for ttype in ["genefamilies", "pathabundance"]:
            cmd_norm = f"humann_renorm_table --input {prefix}_{ttype}.tsv --output {prefix}_{ttype}_relab.tsv --units relab"
            subprocess.run(f"conda run -n {env_humann} {cmd_norm}", shell=True, check=True)

        # [Step 3] 转换为 EC 并归一化为 CPM
        print("[3/5] 正在执行 EC 酶学转换并计算 CPM...")
        cmd_ec = f"humann_regroup_table --input {prefix}_genefamilies.tsv --groups uniref90_level4ec --output {prefix}_genefamilies_ec.tsv"
        cmd_ec_cpm = f"humann_renorm_table --input {prefix}_genefamilies_ec.tsv --units cpm --output {prefix}_genefamilies_ec_cpm.tsv"
        subprocess.run(f"conda run -n {env_humann} {cmd_ec}", shell=True, check=True)
        subprocess.run(f"conda run -n {env_humann} {cmd_ec_cpm}", shell=True, check=True)

        # [Step 4] 结果打包
        print("[4/5] 正在打包核心结果文件...")
        tar_name = hm_out_dir / "humann_result_files.tar.gz"
        files_to_pack = [
            f"Global_humann_genefamilies_relab.tsv",
            f"Global_humann_pathabundance_relab.tsv",
            f"Global_humann_pathcoverage.tsv",
            f"Global_humann_genefamilies_ec_cpm.tsv"
        ]
        
        current_dir = os.getcwd()
        os.chdir(hm_out_dir)
        try:
            cmd_tar = ["tar", "-czf", tar_name.name] + files_to_pack
            subprocess.run(cmd_tar, check=True)
            print(f"✅ 文件打包完成: {tar_name}")
        finally:
            os.chdir(current_dir)

        # [Step 5] 智能清理冗余巨大中间表
        if conf['pipeline_base_config'].get('clean_extra_humann_files', True):
            print("[5/5] 正在清理庞大的临时原始合表数据...")
            for f in [f"{prefix}_genefamilies.tsv", f"{prefix}_pathabundance.tsv", f"{prefix}_genefamilies_ec.tsv"]:
                if os.path.exists(f): os.remove(f)

        print("[*] HUMAnN3 全局大表处理完毕！")


def main():
    parser = argparse.ArgumentParser(description="Chunked Profiling Framework")
    parser.add_argument('config', help="Config YAML file")
    parser.add_argument('--merge', action='store_true', help="跨 Chunk 执行大表合并与数据清洗")
    args = parser.parse_args()

    with open(args.config) as f: conf = yaml.safe_load(f)

    if args.merge:
        base_work_dir = Path(conf['base_work_dir']) / conf['project_name']
        run_global_merge(conf, base_work_dir)
    else:
        ChunkPipeline(conf).run()

if __name__ == "__main__":
    main()
