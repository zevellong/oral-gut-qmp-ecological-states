#!/usr/bin/env python3
import os
import re
import sys
import yaml
import json
import shutil
import collections.abc
import copy
from pathlib import Path
import argparse

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class UnifiedDispatcher:
    def __init__(self, d_conf, force_overwrite=False, flush_tail=False):
        self.d_conf = d_conf
        self.proj_name = self.d_conf['project_name']
        self.p_type = self.d_conf.get('pipeline_type', 'ngs').lower()
        
        self.mode = 'greedy' 
        self.scheduler = self.d_conf.get('scheduler', 'lsf').lower()
        self.target_merge = self.d_conf.get('target_merge', False)

        self.base_work_dir = Path(self.d_conf['base_work_dir']) / self.proj_name
        self.chunk_size = self.d_conf.get('chunk_size', 8)
        self.use_symlink = self.d_conf.get('use_symlink', True)
        self.exclude_dirs = [Path(d).absolute() for d in (self.d_conf.get('exclude_dirs') or [])]

        # 👑 修复点 1：直接使用配置文件中的真实节点名作为 Worker
        old_nodes = self.d_conf.get('node_list', [])
        if old_nodes:
            self.worker_list = old_nodes
            self.max_workers = len(old_nodes)
        else:
            self.max_workers = self.d_conf.get('max_workers', 5)
            self.worker_list = [f"Worker_{i:02d}" for i in range(1, self.max_workers + 1)]

        self.script_out_dir = self.base_work_dir / "00_lsf_submit_scripts"
        self.logs_dir = self.base_work_dir / "logs"
        self.cache_file = self.base_work_dir / ".chunk_allocation_cache.json"

        self.samples = {}
        self.force_overwrite = force_overwrite
        self.flush_tail = flush_tail

    def _load_cache(self):
        default_cache = {'assigned_samples': [], 'buffer': [], 'chunks': {}, 'next_chunk_idx': 1}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f: return json.load(f)
            except Exception: pass
        return default_cache

    def _save_cache(self, cache_data):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=4, ensure_ascii=False)

    def _compute_incremental_chunks(self):
        cache = self._load_cache()
        new_samples = [s for s in sorted(self.samples.keys())
                       if s not in cache['assigned_samples'] and s not in cache['buffer']]

        if new_samples:
            cache['buffer'].extend(new_samples)
            print(f"[+] 新增样本入池: {len(new_samples)} 个 (当前缓冲池: {len(cache['buffer'])} 个)")

        chunks_created = 0
        while len(cache['buffer']) >= self.chunk_size:
            chunk_samples = cache['buffer'][:self.chunk_size]
            cache['buffer'] = cache['buffer'][self.chunk_size:]
            cid = f"chunk_{cache['next_chunk_idx']:03d}"

            cache['chunks'][cid] = {'samples': chunk_samples}
            cache['assigned_samples'].extend(chunk_samples)
            cache['next_chunk_idx'] += 1
            chunks_created += 1

        if self.flush_tail and len(cache['buffer']) > 0:
            print(f"[!] 强制泄洪：剩余 {len(cache['buffer'])} 个样本打包装车。")
            chunk_samples = cache['buffer']
            cache['buffer'] = []
            cid = f"chunk_{cache['next_chunk_idx']:03d}"
            cache['chunks'][cid] = {'samples': chunk_samples}
            cache['assigned_samples'].extend(chunk_samples)
            cache['next_chunk_idx'] += 1
            chunks_created += 1

        self._save_cache(cache)
        return cache['chunks']

    def _get_chunk_status(self, chunk_dir):
        if (chunk_dir / ".state_completed").exists(): return "completed"
        if (chunk_dir / ".state_failed").exists(): return "failed"
        if (chunk_dir / ".state_running").exists(): return "running"
        if (chunk_dir / ".state_submitted").exists(): return "submitted"
        return "pending"

    def _wrap_with_state_markers(self, chunk_dir, core_command):
        return f"""
rm -f "{chunk_dir}/.state_completed" "{chunk_dir}/.state_failed" "{chunk_dir}/.state_submitted"
touch "{chunk_dir}/.state_running"

if {core_command}; then
    rm -f "{chunk_dir}/.state_running"
    touch "{chunk_dir}/.state_completed"
else
    rm -f "{chunk_dir}/.state_running"
    touch "{chunk_dir}/.state_failed"
    exit 1
fi
"""

    def _build_pipeline_config(self, out_yaml_path, input_dirs, work_dir, node_config):
        conf = copy.deepcopy(self.d_conf.get('pipeline_base_config', {}))
        conf['input_dirs'] = input_dirs
        conf['work_dir'] = str(work_dir)
        conf['total_node_cores'] = node_config.get('lsf_cores', 52)
        if 'threads' not in conf: conf['threads'] = {}
        deep_update(conf['threads'], node_config.get('threads', {}))

        if 'methy_config' in self.d_conf:
            conf['methy_config'] = self.d_conf['methy_config']

        out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_yaml_path, 'w') as f: yaml.dump(conf, f, default_flow_style=False)
        return out_yaml_path

    def _build_lsf_script(self, sh_path, job_name, target_hosts, node_config, run_commands):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        sh_path.parent.mkdir(parents=True, exist_ok=True)

        base_conf = self.d_conf.get('pipeline_base_config', {})
        
        module_cmd = base_conf.get('module_cmd', 'module load apps/Miniconda3')
        conda_path = base_conf.get('conda_sh_path', '/public/software/apps/Miniconda3/etc/profile.d/conda.sh')
        conda_env = base_conf.get('conda_env', 'zwml')
        
        init_block = f"{module_cmd}\n" if module_cmd else ""
        init_block += f'CONDA_SH="{conda_path}"\n'
        init_block += 'if [ -f "$CONDA_SH" ]; then . "$CONDA_SH"; fi\n'
        if conda_env:
            init_block += f'conda activate {conda_env}\n'
        
        cores = node_config.get('lsf_cores', 24)
        mem_raw = str(node_config.get('lsf_mem', '250G'))
        partition = node_config.get('partition', 'normal')
        req_nodes = node_config.get('nodes', 1) 

        if self.scheduler == 'slurm':
            mem_slurm = mem_raw.replace('GB', 'G').replace('MB', 'M')
            content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={self.logs_dir}/{job_name}_%j.out
#SBATCH --error={self.logs_dir}/{job_name}_%j.err
#SBATCH --nodes={req_nodes}
#SBATCH --ntasks={cores}
#SBATCH --mem={mem_slurm}
#SBATCH --partition={partition}

echo "=== Slurm Worker Start ==="
echo "Virtual Worker ID: {job_name}"
echo "Target Partition: {partition} (Requested Nodes: {req_nodes})"
echo "Real Allocated Node: $(hostname)"
echo "Time: $(date)"

{init_block}

{run_commands}

echo "=== Slurm Worker Complete ==="
echo "Time: $(date)"
"""
        else:
            # 👑 修复点 2：只要不是虚拟名字，就强制挂载指定的节点，并加入 span[hosts=1]
            host_directive = f'#BSUB -m "{target_hosts}"' if "Worker_" not in target_hosts else ""
            content = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {self.logs_dir}/{job_name}_%J.out
#BSUB -e {self.logs_dir}/{job_name}_%J.err
#BSUB -n {cores}
#BSUB -R "rusage[mem={mem_raw}] span[hosts=1]"
{host_directive}

echo "=== LSF Worker Start ==="
echo "Target Node: {target_hosts}"
echo "Real Host Node: $(hostname)"
echo "Time: $(date)"

{init_block}

{run_commands}

echo "=== LSF Worker Complete ==="
echo "Time: $(date)"
"""
        with open(sh_path, 'w') as f: f.write(content)
        return sh_path

    def _dispatch_greedy(self, chunk_dict, node_defaults, node_overrides):
        node_chunk_map = {w: [] for w in self.worker_list}
        node_idx = 0
        for chunk_id, info in chunk_dict.items():
            assigned_worker = self.worker_list[node_idx % len(self.worker_list)]
            node_idx += 1
            status = self._get_chunk_status(self.base_work_dir / f"{self.proj_name}_{chunk_id}")
            node_chunk_map[assigned_worker].append((chunk_id, info['samples'], status))

        self._print_topology(node_chunk_map)
        generated_scripts = []

        for worker_name, assigned_chunks in node_chunk_map.items():
            if not assigned_chunks: continue
            
            node_config = copy.deepcopy(node_defaults)
            if worker_name in node_overrides: deep_update(node_config, node_overrides[worker_name])

            python_commands = ""
            chunk_dirs = []
            all_completed = True
            
            for chunk_id, chunk_samples, status in assigned_chunks:
                if status != 'completed': all_completed = False
                chunk_work_dir = self.base_work_dir / f"{self.proj_name}_{chunk_id}"
                chunk_dirs.append(chunk_work_dir)

                isolated_input_dir = chunk_work_dir / "00_input_links"
                self._create_symlinks(chunk_samples, isolated_input_dir)

                conf_yaml = chunk_work_dir / "00_configs" / f"{chunk_id}_config.yaml"
                self._build_pipeline_config(conf_yaml, [str(isolated_input_dir)], chunk_work_dir / "work", node_config)

                raw_python_cmd = f"python {self.d_conf['pipeline_script']} {conf_yaml}"
                python_commands += f"\n\necho '>>> 🚀 Worker {worker_name} 开始批次执行: {chunk_id} <<<'\n"
                python_commands += self._wrap_with_state_markers(chunk_work_dir, raw_python_cmd)

            job_name = f"{self.proj_name}_{worker_name}"
            lsf_sh = self.script_out_dir / f"submit_{worker_name}.sh"
            
            self._build_lsf_script(lsf_sh, job_name, worker_name, node_config, python_commands)

            overall_status = 'completed' if all_completed else ('running' if any(s=='running' for _,_,s in assigned_chunks) else 'pending')
            generated_scripts.append((lsf_sh, worker_name, overall_status, chunk_dirs))

        self._generate_master_submit_script("greedy", generated_scripts)

    def _generate_master_submit_script(self, mode_name, generated_scripts):
        master_script = self.script_out_dir / f"submit_all_workers.sh"
        sch_name = "Slurm" if self.scheduler == 'slurm' else "LSF"
        submit_cmd = "sbatch" if self.scheduler == 'slurm' else "bsub <"

        with open(master_script, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write(f'echo "=== {sch_name} 并发调度投递引擎 === "\n\n')
            for script_path, worker_name, status, chunk_dirs in generated_scripts:
                touch_cmds = " && ".join([f"touch \"{d}/.state_submitted\"" for d in chunk_dirs])

                if status == 'completed':
                    f.write(f"# [✅ 满载] {worker_name} 的所有任务均已跑完。\n")
                elif status == 'running':
                    f.write(f"# [🚀 运行中] {worker_name} 的进程依然存活，安全忽略。\n")
                elif status == 'submitted':
                    f.write(f"# [⏳ 排队中] {worker_name} 已存在于队列，安全忽略。\n")
                elif status == 'failed':
                    f.write(f"{submit_cmd} {script_path} && {touch_cmds}  # [❌ 曾失败] 唤醒 {worker_name} 重新投递。\n")
                else:
                    f.write(f"{submit_cmd} {script_path} && {touch_cmds}  # [💼 上工] 唤醒 {worker_name} 进入调度池\n")

        print(f"[*] 构建完毕！请一键唤醒所有打工人: bash {master_script}")

    def _create_symlinks(self, sid_list, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        for sid in sid_list:
            for fpath in self.samples[sid]:
                target_path = target_dir / fpath.name
                if not target_path.exists():
                    if self.use_symlink: target_path.symlink_to(fpath.absolute())
                    else: shutil.copy2(fpath.absolute(), target_path)

    def discover_and_check_files(self):
        print(f"[*] 扫描输入目录 (Pipeline: {self.p_type.upper()})...")
        pattern = re.compile(r"(.+?)(_R1|_1|_01)(\.fastq|\.fq)(\.gz)?$") if self.p_type == 'ngs' else re.compile(r"(.+?)(\.fastq|\.fq)(\.gz)?$")
        for d_str in self.d_conf.get('input_dirs', []):
            p = Path(d_str)
            if not p.exists(): continue
            for f in p.rglob("*.*"):
                if not f.is_file(): continue
                if any((ex == f.absolute() or ex in f.absolute().parents) for ex in self.exclude_dirs): continue
                match = pattern.match(f.name)
                if match:
                    if self.p_type == 'ngs':
                        prefix, tag, ext, gz = match.groups()
                        r2_tag = tag.replace('1', '2') if '01' not in tag else tag.replace('01', '02')
                        r2_path = f.parent / f"{prefix}{r2_tag}{ext}{gz or ''}"
                        if r2_path.exists(): self._add_sample(prefix, [f, r2_path])
                    else:
                        self._add_sample(match.group(1), [f])
        print(f"[+] 系统共感知到 {len(self.samples)} 个合规样本。")

    def _add_sample(self, sample_id, file_paths):
        if sample_id not in self.samples: self.samples[sample_id] = file_paths

    def _check_and_clean_workspace(self):
        if self.base_work_dir.exists() and self.force_overwrite:
            print(f"\n[*] 启用 --force，彻底抹除旧目录: {self.base_work_dir}")
            shutil.rmtree(self.base_work_dir)

    def _dispatch_target_merge(self):
        print(f"\n[*] 🚀 启动 [Target-Merge] 靶向增量模式")
        default_merged = self.base_work_dir / "00_Merged_Project_Results"
        merged_dir = Path(self.d_conf.get('merged_result_dir', default_merged))
        if not merged_dir.exists():
            return print(f"[❌ 错误] 找不到合并总仓库: {merged_dir}")

        node_name = self.worker_list[0]
        node_config = copy.deepcopy(self.d_conf.get('node_defaults', {}))

        out_yaml = self.base_work_dir / "00_configs_merged" / "merged_target_config.yaml"
        self._build_pipeline_config(out_yaml, self.d_conf.get('input_dirs', []), merged_dir, node_config)

        lsf_sh = self.script_out_dir / "submit_merged_target.sh"
        run_cmd = f"python {self.d_conf['pipeline_script']} {out_yaml}"
        self._build_lsf_script(lsf_sh, f"{self.proj_name}_merge", node_name, node_config, run_cmd)
        
        sch_name = "Slurm" if self.scheduler == 'slurm' else "LSF"
        submit_cmd = "sbatch" if self.scheduler == 'slurm' else "bsub <"
        print(f"[*] ✅ 已生成合并脚本。{sch_name} 提交命令: {submit_cmd} {lsf_sh}\n")

    def _print_topology(self, node_chunk_map):
        sch_name = "Slurm" if self.scheduler == 'slurm' else "LSF"
        print("\n" + "="*60 + f"\n 📊 {sch_name} 并发调度拓扑概览\n" + "="*60)
        for worker, chunks in node_chunk_map.items():
            if not chunks: continue
            print(f"\n👷  挂载目标节点 (Node): \033[1;36m{worker}\033[0m")
            for i, (chunk_id, samples, status) in enumerate(chunks):
                branch = " └──" if i == len(chunks) - 1 else " ├──"
                color = "\033[1;32m" if status == 'completed' else \
                        "\033[1;34m" if status == 'running' else \
                        "\033[1;35m" if status == 'submitted' else \
                        "\033[1;31m" if status == 'failed' else "\033[1;33m"
                print(f"{branch} 📦 {color}{chunk_id} [{status}]\033[0m: [{', '.join(samples[:2])} ...等]")

    def dispatch(self):
        if self.target_merge: return self._dispatch_target_merge()

        self.discover_and_check_files()
        if not self.samples: return print("[!] 未找到有效样本。")
        self._check_and_clean_workspace()
        self.script_out_dir.mkdir(parents=True, exist_ok=True)

        chunk_dict = self._compute_incremental_chunks()
        node_def = self.d_conf.get('node_defaults', {})
        node_over = self.d_conf.get('node_overrides', {})

        self._dispatch_greedy(chunk_dict, node_def, node_over)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工业级流式分发器 (虚拟 Worker 并发版)")
    parser.add_argument('config', help="全局 YAML 配置文件路径")
    parser.add_argument('-f', '--force', action='store_true', help="强制覆写工作目录")
    parser.add_argument('--flush', action='store_true', help="清空缓冲池组块 (解决末尾不满数等待)")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        d_conf = yaml.safe_load(f)

    dispatcher = UnifiedDispatcher(d_conf=d_conf, force_overwrite=args.force, flush_tail=args.flush)
    dispatcher.dispatch()