import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.dont_write_bytecode = True

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "llama_launcher_config.json")
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(APP_DIR, "llama_launcher_config.json")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from field_help import HELP

DEFAULT_EXE = "llama-server.exe" if os.name == "nt" else "llama-server"
LLAMA_SERVER_EXE = os.path.join(APP_DIR, DEFAULT_EXE)

KIND_FLAG = "flag"
KIND_BOOL = "bool"
KIND_CHOICE = "choice"
KIND_VALUE = "value"
KIND_PATH = "path"

def F(key, label, flag, default=False, aliases=None, help=""):
    return dict(key=key, label=label, flag=flag, kind=KIND_FLAG,
                default=default, aliases=aliases or [], help=help)

def B(key, label, on_flag, off_flag, default="default", aliases=None, help=""):
    return dict(key=key, label=label, on_flag=on_flag, off_flag=off_flag,
                kind=KIND_BOOL, default=default, aliases=aliases or [], help=help)

def C(key, label, flag, choices, default="", essential=False, aliases=None, help=""):
    return dict(key=key, label=label, flag=flag, kind=KIND_CHOICE,
                choices=choices, default=default, essential=essential,
                aliases=aliases or [], help=help)

def V(key, label, flag, default="", essential=False, aliases=None, help=""):
    return dict(key=key, label=label, flag=flag, kind=KIND_VALUE,
                default=default, essential=essential, aliases=aliases or [], help=help)

def P(key, label, flag, default="", subtype="file", essential=False, aliases=None, help=""):
    return dict(key=key, label=label, flag=flag, kind=KIND_PATH,
                subtype=subtype, default=default, essential=essential,
                aliases=aliases or [], help=help)

CACHE_TYPES = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
ONOFFAUTO = ["on", "off", "auto"]
ONOFF = ["on", "off"]
ZEROONE = ["0", "1"]
PRIO = ["-1", "0", "1", "2", "3"]

COMMON = [
    V('threads', 'Threads (-t)', '--threads', '-1', aliases=['-t'], help=HELP['threads']),
    V('threads_batch', 'Threads batch (-tb)', '--threads-batch', '', aliases=['-tb'], help=HELP['threads_batch']),
    V('cpu_mask', 'CPU mask (-C)', '--cpu-mask', '', aliases=['-C'], help=HELP['cpu_mask']),
    V('cpu_range', 'CPU range (-Cr)', '--cpu-range', '', aliases=['-Cr'], help=HELP['cpu_range']),
    C('cpu_strict', 'CPU strict placement', '--cpu-strict', ZEROONE, '0', help=HELP['cpu_strict']),
    C('prio', 'Process/thread priority', '--prio', PRIO, '0', help=HELP['prio']),
    V('poll', 'Poll level (0-100)', '--poll', '50', help=HELP['poll']),
    V('cpu_mask_batch', 'CPU mask batch (-Cb)', '--cpu-mask-batch', '', aliases=['-Cb'], help=HELP['cpu_mask_batch']),
    V('cpu_range_batch', 'CPU range batch (-Crb)', '--cpu-range-batch', '', aliases=['-Crb'], help=HELP['cpu_range_batch']),
    C('cpu_strict_batch', 'CPU strict batch', '--cpu-strict-batch', ZEROONE, '0', help=HELP['cpu_strict_batch']),
    C('prio_batch', 'Priority batch', '--prio-batch', PRIO, '0', help=HELP['prio_batch']),
    C('poll_batch', 'Poll batch', '--poll-batch', ZEROONE, '0', help=HELP['poll_batch']),
    V('ctx_size', 'Context size (-c)', '--ctx-size', '32768', essential=True, aliases=['-c'], help=HELP['ctx_size']),
    V('n_predict', 'Predict tokens (-n)', '--n-predict', '-1', aliases=['-n', '--predict'], help=HELP['n_predict']),
    V('batch_size', 'Batch size (-b)', '--batch-size', '2048', essential=True, aliases=['-b'], help=HELP['batch_size']),
    V('ubatch_size', 'Micro-batch size (-ub)', '--ubatch-size', '512', essential=True, aliases=['-ub'], help=HELP['ubatch_size']),
    V('keep', 'Keep tokens', '--keep', '0', help=HELP['keep']),
    F('swa_full', 'Use full-size SWA cache', '--swa-full', help=HELP['swa_full']),
    C('flash_attn', 'Flash Attention (-fa)', '--flash-attn', ONOFFAUTO, 'auto', essential=True, aliases=['-fa'], help=HELP['flash_attn']),
    B('perf', 'Internal perf timings', '--perf', '--no-perf', 'default', help=HELP['perf']),
    B('escape', 'Process escape sequences (-e)', '--escape', '--no-escape', 'default', aliases=['-e'], help=HELP['escape']),
    C('rope_scaling', 'RoPE scaling method', '--rope-scaling', ['none', 'linear', 'yarn'], help=HELP['rope_scaling']),
    V('rope_scale', 'RoPE scale factor', '--rope-scale', '', help=HELP['rope_scale']),
    V('rope_freq_base', 'RoPE freq base', '--rope-freq-base', '', help=HELP['rope_freq_base']),
    V('rope_freq_scale', 'RoPE freq scale', '--rope-freq-scale', '', help=HELP['rope_freq_scale']),
    V('yarn_orig_ctx', 'YaRN original ctx', '--yarn-orig-ctx', '0', help=HELP['yarn_orig_ctx']),
    V('yarn_ext_factor', 'YaRN ext factor', '--yarn-ext-factor', '-1.00', help=HELP['yarn_ext_factor']),
    V('yarn_attn_factor', 'YaRN attn factor', '--yarn-attn-factor', '-1.00', help=HELP['yarn_attn_factor']),
    V('yarn_beta_slow', 'YaRN beta slow', '--yarn-beta-slow', '-1.00', help=HELP['yarn_beta_slow']),
    V('yarn_beta_fast', 'YaRN beta fast', '--yarn-beta-fast', '-1.00', help=HELP['yarn_beta_fast']),
    B('kv_offload', 'KV cache offloading', '--kv-offload', '--no-kv-offload', 'default', aliases=['-kvo', '-nkvo'], help=HELP['kv_offload']),
    B('repack', 'Weight repacking', '--repack', '--no-repack', 'default', aliases=['-nr'], help=HELP['repack']),
    F('no_host', 'Bypass host buffer (--no-host)', '--no-host', help=HELP['no_host']),
    C('cache_type_k', 'KV cache type K (-ctk)', '--cache-type-k', CACHE_TYPES, 'f16', aliases=['-ctk'], help=HELP['cache_type_k']),
    C('cache_type_v', 'KV cache type V (-ctv)', '--cache-type-v', CACHE_TYPES, 'f16', aliases=['-ctv'], help=HELP['cache_type_v']),
    V('defrag_thold', 'Defrag threshold (-dt) [deprecated]', '--defrag-thold', '', aliases=['-dt'], help=HELP['defrag_thold']),
    V('rpc', 'RPC servers', '--rpc', '', help=HELP['rpc']),
    C('load_mode', 'Model load mode (-lm)', '--load-mode', ['auto', 'none', 'mmap', 'mlock', 'mmap+mlock', 'dio'], 'auto', aliases=['-lm'], help=HELP['load_mode']),
    C('lazy_mode', 'Lazy mode (-lzm)', '--lazy-mode', ['on', 'auto', 'off'], 'auto', aliases=['-lzm'], help=HELP['lazy_mode']),
    C('numa', 'NUMA optimization', '--numa', ['distribute', 'isolate', 'numactl'], help=HELP['numa']),
    V('device', 'Devices (-dev)', '--device', '', aliases=['-dev'], help=HELP['device']),
    F('list_devices', 'List available devices', '--list-devices', help=HELP['list_devices']),
    V('override_tensor', 'Override tensor buffer (-ot)', '--override-tensor', '', aliases=['-ot'], help=HELP['override_tensor']),
    F('cpu_moe', 'Keep all MoE weights on CPU (-cmoe)', '--cpu-moe', aliases=['-cmoe'], help=HELP['cpu_moe']),
    V('n_cpu_moe', 'N CPU MoE layers (-ncmoe)', '--n-cpu-moe', '', aliases=['-ncmoe'], help=HELP['n_cpu_moe']),
    V('n_cpu_ffn', 'N CPU FFN layers (-ncffn)', '--n-cpu-ffn', '', aliases=['-ncffn'], help=HELP['n_cpu_ffn']),
    V('n_gpu_layers', 'GPU layers (-ngl)', '--gpu-layers', 'all', essential=True, aliases=['-ngl', '--n-gpu-layers'], help=HELP['n_gpu_layers']),
    C('split_mode', 'Split mode (-sm)', '--split-mode', ['none', 'layer', 'row', 'tensor'], aliases=['-sm'], help=HELP['split_mode']),
    V('tensor_split', 'Tensor split (-ts)', '--tensor-split', '', aliases=['-ts'], help=HELP['tensor_split']),
    V('main_gpu', 'Main GPU index (-mg)', '--main-gpu', '0', aliases=['-mg'], help=HELP['main_gpu']),
    C('fit', 'Auto-fit to VRAM (-fit)', '--fit', ONOFFAUTO, 'on', aliases=['-fit'], help=HELP['fit']),
    V('fit_target', 'Fit target MiB (-fitt)', '--fit-target', '', aliases=['-fitt'], help=HELP['fit_target']),
    V('fit_ctx', 'Fit min ctx (-fitc)', '--fit-ctx', '', aliases=['-fitc'], help=HELP['fit_ctx']),
    F('check_tensors', 'Check tensors for invalid values', '--check-tensors', help=HELP['check_tensors']),
    V('override_kv', 'Override KV metadata', '--override-kv', '', help=HELP['override_kv']),
    B('op_offload', 'Offload host tensor ops to device', '--op-offload', '--no-op-offload', 'default', help=HELP['op_offload']),
    V('lora', 'LoRA adapter(s)', '--lora', '', help=HELP['lora']),
    V('lora_scaled', 'LoRA scaled', '--lora-scaled', '', help=HELP['lora_scaled']),
    V('control_vector', 'Control vector', '--control-vector', '', help=HELP['control_vector']),
    V('control_vector_scaled', 'Control vector scaled', '--control-vector-scaled', '', help=HELP['control_vector_scaled']),
    V('control_vector_layer_range', 'Control vector layer range', '--control-vector-layer-range', '', help=HELP['control_vector_layer_range']),
    P('model', 'Model file (-m)', '--model', '', subtype='file', essential=True, aliases=['-m'], help=HELP['model']),
    V('model_url', 'Model URL (-mu)', '--model-url', '', aliases=['-mu'], help=HELP['model_url']),
    V('docker_repo', 'Docker Hub repo (-dr)', '--docker-repo', '', aliases=['-dr'], help=HELP['docker_repo']),
    V('hf_repo', 'HF repo (-hf)', '--hf-repo', '', aliases=['-hf', '-hfr'], help=HELP['hf_repo']),
    V('hf_file', 'HF file (-hff)', '--hf-file', '', aliases=['-hff'], help=HELP['hf_file']),
    V('hf_token', 'HF token (-hft)', '--hf-token', '', aliases=['-hft'], help=HELP['hf_token']),
    F('log_disable', 'Disable log', '--log-disable', help=HELP['log_disable']),
    P('log_file', 'Log file', '--log-file', '', subtype='file', help=HELP['log_file']),
    B('log_jsonl', 'Log as JSONL', '--log-jsonl', '--no-log-jsonl', 'default', help=HELP['log_jsonl']),
    C('log_colors', 'Log colors', '--log-colors', ONOFFAUTO, 'auto', help=HELP['log_colors']),
    F('verbose', 'Verbose (-v)', '--verbose', default=True, aliases=['-v', '--log-verbose'], help=HELP['verbose']),
    F('offline', 'Offline mode', '--offline', help=HELP['offline']),
    C('verbosity', 'Verbosity level (-lv)', '--verbosity', ['0', '1', '2', '3', '4', '5'], '3', essential=True, aliases=['-lv', '--log-verbosity'], help=HELP['verbosity']),
    B('log_prefix', 'Log prefix', '--log-prefix', '--no-log-prefix', 'on', help=HELP['log_prefix']),
    B('log_timestamps', 'Log timestamps', '--log-timestamps', '--no-log-timestamps', 'on', help=HELP['log_timestamps']),
    C('cache_type_k_draft', 'KV cache type K draft (-ctkd)', '--cache-type-k-draft', CACHE_TYPES, 'f16', aliases=['-ctkd', '--spec-draft-type-k'], help=HELP['cache_type_k_draft']),
    C('cache_type_v_draft', 'KV cache type V draft (-ctvd)', '--cache-type-v-draft', CACHE_TYPES, 'f16', aliases=['-ctvd', '--spec-draft-type-v'], help=HELP['cache_type_v_draft']),
]

SAMPLING = [
    V('samplers', 'Samplers order', '--samplers', '', help=HELP['samplers']),
    V('seed', 'Seed (-s)', '--seed', '-1', aliases=['-s'], help=HELP['seed']),
    V('sampler_seq', 'Sampler seq', '--sampler-seq', '', aliases=['--sampling-seq'], help=HELP['sampler_seq']),
    F('ignore_eos', 'Ignore EOS', '--ignore-eos', help=HELP['ignore_eos']),
    V('temp', 'Temperature', '--temp', '0.80', essential=True, aliases=['--temperature'], help=HELP['temp']),
    V('top_k', 'Top-K', '--top-k', '40', essential=True, help=HELP['top_k']),
    V('top_p', 'Top-P', '--top-p', '0.95', essential=True, help=HELP['top_p']),
    V('min_p', 'Min-P', '--min-p', '0.05', essential=True, help=HELP['min_p']),
    V('top_n_sigma', 'Top-N-sigma', '--top-nsigma', '', aliases=['--top-n-sigma'], help=HELP['top_n_sigma']),
    V('xtc_probability', 'XTC probability', '--xtc-probability', '', help=HELP['xtc_probability']),
    V('xtc_threshold', 'XTC threshold', '--xtc-threshold', '', help=HELP['xtc_threshold']),
    V('typical_p', 'Typical-P', '--typical', '', aliases=['--typical-p'], help=HELP['typical_p']),
    V('repeat_last_n', 'Repeat last N', '--repeat-last-n', '', help=HELP['repeat_last_n']),
    V('repeat_penalty', 'Repeat penalty', '--repeat-penalty', '', help=HELP['repeat_penalty']),
    V('presence_penalty', 'Presence penalty', '--presence-penalty', '', help=HELP['presence_penalty']),
    V('frequency_penalty', 'Frequency penalty', '--frequency-penalty', '', help=HELP['frequency_penalty']),
    V('dry_multiplier', 'DRY multiplier', '--dry-multiplier', '', help=HELP['dry_multiplier']),
    V('dry_base', 'DRY base', '--dry-base', '', help=HELP['dry_base']),
    V('dry_allowed_length', 'DRY allowed length', '--dry-allowed-length', '', help=HELP['dry_allowed_length']),
    V('dry_penalty_last_n', 'DRY penalty last N', '--dry-penalty-last-n', '', help=HELP['dry_penalty_last_n']),
    V('dry_sequence_breaker', 'DRY sequence breaker', '--dry-sequence-breaker', '', help=HELP['dry_sequence_breaker']),
    V('adaptive_target', 'Adaptive-p target', '--adaptive-target', '', help=HELP['adaptive_target']),
    V('adaptive_decay', 'Adaptive-p decay', '--adaptive-decay', '', help=HELP['adaptive_decay']),
    V('dynatemp_range', 'Dynatemp range', '--dynatemp-range', '', help=HELP['dynatemp_range']),
    V('dynatemp_exp', 'Dynatemp exponent', '--dynatemp-exp', '', help=HELP['dynatemp_exp']),
    C('mirostat', 'Mirostat', '--mirostat', ['0', '1', '2'], '0', help=HELP['mirostat']),
    V('mirostat_lr', 'Mirostat learning rate', '--mirostat-lr', '', help=HELP['mirostat_lr']),
    V('mirostat_ent', 'Mirostat target entropy', '--mirostat-ent', '', help=HELP['mirostat_ent']),
    V('logit_bias', 'Logit bias (-l)', '--logit-bias', '', aliases=['-l'], help=HELP['logit_bias']),
    V('grammar', 'Grammar (BNF)', '--grammar', '', help=HELP['grammar']),
    P('grammar_file', 'Grammar file', '--grammar-file', '', subtype='file', help=HELP['grammar_file']),
    V('json_schema', 'JSON schema (-j)', '--json-schema', '', aliases=['-j'], help=HELP['json_schema']),
    P('json_schema_file', 'JSON schema file (-jf)', '--json-schema-file', '', subtype='file', aliases=['-jf'], help=HELP['json_schema_file']),
    F('backend_sampling', 'Backend sampling (-bs)', '--backend-sampling', aliases=['-bs'], help=HELP['backend_sampling']),
]

SPECULATIVE = [
    V('hf_repo_draft', 'HF draft repo (-hfd)', '--spec-draft-hf', '', aliases=['-hfd', '-hfrd', '--hf-repo-draft'], help=HELP['hf_repo_draft']),
    V('threads_draft', 'Draft threads (-td)', '--spec-draft-threads', '', aliases=['-td', '--threads-draft'], help=HELP['threads_draft']),
    V('threads_batch_draft', 'Draft threads batch (-tbd)', '--spec-draft-threads-batch', '', aliases=['-tbd', '--threads-batch-draft'], help=HELP['threads_batch_draft']),
    V('cpu_mask_draft', 'Draft CPU mask (-Cd)', '--spec-draft-cpu-mask', '', aliases=['-Cd', '--cpu-mask-draft'], help=HELP['cpu_mask_draft']),
    V('cpu_range_draft', 'Draft CPU range (-Crd)', '--spec-draft-cpu-range', '', aliases=['-Crd', '--cpu-range-draft'], help=HELP['cpu_range_draft']),
    C('cpu_strict_draft', 'Draft CPU strict', '--cpu-strict-draft', ZEROONE, '', aliases=['--cpu-strict-draft'], help=HELP['cpu_strict_draft']),
    C('prio_draft', 'Draft priority', '--spec-draft-prio', PRIO, '', aliases=['--prio-draft'], help=HELP['prio_draft']),
    C('poll_draft', 'Draft poll', '--spec-draft-poll', ZEROONE, '', aliases=['--poll-draft'], help=HELP['poll_draft']),
    V('cpu_mask_batch_draft', 'Draft CPU mask batch (-Cbd)', '--spec-draft-cpu-mask-batch', '', aliases=['-Cbd', '--cpu-mask-batch-draft'], help=HELP['cpu_mask_batch_draft']),
    C('cpu_strict_batch_draft', 'Draft CPU strict batch', '--spec-draft-cpu-strict-batch', ZEROONE, '', aliases=['--cpu-strict-batch-draft'], help=HELP['cpu_strict_batch_draft']),
    C('prio_batch_draft', 'Draft priority batch', '--spec-draft-prio-batch', PRIO, '', aliases=['--prio-batch-draft'], help=HELP['prio_batch_draft']),
    C('poll_batch_draft', 'Draft poll batch', '--spec-draft-poll-batch', ZEROONE, '', aliases=['--poll-batch-draft'], help=HELP['poll_batch_draft']),
    V('override_tensor_draft', 'Draft override tensor (-otd)', '--spec-draft-override-tensor', '', aliases=['-otd', '--override-tensor-draft'], help=HELP['override_tensor_draft']),
    F('cpu_moe_draft', 'Draft: keep all MoE on CPU (-cmoed)', '--spec-draft-cpu-moe', aliases=['-cmoed', '--cpu-moe-draft'], help=HELP['cpu_moe_draft']),
    V('n_cpu_moe_draft', 'Draft N CPU MoE layers', '--spec-draft-n-cpu-moe', '', aliases=['--spec-draft-ncmoe', '-ncmoed', '--n-cpu-moe-draft'], help=HELP['n_cpu_moe_draft']),
    V('spec_draft_n_max', 'Draft tokens max', '--spec-draft-n-max', '', help=HELP['spec_draft_n_max']),
    V('spec_draft_n_min', 'Draft tokens min', '--spec-draft-n-min', '', help=HELP['spec_draft_n_min']),
    V('spec_synth_len', 'Spec synth acceptance length', '--spec-synth-len', '', help=HELP['spec_synth_len']),
    V('spec_synth_rates', 'Spec synth acceptance rates', '--spec-synth-rates', '', help=HELP['spec_synth_rates']),
    V('spec_draft_p_split', 'Draft split probability', '--spec-draft-p-split', '', aliases=['--draft-p-split'], help=HELP['spec_draft_p_split']),
    V('spec_draft_p_min', 'Draft min probability', '--spec-draft-p-min', '', aliases=['--draft-p-min'], help=HELP['spec_draft_p_min']),
    B('spec_draft_backend_sampling', 'Draft backend sampling', '--spec-draft-backend-sampling', '--no-spec-draft-backend-sampling', 'default', help=HELP['spec_draft_backend_sampling']),
    V('device_draft', 'Draft devices (-devd)', '--spec-draft-device', '', aliases=['-devd', '--device-draft'], help=HELP['device_draft']),
    V('n_gpu_layers_draft', 'Draft GPU layers (-ngld)', '--spec-draft-ngl', '', aliases=['-ngld', '--gpu-layers-draft', '--n-gpu-layers-draft'], help=HELP['n_gpu_layers_draft']),
    P('model_draft', 'Draft model file (-md)', '--spec-draft-model', '', subtype='file', aliases=['-md', '--model-draft'], help=HELP['model_draft']),
    V('spec_type', 'Speculative type(s)', '--spec-type', '', help=HELP['spec_type']),
    V('spec_ngram_mod_n_min', 'ngram-mod N min', '--spec-ngram-mod-n-min', '', help=HELP['spec_ngram_mod_n_min']),
    V('spec_ngram_mod_n_max', 'ngram-mod N max', '--spec-ngram-mod-n-max', '', help=HELP['spec_ngram_mod_n_max']),
    V('spec_ngram_mod_n_match', 'ngram-mod lookup length', '--spec-ngram-mod-n-match', '', help=HELP['spec_ngram_mod_n_match']),
    V('spec_ngram_simple_size_n', 'ngram-simple size N', '--spec-ngram-simple-size-n', '', help=HELP['spec_ngram_simple_size_n']),
    V('spec_ngram_simple_size_m', 'ngram-simple size M', '--spec-ngram-simple-size-m', '', help=HELP['spec_ngram_simple_size_m']),
    V('spec_ngram_simple_min_hits', 'ngram-simple min hits', '--spec-ngram-simple-min-hits', '', help=HELP['spec_ngram_simple_min_hits']),
    V('spec_ngram_map_k_size_n', 'ngram-map-k size N', '--spec-ngram-map-k-size-n', '', help=HELP['spec_ngram_map_k_size_n']),
    V('spec_ngram_map_k_size_m', 'ngram-map-k size M', '--spec-ngram-map-k-size-m', '', help=HELP['spec_ngram_map_k_size_m']),
    V('spec_ngram_map_k_min_hits', 'ngram-map-k min hits', '--spec-ngram-map-k-min-hits', '', help=HELP['spec_ngram_map_k_min_hits']),
    V('spec_ngram_map_k4v_size_n', 'ngram-map-k4v size N', '--spec-ngram-map-k4v-size-n', '', help=HELP['spec_ngram_map_k4v_size_n']),
    V('spec_ngram_map_k4v_size_m', 'ngram-map-k4v size M', '--spec-ngram-map-k4v-size-m', '', help=HELP['spec_ngram_map_k4v_size_m']),
    V('spec_ngram_map_k4v_min_hits', 'ngram-map-k4v min hits', '--spec-ngram-map-k4v-min-hits', '', help=HELP['spec_ngram_map_k4v_min_hits']),
    F('spec_default', 'Enable default speculative config', '--spec-default', help=HELP['spec_default']),
]

SERVER = [
    P('lookup_cache_static', 'Static lookup cache (-lcs)', '--lookup-cache-static', '', subtype='file', aliases=['-lcs'], help=HELP['lookup_cache_static']),
    P('lookup_cache_dynamic', 'Dynamic lookup cache (-lcd)', '--lookup-cache-dynamic', '', subtype='file', aliases=['-lcd'], help=HELP['lookup_cache_dynamic']),
    V('ctx_checkpoints', 'Context checkpoints (-ctxcp)', '--ctx-checkpoints', '', aliases=['-ctxcp', '--swa-checkpoints'], help=HELP['ctx_checkpoints']),
    V('checkpoint_min_step', 'Checkpoint min spacing (-cms)', '--checkpoint-min-step', '', aliases=['-cms'], help=HELP['checkpoint_min_step']),
    V('cache_ram', 'Cache RAM MiB (-cram)', '--cache-ram', '', aliases=['-cram'], help=HELP['cache_ram']),
    B('kv_unified', 'Unified KV buffer (-kvu)', '--kv-unified', '--no-kv-unified', 'default', aliases=['-kvu', '-no-kvu'], help=HELP['kv_unified']),
    V('kv_unified_per_slot', 'KV unified per-slot limit', '--kv-unified-per-slot', '', help=HELP['kv_unified_per_slot']),
    B('cache_idle_slots', 'Cache idle slots', '--cache-idle-slots', '--no-cache-idle-slots', 'default', help=HELP['cache_idle_slots']),
    B('context_shift', 'Context shift', '--context-shift', '--no-context-shift', 'default', help=HELP['context_shift']),
    V('reverse_prompt', 'Reverse prompt (-r)', '--reverse-prompt', '', aliases=['-r'], help=HELP['reverse_prompt']),
    F('special', 'Special tokens output (-sp)', '--special', aliases=['-sp'], help=HELP['special']),
    B('warmup', 'Warmup run', '--warmup', '--no-warmup', 'default', help=HELP['warmup']),
    F('spm_infill', 'SPM infill pattern', '--spm-infill', help=HELP['spm_infill']),
    C('pooling', 'Pooling type', '--pooling', ['none', 'mean', 'cls', 'last', 'rank'], help=HELP['pooling']),
    V('parallel', 'Parallel slots (-np)', '--parallel', '', aliases=['-np'], help=HELP['parallel']),
    B('cont_batching', 'Continuous batching (-cb)', '--cont-batching', '--no-cont-batching', 'default', aliases=['-cb', '-nocb'], help=HELP['cont_batching']),
    P('mmproj', 'Multimodal projector (-mm)', '--mmproj', '', subtype='file', aliases=['-mm'], help=HELP['mmproj']),
    V('mmproj_url', 'Multimodal projector URL', '--mmproj-url', '', aliases=['-mmu'], help=HELP['mmproj_url']),
    C('mmproj_auto', 'Multimodal projector auto-use', '--mmproj-auto', ['default', 'enabled (--mmproj-auto)', 'disabled (--no-mmproj)'], aliases=['--no-mmproj', '--no-mmproj-auto'], help=HELP['mmproj_auto']),
    B('mmproj_offload', 'Multimodal projector GPU offload', '--mmproj-offload', '--no-mmproj-offload', 'default', help=HELP['mmproj_offload']),
    V('mmproj_device', 'MMProj device (--mmdev)', '--mmproj-device', '', aliases=['--mmdev'], help=HELP['mmproj_device']),
    V('image_min_tokens', 'Image min tokens', '--image-min-tokens', '', help=HELP['image_min_tokens']),
    V('image_max_tokens', 'Image max tokens', '--image-max-tokens', '', help=HELP['image_max_tokens']),
    V('mtmd_batch_max_tokens', 'MTMD batch max tokens', '--mtmd-batch-max-tokens', '', help=HELP['mtmd_batch_max_tokens']),
    V('video_fps', 'Video FPS', '--video-fps', '', help=HELP['video_fps']),
    V('video_timestamp_interval', 'Video timestamp interval (ms)', '--video-timestamp-interval', '', help=HELP['video_timestamp_interval']),
    P('video_ffmpeg_dir', 'Video ffmpeg directory', '--video-ffmpeg-dir', '', subtype='dir', help=HELP['video_ffmpeg_dir']),
    V('alias', 'Model alias (-a)', '--alias', '', aliases=['-a'], help=HELP['alias']),
    V('tags', 'Model tags', '--tags', '', help=HELP['tags']),
    V('embd_normalize', 'Embedding normalization', '--embd-normalize', '', help=HELP['embd_normalize']),
    V('host', 'Host', '--host', '127.0.0.1', essential=True, help=HELP['host']),
    V('port', 'Port', '--port', '8080', essential=True, help=HELP['port']),
    F('reuse_port', 'Reuse port', '--reuse-port', help=HELP['reuse_port']),
    P('path', 'Static files path', '--path', '', subtype='dir', help=HELP['path']),
    V('cors_origins', 'CORS origins', '--cors-origins', '', help=HELP['cors_origins']),
    V('cors_methods', 'CORS methods', '--cors-methods', '', help=HELP['cors_methods']),
    V('cors_headers', 'CORS headers', '--cors-headers', '', help=HELP['cors_headers']),
    B('cors_credentials', 'CORS credentials', '--cors-credentials', '--no-cors-credentials', 'default', help=HELP['cors_credentials']),
    V('api_prefix', 'API prefix', '--api-prefix', '', help=HELP['api_prefix']),
    V('ui_config', 'UI config (JSON)', '--ui-config', '', aliases=['--webui-config'], help=HELP['ui_config']),
    P('ui_config_file', 'UI config file', '--ui-config-file', '', subtype='file', aliases=['--webui-config-file'], help=HELP['ui_config_file']),
    B('ui_mcp_proxy', 'UI MCP CORS proxy (experimental)', '--ui-mcp-proxy', '--no-ui-mcp-proxy', 'default', aliases=['--webui-mcp-proxy', '--no-webui-mcp-proxy'], help=HELP['ui_mcp_proxy']),
    V('tools', 'Built-in agent tools', '--tools', '', help=HELP['tools']),
    V('tools_runtime', 'Tools runtime', '--tools-runtime', '', help=HELP['tools_runtime']),
    P('mcp_servers_config', 'MCP servers config (JSON)', '--mcp-servers-config', '', subtype='file', help=HELP['mcp_servers_config']),
    V('mcp_servers_json', 'MCP servers (inline JSON)', '--mcp-servers-json', '', help=HELP['mcp_servers_json']),
    B('agent', 'Agent mode (-ag)', '--agent', '--no-agent', 'default', aliases=['-ag', '-no-ag'], help=HELP['agent']),
    B('ui', 'Web UI', '--ui', '--no-ui', 'default', aliases=['--webui', '--no-webui'], help=HELP['ui']),
    F('embedding', 'Embedding mode', '--embedding', aliases=['--embeddings'], help=HELP['embedding']),
    F('rerank', 'Reranking endpoint', '--rerank', aliases=['--reranking'], help=HELP['rerank']),
    V('api_key', 'API key', '--api-key', '', help=HELP['api_key']),
    P('api_key_file', 'API key file', '--api-key-file', '', subtype='file', help=HELP['api_key_file']),
    P('ssl_key_file', 'SSL key file', '--ssl-key-file', '', subtype='file', help=HELP['ssl_key_file']),
    P('ssl_cert_file', 'SSL cert file', '--ssl-cert-file', '', subtype='file', help=HELP['ssl_cert_file']),
    V('chat_template_kwargs', 'Chat template kwargs (JSON)', '--chat-template-kwargs', '', help=HELP['chat_template_kwargs']),
    V('timeout', 'R/W timeout seconds (-to)', '--timeout', '', aliases=['-to'], help=HELP['timeout']),
    V('sse_ping_interval', 'SSE ping interval', '--sse-ping-interval', '', help=HELP['sse_ping_interval']),
    V('threads_http', 'HTTP threads', '--threads-http', '', help=HELP['threads_http']),
    B('cache_prompt', 'Prompt caching', '--cache-prompt', '--no-cache-prompt', 'default', help=HELP['cache_prompt']),
    V('cache_reuse', 'Cache reuse chunk size', '--cache-reuse', '', help=HELP['cache_reuse']),
    F('metrics', 'Prometheus metrics endpoint', '--metrics', help=HELP['metrics']),
    F('props', 'Allow changing global properties', '--props', help=HELP['props']),
    B('slots', 'Slots monitoring endpoint', '--slots', '--no-slots', 'default', help=HELP['slots']),
    P('slot_save_path', 'Slot save path', '--slot-save-path', '', subtype='dir', help=HELP['slot_save_path']),
    P('media_path', 'Media path', '--media-path', '', subtype='dir', help=HELP['media_path']),
    P('models_dir', 'Models directory (router)', '--models-dir', '', subtype='dir', help=HELP['models_dir']),
    P('models_preset', 'Models preset INI (router)', '--models-preset', '', subtype='file', help=HELP['models_preset']),
    V('models_max', 'Max simultaneous models (router)', '--models-max', '', help=HELP['models_max']),
    B('models_autoload', 'Auto-load models (router)', '--models-autoload', '--no-models-autoload', 'default', help=HELP['models_autoload']),
    B('jinja', 'Jinja template engine', '--jinja', '--no-jinja', 'default', help=HELP['jinja']),
    C('reasoning_format', 'Reasoning format', '--reasoning-format', ['auto', 'none', 'deepseek', 'deepseek-legacy'], 'auto', help=HELP['reasoning_format']),
    C('reasoning', 'Reasoning/thinking (-rea)', '--reasoning', ONOFFAUTO, 'auto', aliases=['-rea'], help=HELP['reasoning']),
    C('reasoning_effort', 'Reasoning effort', '--reasoning-effort', ['default', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'], 'default', help=HELP['reasoning_effort']),
    V('reasoning_budget', 'Reasoning budget', '--reasoning-budget', '', help=HELP['reasoning_budget']),
    V('reasoning_budget_message', 'Reasoning budget message', '--reasoning-budget-message', '', help=HELP['reasoning_budget_message']),
    B('reasoning_preserve', 'Preserve reasoning trace', '--reasoning-preserve', '--no-reasoning-preserve', 'default', help=HELP['reasoning_preserve']),
    V('chat_template', 'Chat template', '--chat-template', '', help=HELP['chat_template']),
    P('chat_template_file', 'Chat template file', '--chat-template-file', '', subtype='file', help=HELP['chat_template_file']),
    B('skip_chat_parsing', 'Skip chat parsing', '--skip-chat-parsing', '--no-skip-chat-parsing', 'default', help=HELP['skip_chat_parsing']),
    B('prefill_assistant', 'Prefill assistant response', '--prefill-assistant', '--no-prefill-assistant', 'default', help=HELP['prefill_assistant']),
    V('slot_prompt_similarity', 'Slot prompt similarity (-sps)', '--slot-prompt-similarity', '', aliases=['-sps'], help=HELP['slot_prompt_similarity']),
    F('lora_init_without_apply', 'Load LoRA without applying', '--lora-init-without-apply', help=HELP['lora_init_without_apply']),
    V('sleep_idle_seconds', 'Sleep after idle (seconds)', '--sleep-idle-seconds', '', help=HELP['sleep_idle_seconds']),
    P('log_prompts_dir', 'Log prompts directory', '--log-prompts-dir', '', subtype='dir', help=HELP['log_prompts_dir']),
    F('embd_gemma_default', 'Use default EmbeddingGemma', '--embd-gemma-default', help=HELP['embd_gemma_default']),
    F('fim_qwen_1_5b_default', 'Use default Qwen2.5 Coder 1.5B', '--fim-qwen-1.5b-default', help=HELP['fim_qwen_1_5b_default']),
    F('fim_qwen_3b_default', 'Use default Qwen2.5 Coder 3B', '--fim-qwen-3b-default', help=HELP['fim_qwen_3b_default']),
    F('fim_qwen_7b_default', 'Use default Qwen2.5 Coder 7B', '--fim-qwen-7b-default', help=HELP['fim_qwen_7b_default']),
    F('fim_qwen_7b_spec', 'Qwen2.5 Coder 7B + 0.5B draft (spec)', '--fim-qwen-7b-spec', help=HELP['fim_qwen_7b_spec']),
    F('fim_qwen_14b_spec', 'Qwen2.5 Coder 14B + 0.5B draft (spec)', '--fim-qwen-14b-spec', help=HELP['fim_qwen_14b_spec']),
    F('fim_qwen_30b_default', 'Use default Qwen3 Coder 30B A3B Instruct', '--fim-qwen-30b-default', help=HELP['fim_qwen_30b_default']),
    F('gpt_oss_20b_default', 'Use gpt-oss-20b default', '--gpt-oss-20b-default', help=HELP['gpt_oss_20b_default']),
    F('gpt_oss_120b_default', 'Use gpt-oss-120b default', '--gpt-oss-120b-default', help=HELP['gpt_oss_120b_default']),
    F('vision_gemma_4b_default', 'Use Gemma 3 4B QAT default', '--vision-gemma-4b-default', help=HELP['vision_gemma_4b_default']),
    F('vision_gemma_12b_default', 'Use Gemma 3 12B QAT default', '--vision-gemma-12b-default', help=HELP['vision_gemma_12b_default']),
]

ALL_CATEGORIES = [
    ("Common", COMMON),
    ("Sampling", SAMPLING),
    ("Speculative", SPECULATIVE),
    ("Server", SERVER),
]

ALL_FIELDS = []
for _cat, _fields in ALL_CATEGORIES:
    for _f in _fields:
        _f["category"] = _cat
        ALL_FIELDS.append(_f)

FIELDS_BY_KEY = {f["key"]: f for f in ALL_FIELDS}

QUICK_KEYS = [
    "model", "ctx_size", "n_gpu_layers", "threads", "batch_size", "ubatch_size",
    "flash_attn", "host", "port", "temp", "top_k", "top_p", "min_p",
]

FLAG_TO_FIELD = {}
for _f in ALL_FIELDS:
    k = _f["key"]
    flags_to_map = []
    if _f["kind"] == KIND_FLAG:
        flags_to_map.append(_f["flag"])
    elif _f["kind"] == KIND_BOOL:
        flags_to_map.extend([_f["on_flag"], _f["off_flag"]])
    elif _f["kind"] in (KIND_CHOICE, KIND_VALUE, KIND_PATH):
        flags_to_map.append(_f["flag"])

    flags_to_map.extend(_f.get("aliases", []))

    for flg in flags_to_map:
        if flg:
            FLAG_TO_FIELD[flg] = k

def make_combobox(parent, **kw):
    combo = ttk.Combobox(parent, **kw)
    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        combo.unbind(seq)
        combo.bind(seq, lambda e: "break")
    return combo

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        if not hasattr(widget, "_tooltips"):
            widget._tooltips = []
        widget._tooltips.append(self)
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def close(self):
        if self.tip:
            self.tip.destroy()
            self.tip = None

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tip, text=self.text, justify="left",
                          background="#ffffe0", relief="solid", borderwidth=1,
                          font=("Segoe UI", 8), wraplength=560, padx=6, pady=4)
        label.pack()

    def _hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas)

        self.body.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self._win, width=e.width),
        )
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def _wheel(event):
            delta = event.delta
            if delta == 0:
                return
            self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        def _bind(_e):
            self.canvas.bind_all("<MouseWheel>", _wheel)

        def _unbind(_e):
            self.canvas.unbind_all("<MouseWheel>")

        self.canvas.bind("<Enter>", _bind)
        self.canvas.bind("<Leave>", _unbind)

class LauncherApp:
    def __init__(self, root):
        self.root = root
        root.title("Llama.cpp Server Launcher")
        root.geometry("1080x760")

        self.vars = {}
        self.proc = None
        self.proc_tunnel = None
        self.tunnel_url = ""
        self.reader_thread = None
        self.out_queue = queue.Queue()
        self._syncing_from_preview = False

        self.workdir_var = tk.StringVar(value="")
        self.extra_args_rows = []
        self.status_var = tk.StringVar(value="Idle")
        self.tunnel_status_var = tk.StringVar(value="Idle")
        self.tab_content_frames = []

        self._build_vars()
        self._build_menu()
        self._build_notebook()
        self._build_statusbar()

        self._load_config()
        self.root.after(150, self._poll_output)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_vars(self):
        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            if kind == KIND_FLAG:
                self.vars[key] = {"var": tk.BooleanVar(value=bool(f.get("default", False)))}
            elif kind == KIND_BOOL:
                self.vars[key] = {"var": tk.StringVar(value=f.get("default", "default"))}
            elif kind in (KIND_CHOICE, KIND_VALUE, KIND_PATH):
                include_default = bool(f.get("essential", False) or key in {"verbosity"})
                self.vars[key] = {
                    "include": tk.BooleanVar(value=include_default),
                    "value": tk.StringVar(value=str(f.get("default", ""))),
                }
            for v in self.vars[key].values():
                v.trace_add("write", self._on_field_changed)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Export Settings...", command=self._export_settings)
        filemenu.add_command(label="Import Settings...", command=self._import_settings)
        filemenu.add_separator()
        filemenu.add_command(label="Reset All To Defaults", command=self._reset_defaults)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.root.config(menu=menubar)

    def _show_about(self):
        messagebox.showinfo("About",
            "Llama.cpp Server Launcher\n\nA GUI front-end for building and running llama-server commands.\n"
            "Hover over any (?) icon to see detailed usage and optimization tooltips.")

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        self._build_quick_tab()
        for cat, fields in ALL_CATEGORIES:
            self._build_category_tab(cat, fields)
        self._build_run_tab()

    def _build_quick_tab(self):
        frame = ScrollFrame(self.notebook)
        self.notebook.add(frame, text="Quick Start")
        info = ttk.Label(frame.body,
            text="The most commonly used flags. Everything else is available in the "
                 "other tabs, fully unchecked by default.",
            wraplength=900, foreground="#444")
        info.grid(row=0, column=0, columnspan=5, sticky="w", padx=8, pady=(8, 12))

        content = ttk.Frame(frame.body)
        content.grid(row=1, column=0, columnspan=5, sticky="nsew")
        frame.body.columnconfigure(0, weight=1)

        fields = [FIELDS_BY_KEY[key] for key in QUICK_KEYS]
        self.tab_content_frames.append((content, fields, frame, "Quick Start"))
        self._refresh_tab_content(content, fields)

    def _build_category_tab(self, name, fields):
        frame = ScrollFrame(self.notebook)
        self.notebook.add(frame, text=name)

        content = ttk.Frame(frame.body)
        content.grid(row=0, column=0, columnspan=5, sticky="nsew", pady=(8, 0))
        frame.body.columnconfigure(0, weight=1)

        self.tab_content_frames.append((content, fields, frame, name))
        self._refresh_tab_content(content, fields)

    def _refresh_tab_content(self, parent, fields):
        for child in parent.winfo_children():
            for tip in list(getattr(child, "_tooltips", ())):
                tip.close()
            child.destroy()
        row = 0
        for f in fields:
            self._render_field(parent, f, row)
            row += 1

    def _build_run_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="Run")
        row = 0
        ttk.Label(frame, text="Working directory (optional):").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.workdir_var, width=70).grid(row=row, column=1, sticky="we", padx=4)
        ttk.Button(frame, text="Browse...", command=self._browse_workdir).grid(row=row, column=2)
        row += 1
        ttk.Label(frame, text="Extra arguments (one flag + optional value per row):").grid(row=row, column=0, sticky="nw", pady=(6, 0))
        self.extra_args_frame = ttk.Frame(frame)
        self.extra_args_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=4, pady=(2, 4))
        if not self.extra_args_rows:
            self.extra_args_rows.append({"flag": tk.StringVar(value=""), "value": tk.StringVar(value="")})
        self._rebuild_extra_args()
        row += 1
        ttk.Button(frame, text="Add extra argument", command=self._extra_add_row).grid(row=row, column=0, sticky="w", pady=(0, 6))
        row += 1
        btns = ttk.Frame(frame)
        btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=10)
        ttk.Button(btns, text="Preview Command", command=self._preview_command).pack(side="left", padx=2)
        ttk.Button(btns, text="Copy Command", command=self._copy_command).pack(side="left", padx=2)
        self.start_btn = ttk.Button(btns, text="Start Server", command=self._start_server)
        self.start_btn.pack(side="left", padx=(20, 2))
        self.stop_btn = ttk.Button(btns, text="Stop Server", command=self._stop_server, state="disabled")
        self.stop_btn.pack(side="left", padx=2)
        ttk.Button(btns, text="Open in Browser", command=self._open_browser).pack(side="left", padx=2)
        ttk.Button(btns, text="Copy Tunnel URL", command=self._copy_tunnel_url).pack(side="left", padx=2)
        ttk.Button(btns, text="Clear Console", command=self._clear_console).pack(side="left", padx=(20, 2))
        row += 1
        ttk.Label(frame, text="Command preview:").grid(row=row, column=0, sticky="nw")
        self.preview_text = tk.Text(frame, height=4, wrap="word")
        self.preview_text.grid(row=row, column=1, columnspan=2, sticky="we", pady=4)

        self.preview_text.bind("<KeyRelease>", lambda e: self._auto_sync_from_preview())
        self.preview_text.bind("<ButtonRelease>", lambda e: self._auto_sync_from_preview())
        row += 1
        ttk.Label(frame, text="Console output:").grid(row=row, column=0, sticky="nw", pady=(8, 0))
        row += 1
        console_frame = ttk.Frame(frame)
        console_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=4)
        frame.rowconfigure(row, weight=1)
        frame.columnconfigure(1, weight=1)
        self.console = tk.Text(console_frame, height=20, wrap="none", bg="#111", fg="#0f0",
                                insertbackground="#0f0", font=("Consolas", 9))
        vsb = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        hsb = ttk.Scrollbar(console_frame, orient="horizontal", command=self.console.xview)
        self.console.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.console.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)

        row += 1
        ttk.Label(frame, text="Cloudflare Tunnel:").grid(row=row, column=0, sticky="w", pady=(12, 0))
        tunnel_btns = ttk.Frame(frame)
        tunnel_btns.grid(row=row, column=1, columnspan=2, sticky="w", pady=4)
        self.tunnel_start_btn = ttk.Button(tunnel_btns, text="Start Tunnel", command=self._start_tunnel)
        self.tunnel_start_btn.pack(side="left", padx=2)
        self.tunnel_stop_btn = ttk.Button(tunnel_btns, text="Stop Tunnel", command=self._stop_tunnel, state="disabled")
        self.tunnel_stop_btn.pack(side="left", padx=2)
        ttk.Label(tunnel_btns, textvariable=self.tunnel_status_var).pack(side="left", padx=(10, 0))

    def _rebuild_extra_args(self):
        if not hasattr(self, "extra_args_frame") or not self.extra_args_frame.winfo_exists():
            return
        for child in self.extra_args_frame.winfo_children():
            child.destroy()
        for idx, rd in enumerate(self.extra_args_rows):
            rf = ttk.Frame(self.extra_args_frame)
            rf.pack(fill="x", padx=2, pady=1)
            ttk.Label(rf, text=f"#{idx + 1}").pack(side="left")
            ttk.Entry(rf, textvariable=rd["flag"], width=32).pack(side="left", padx=(4, 4))
            ttk.Entry(rf, textvariable=rd["value"], width=32).pack(side="left", padx=(0, 4))
            ttk.Button(rf, text="Remove", width=8,
                       command=lambda i=idx: self._extra_remove_row(i)).pack(side="left")

    def _extra_add_row(self, flag="", value=""):
        self.extra_args_rows.append({"flag": tk.StringVar(value=flag), "value": tk.StringVar(value=value)})
        self._rebuild_extra_args()

    def _extra_remove_row(self, idx):
        if 0 <= idx < len(self.extra_args_rows):
            self.extra_args_rows.pop(idx)
        self._rebuild_extra_args()

    def _extra_get_args(self):
        args = []
        for rd in self.extra_args_rows:
            fl = rd["flag"].get().strip()
            val = rd["value"].get().strip()
            if fl:
                args.append(fl)
                if val:
                    args.append(val)
        return args

    def _set_extra_args(self, data):
        self.extra_args_rows = []
        if isinstance(data, str):
            toks = shlex.split(data) if data.strip() else []
            i = 0
            while i < len(toks):
                fl = toks[i]
                val = ""
                if i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                    val = toks[i + 1]
                    i += 1
                self.extra_args_rows.append({"flag": tk.StringVar(value=fl), "value": tk.StringVar(value=val)})
                i += 1
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self.extra_args_rows.append({"flag": tk.StringVar(value=item.get("flag", "")), "value": tk.StringVar(value=item.get("value", ""))})
                elif isinstance(item, str) and item.strip():
                    self.extra_args_rows.append({"flag": tk.StringVar(value=item), "value": tk.StringVar(value="")})
        if not self.extra_args_rows:
            self.extra_args_rows.append({"flag": tk.StringVar(value=""), "value": tk.StringVar(value="")})
        self._rebuild_extra_args()

    def _browse_workdir(self):
        path = filedialog.askdirectory(title="Select working directory")
        if path:
            self.workdir_var.set(path)

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, relief="sunken", padding=(6, 2))
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")
        ttk.Label(bar, text=" | ").pack(side="left")
        ttk.Label(bar, textvariable=self.tunnel_status_var).pack(side="left")

    def _render_field(self, parent, f, row):
        key = f["key"]
        kind = f["kind"]
        entry_widget = None
        if kind == KIND_FLAG:
            cb = ttk.Checkbutton(parent, text=f["label"], variable=self.vars[key]["var"])
            cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
            entry_widget = cb
        elif kind == KIND_BOOL:
            self._field_label(parent, f["label"], row, 0, padx=(8, 0))
            combo = make_combobox(parent, textvariable=self.vars[key]["var"],
                                values=["default", "on", "off"], state="readonly", width=12)
            combo.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            entry_widget = combo
        elif kind == KIND_CHOICE:
            inc = ttk.Checkbutton(parent, variable=self.vars[key]["include"])
            inc.grid(row=row, column=0, sticky="w", padx=(8, 0), pady=2)
            self._field_label(parent, f["label"], row, 1)
            choices = f.get("choices", [])
            combo = make_combobox(parent, textvariable=self.vars[key]["value"],
                                values=choices, state="readonly", width=22)
            combo.grid(row=row, column=2, sticky="w", padx=4, pady=2)
            entry_widget = combo
        elif kind in (KIND_VALUE, KIND_PATH):
            inc = ttk.Checkbutton(parent, variable=self.vars[key]["include"])
            inc.grid(row=row, column=0, sticky="w", padx=(8, 0), pady=2)
            self._field_label(parent, f["label"], row, 1)
            entry = ttk.Entry(parent, textvariable=self.vars[key]["value"], width=42)
            entry.grid(row=row, column=2, sticky="we", padx=4, pady=2)
            entry_widget = entry
            if kind == KIND_PATH:
                subtype = f.get("subtype", "file")
                btn = ttk.Button(parent, text="Browse...",
                                command=lambda k=key, st=subtype: self._browse_field(k, st))
                btn.grid(row=row, column=3, sticky="w", padx=2, pady=2)
        parent.columnconfigure(2, weight=1)
        help_text = f.get("help", "")
        if help_text:
            help_lbl = ttk.Label(parent, text="(?)", foreground="#0066cc", cursor="hand2")
            help_lbl.grid(row=row, column=4, sticky="w", padx=6)
            Tooltip(help_lbl, help_text)
            if entry_widget is not None:
                Tooltip(entry_widget, help_text)
        return row + 1

    def _field_label(self, parent, text, row, col, **kw):
        label = tk.Label(parent)
        label.configure(text=text)
        grid_kwargs = {"row": row, "column": col, "sticky": "w", "padx": 4, "pady": 2}
        grid_kwargs.update(kw)
        label.grid(**grid_kwargs)
        return label

    def _browse_field(self, key, subtype):
        if subtype == "dir":
            path = filedialog.askdirectory(title="Select directory")
        else:
            path = filedialog.askopenfilename(title="Select file")
        if path:
            self.vars[key]["value"].set(path)

    def _build_command_args(self):
        args = []
        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            v = self.vars[key]
            if kind == KIND_FLAG:
                if v["var"].get(): args.append(f["flag"])
            elif kind == KIND_BOOL:
                choice = v["var"].get()
                if choice == "on": args.append(f["on_flag"])
                elif choice == "off": args.append(f["off_flag"])
            elif kind == KIND_CHOICE:
                if key == "mmproj_auto":
                    if v["include"].get():
                        val = v["value"].get()
                        if val.startswith("enabled"): args.append("--mmproj-auto")
                        elif val.startswith("disabled"): args.append("--no-mmproj")
                    continue
                if v["include"].get():
                    val = v["value"].get().strip()
                    if val:
                        args.append(f["flag"])
                        args.append(val)
            elif kind in (KIND_VALUE, KIND_PATH):
                if v["include"].get():
                    val = v["value"].get().strip()
                    if val:
                        args.append(f["flag"])
                        args.append(val)
        args.extend(self._extra_get_args())
        return args

    def _full_command(self):
        exe = LLAMA_SERVER_EXE
        return [exe] + self._build_command_args()

    def _quote(self, s):
        if not s: return '""'
        if any(c in s for c in " \t\"'"):
            return '"' + s.replace('"', '\\"') + '"'
        return s

    def _command_string(self):
        return " ".join(self._quote(p) for p in self._full_command())

    def _preview_command(self):
        cmd = self._command_string()
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", cmd)

    def _on_field_changed(self, *args):
        if self._syncing_from_preview:
            return
        if not hasattr(self, "preview_text"):
            return
        try:
            self._preview_command()
        except tk.TclError:
            pass

    def _copy_command(self):
        self._preview_command()
        cmd = self.preview_text.get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.status_var.set("Command copied to clipboard.")

    def _log(self, text):
        self.console.insert("end", text)
        self.console.see("end")

    def _activate_server_state(self, running):
        if running:
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set(f"Running (PID {self.proc.pid})")
        else:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status_var.set("Idle")

    def _start_server(self):
        if self.proc is not None:
            messagebox.showwarning("Already running", "The server is already running.")
            return
        model_val = self.vars["model"]["value"].get().strip()
        if self.vars["model"]["include"].get() and not model_val:
            if not messagebox.askyesno("No model set", "The Model field is checked but empty. Start anyway?"):
                return
        cmd = self._full_command()
        self._preview_command()
        workdir = self.workdir_var.get().strip() or None
        self._log(f"\n$ {' '.join(self._quote(p) for p in cmd)}\n")
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Starting...\n")
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.proc = subprocess.Popen(
                cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
                creationflags=creation_flags
            )
        except FileNotFoundError:
            self._log("ERROR: executable not found. Check the path at the top of the window.\n")
            self.proc = None
            return
        except Exception as e:
            self._log(f"ERROR: failed to start process: {e}\n")
            self.proc = None
            return
        self._activate_server_state(True)
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()
        self._save_config()

    def _reader_worker(self):
        proc = self.proc
        if proc is None or proc.stdout is None:
            self.out_queue.put("__PROCESS_EXITED__:1")
            return
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                self.out_queue.put(line)
        except Exception as e:
            self.out_queue.put(f"[reader error] {e}\n")
        finally:
            try:
                code = proc.wait(timeout=2)
            except Exception:
                code = proc.poll()
            if code is None:
                code = 0
            self.out_queue.put(f"__PROCESS_EXITED__:{code}")

    def _poll_output(self):
        try:
            while True:
                line = self.out_queue.get_nowait()
                if isinstance(line, str) and line.startswith("__PROCESS_EXITED__:"):
                    code = line.split(":", 1)[1]
                    self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Process exited (code {code})\n")
                    self.proc = None
                    self._activate_server_state(False)
                else:
                    self._log(line)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_output)

    def _stop_server(self):
        if self.proc is None:
            self._activate_server_state(False)
            return
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Stopping...\n")
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        except Exception as e:
            self._log(f"ERROR while stopping: {e}\n")
        self.proc = None
        self._activate_server_state(False)

    def _start_tunnel(self):
        if self.proc_tunnel is not None:
            messagebox.showwarning("Already running", "A tunnel is already running.")
            return
        if self.proc is None:
            if not messagebox.askyesno("Server not running",
                    "The llama-server is not running. A tunnel requires a running server. Continue?"):
                return
        cloudflared = os.path.join(APP_DIR, "cloudflared.exe")
        if not os.path.exists(cloudflared):
            messagebox.showerror("Not found", "cloudflared.exe not found in the application directory.")
            return
        self._log(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Cloudflare Tunnel...\n")
        try:
            self.proc_tunnel = subprocess.Popen(
                [cloudflared, "tunnel", "--url", "http://127.0.0.1:8080"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            self.tunnel_status_var.set("Running")
            self.tunnel_start_btn.config(state="disabled")
            self.tunnel_stop_btn.config(state="normal")
            threading.Thread(target=self._cloudflare_reader, daemon=True).start()
        except FileNotFoundError:
            self._log("ERROR: cloudflared.exe not found.\n")
        except Exception as e:
            self._log(f"ERROR: failed to start tunnel: {e}\n")
            self.proc_tunnel = None

    def _stop_tunnel(self):
        if self.proc_tunnel is None: return
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Stopping tunnel...\n")
        try: self.proc_tunnel.terminate()
        except Exception: pass
        self.proc_tunnel = None
        self.tunnel_status_var.set("Stopped")
        self.tunnel_start_btn.config(state="normal")
        self.tunnel_stop_btn.config(state="disabled")

    def _open_browser(self):
        host = self.vars["host"]["value"].get().strip() or "127.0.0.1"
        port = self.vars["port"]["value"].get().strip() or "8080"
        if host in ("0.0.0.0", ""): host = "127.0.0.1"
        webbrowser.open(f"http://{host}:{port}")

    def _clear_console(self):
        self.console.delete("1.0", "end")

    def _auto_sync_from_preview(self):
        try:
            text = self.preview_text.get("1.0", "end").strip()
            if not text:
                return
            tokens = shlex.split(text, posix=(os.name != "nt"))
            if not tokens:
                return
            tokens = tokens[1:]
            self._syncing_from_preview = True
            try:
                self._apply_parsed_args(tokens)
            finally:
                self._syncing_from_preview = False
        except Exception:
            pass

    def _apply_parsed_args(self, tokens):
        i = 0
        matched_flags = set()

        while i < len(tokens):
            token = tokens[i]
            val_override = None

            if "=" in token and token.startswith("-"):
                parts = token.split("=", 1)
                token = parts[0]
                val_override = parts[1]

            if not token.startswith("-"):
                i += 1
                continue

            field_key = FLAG_TO_FIELD.get(token)
            if field_key is None:
                i += 1
                continue

            f = FIELDS_BY_KEY.get(field_key)
            if f is None:
                i += 1
                continue

            kind = f["kind"]
            v = self.vars.get(field_key)
            if v is None:
                i += 1
                continue

            if kind == KIND_FLAG:
                v["var"].set(True)
                matched_flags.add(field_key)

            elif kind == KIND_BOOL:
                on_f = f.get("on_flag", "")
                off_f = f.get("off_flag", "")
                if token == on_f:
                    v["var"].set("on")
                elif token == off_f:
                    v["var"].set("off")
                else:
                    v["var"].set("on")
                matched_flags.add(field_key)

            elif kind in (KIND_CHOICE, KIND_VALUE, KIND_PATH):
                val = ""
                if val_override is not None:
                    val = val_override
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    val = tokens[i + 1]
                    i += 1

                v["include"].set(True)
                v["value"].set(val)
                matched_flags.add(field_key)

            i += 1

        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            v = self.vars.get(key)
            if v is None or key in matched_flags:
                continue
            if kind == KIND_FLAG:
                v["var"].set(False)
            elif kind in (KIND_CHOICE, KIND_VALUE, KIND_PATH):
                if not f.get("essential", False):
                    v["include"].set(False)

    def _export_settings(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Settings As..."
        )
        if not path:
            return
        try:
            data = self._get_state()
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            self.status_var.set(f"Settings exported to {path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export settings: {e}")

    def _import_settings(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Import Settings From..."
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._set_state(data)
            self.status_var.set(f"Settings imported from {path}")
        except Exception as e:
            messagebox.showerror("Import Error", f"Could not import settings: {e}")

    def _get_state(self):
        state = {}
        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            v = self.vars[key]
            if kind == KIND_FLAG: state[key] = {"value": bool(v["var"].get())}
            elif kind == KIND_BOOL: state[key] = {"value": v["var"].get()}
            else: state[key] = {"include": bool(v["include"].get()), "value": v["value"].get()}
        return {
            "workdir": self.workdir_var.get(),
            "extra_args": [{"flag": rd["flag"].get(), "value": rd["value"].get()} for rd in self.extra_args_rows],
            "fields": state,
        }

    def _set_state(self, data):
        if not data: return
        self.workdir_var.set(data.get("workdir", ""))
        self._set_extra_args(data.get("extra_args"))
        fields = data.get("fields", {})
        for f in ALL_FIELDS:
            key = f["key"]
            if key not in fields: continue
            kind = f["kind"]
            v = self.vars[key]
            fv = fields[key]
            try:
                if kind == KIND_FLAG: v["var"].set(bool(fv.get("value", False)))
                elif kind == KIND_BOOL: v["var"].set(fv.get("value", "default"))
                else:
                    v["include"].set(bool(fv.get("include", False)))
                    v["value"].set(fv.get("value", ""))
            except Exception: pass

    def _reset_defaults(self):
        if not messagebox.askyesno("Reset", "Reset all fields to their default values?"): return
        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            v = self.vars[key]
            if kind == KIND_FLAG: v["var"].set(bool(f.get("default", False)))
            elif kind == KIND_BOOL: v["var"].set(f.get("default", "default"))
            else:
                v["include"].set(bool(f.get("essential", False)))
                v["value"].set(str(f.get("default", "")))
        self.status_var.set("All fields reset to defaults.")

    def _load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._set_state(data)
                self.status_var.set("Loaded saved configuration.")
            except Exception as e:
                self.status_var.set(f"Could not load config: {e}")

    def _save_config(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._get_state(), fh, indent=2)
        except Exception: pass

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("Server running", "The server is still running. Stop it and exit?"):
                return
            try: self.proc.terminate()
            except Exception: pass
        if self.proc_tunnel is not None:
            try: self.proc_tunnel.terminate()
            except Exception: pass
        self._save_config()
        self.root.destroy()

    def _cloudflare_reader(self):
        for line in self.proc_tunnel.stdout:
            self.out_queue.put("[CF] " + line)
            m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
            if m:
                self.tunnel_url = m.group(0)
                self.out_queue.put(f"\n[CF] Public URL: {self.tunnel_url}\n")

    def _copy_tunnel_url(self):
        if not self.tunnel_url: return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.tunnel_url)

class ServerTabFrame(ttk.Frame):
    def __init__(self, parent, server_id, global_manager):
        super().__init__(parent)
        self.server_id = server_id
        self.global_manager = global_manager

        self.vars = {}
        self.proc = None
        self.proc_tunnel = None
        self.tunnel_url = ""
        self.reader_thread = None
        self.out_queue = queue.Queue()
        self._syncing_from_preview = False

        self.workdir_var = tk.StringVar(value="")
        self.extra_args_rows = []
        self.status_var = tk.StringVar(value="Idle")
        self.tunnel_status_var = tk.StringVar(value="Idle")
        self.port_error_var = tk.StringVar(value="")
        self.server_name_var = tk.StringVar(value=f"Server {server_id}")
        self.tab_content_frames = []

        self._build_vars()
        self._build_notebook()
        self._build_statusbar()

        name_row = ttk.Frame(self)
        name_row.pack(side="top", fill="x", padx=8, pady=(0, 4))
        ttk.Label(name_row, text="Server name:").pack(side="left")
        ttk.Entry(name_row, textvariable=self.server_name_var, width=30).pack(side="left", padx=6)

        self.port_error_label = tk.Label(
            self, textvariable=self.port_error_var, fg="#cc0000",
            font=("Segoe UI", 8, "bold"))
        self.port_error_label.pack(side="bottom", fill="x")

        self.vars["port"]["value"].trace_add(
            "write", lambda *a: self.global_manager.validate_ports())

        self.root = self.winfo_toplevel()
        self.root.after(150, self._poll_output)

    def _build_vars(self):
        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            if kind == KIND_FLAG:
                self.vars[key] = {"var": tk.BooleanVar(value=bool(f.get("default", False)))}
            elif kind == KIND_BOOL:
                self.vars[key] = {"var": tk.StringVar(value=f.get("default", "default"))}
            elif kind in (KIND_CHOICE, KIND_VALUE, KIND_PATH):
                include_default = bool(f.get("essential", False) or key in {"verbosity"})
                self.vars[key] = {
                    "include": tk.BooleanVar(value=include_default),
                    "value": tk.StringVar(value=str(f.get("default", ""))),
                }
            for v in self.vars[key].values():
                v.trace_add("write", self._on_field_changed)

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        self._build_quick_tab()
        for cat, fields in ALL_CATEGORIES:
            self._build_category_tab(cat, fields)
        self._build_run_tab()

    def _build_quick_tab(self):
        frame = ScrollFrame(self.notebook)
        self.notebook.add(frame, text="Quick Start")
        info = ttk.Label(frame.body,
            text="The most commonly used flags. Everything else is available in the "
                 "other tabs, fully unchecked by default.",
            wraplength=900, foreground="#444")
        info.grid(row=0, column=0, columnspan=5, sticky="w", padx=8, pady=(8, 12))

        content = ttk.Frame(frame.body)
        content.grid(row=1, column=0, columnspan=5, sticky="nsew")
        frame.body.columnconfigure(0, weight=1)

        fields = [FIELDS_BY_KEY[key] for key in QUICK_KEYS]
        self.tab_content_frames.append((content, fields, frame, "Quick Start"))
        self._refresh_tab_content(content, fields)

    def _build_category_tab(self, name, fields):
        frame = ScrollFrame(self.notebook)
        self.notebook.add(frame, text=name)

        content = ttk.Frame(frame.body)
        content.grid(row=0, column=0, columnspan=5, sticky="nsew", pady=(8, 0))
        frame.body.columnconfigure(0, weight=1)

        self.tab_content_frames.append((content, fields, frame, name))
        self._refresh_tab_content(content, fields)

    def _refresh_tab_content(self, parent, fields):
        for child in parent.winfo_children():
            for tip in list(getattr(child, "_tooltips", ())):
                tip.close()
            child.destroy()
        row = 0
        for f in fields:
            self._render_field(parent, f, row)
            row += 1

    def _build_run_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="Run")
        row = 0
        ttk.Label(frame, text="Working directory (optional):").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.workdir_var, width=70).grid(row=row, column=1, sticky="we", padx=4)
        ttk.Button(frame, text="Browse...", command=self._browse_workdir).grid(row=row, column=2)
        row += 1
        ttk.Label(frame, text="Extra arguments (one flag + optional value per row):").grid(row=row, column=0, sticky="nw", pady=(6, 0))
        self.extra_args_frame = ttk.Frame(frame)
        self.extra_args_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=4, pady=(2, 4))
        if not self.extra_args_rows:
            self.extra_args_rows.append({"flag": tk.StringVar(value=""), "value": tk.StringVar(value="")})
        self._rebuild_extra_args()
        row += 1
        ttk.Button(frame, text="Add extra argument", command=self._extra_add_row).grid(row=row, column=0, sticky="w", pady=(0, 6))
        row += 1
        btns = ttk.Frame(frame)
        btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=10)
        ttk.Button(btns, text="Preview Command", command=self._preview_command).pack(side="left", padx=2)
        ttk.Button(btns, text="Copy Command", command=self._copy_command).pack(side="left", padx=2)
        self.start_btn = ttk.Button(btns, text="Start Server", command=self._start_server)
        self.start_btn.pack(side="left", padx=(20, 2))
        self.stop_btn = ttk.Button(btns, text="Stop Server", command=self._stop_server, state="disabled")
        self.stop_btn.pack(side="left", padx=2)
        ttk.Button(btns, text="Open in Browser", command=self._open_browser).pack(side="left", padx=2)
        ttk.Button(btns, text="Copy Tunnel URL", command=self._copy_tunnel_url).pack(side="left", padx=2)
        ttk.Button(btns, text="Clear Console", command=self._clear_console).pack(side="left", padx=(20, 2))
        row += 1
        ttk.Label(frame, text="Command preview:").grid(row=row, column=0, sticky="nw")
        self.preview_text = tk.Text(frame, height=4, wrap="word")
        self.preview_text.grid(row=row, column=1, columnspan=2, sticky="we", pady=4)

        self.preview_text.bind("<KeyRelease>", lambda e: self._auto_sync_from_preview())
        self.preview_text.bind("<ButtonRelease>", lambda e: self._auto_sync_from_preview())
        row += 1
        ttk.Label(frame, text="Console output:").grid(row=row, column=0, sticky="nw", pady=(8, 0))
        row += 1
        console_frame = ttk.Frame(frame)
        console_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=4)
        frame.rowconfigure(row, weight=1)
        frame.columnconfigure(1, weight=1)
        self.console = tk.Text(console_frame, height=20, wrap="none", bg="#111", fg="#0f0",
                                insertbackground="#0f0", font=("Consolas", 9))
        vsb = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        hsb = ttk.Scrollbar(console_frame, orient="horizontal", command=self.console.xview)
        self.console.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.console.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)

        row += 1
        ttk.Label(frame, text="Cloudflare Tunnel:").grid(row=row, column=0, sticky="w", pady=(12, 0))
        tunnel_btns = ttk.Frame(frame)
        tunnel_btns.grid(row=row, column=1, columnspan=2, sticky="w", pady=4)
        self.tunnel_start_btn = ttk.Button(tunnel_btns, text="Start Tunnel", command=self._start_tunnel)
        self.tunnel_start_btn.pack(side="left", padx=2)
        self.tunnel_stop_btn = ttk.Button(tunnel_btns, text="Stop Tunnel", command=self._stop_tunnel, state="disabled")
        self.tunnel_stop_btn.pack(side="left", padx=2)
        ttk.Label(tunnel_btns, textvariable=self.tunnel_status_var).pack(side="left", padx=(10, 0))

    def _rebuild_extra_args(self):
        if not hasattr(self, "extra_args_frame") or not self.extra_args_frame.winfo_exists():
            return
        for child in self.extra_args_frame.winfo_children():
            child.destroy()
        for idx, rd in enumerate(self.extra_args_rows):
            rf = ttk.Frame(self.extra_args_frame)
            rf.pack(fill="x", padx=2, pady=1)
            ttk.Label(rf, text=f"#{idx + 1}").pack(side="left")
            ttk.Entry(rf, textvariable=rd["flag"], width=32).pack(side="left", padx=(4, 4))
            ttk.Entry(rf, textvariable=rd["value"], width=32).pack(side="left", padx=(0, 4))
            ttk.Button(rf, text="Remove", width=8,
                       command=lambda i=idx: self._extra_remove_row(i)).pack(side="left")

    def _extra_add_row(self, flag="", value=""):
        self.extra_args_rows.append({"flag": tk.StringVar(value=flag), "value": tk.StringVar(value=value)})
        self._rebuild_extra_args()

    def _extra_remove_row(self, idx):
        if 0 <= idx < len(self.extra_args_rows):
            self.extra_args_rows.pop(idx)
        self._rebuild_extra_args()

    def _extra_get_args(self):
        args = []
        for rd in self.extra_args_rows:
            fl = rd["flag"].get().strip()
            val = rd["value"].get().strip()
            if fl:
                args.append(fl)
                if val:
                    args.append(val)
        return args

    def _set_extra_args(self, data):
        self.extra_args_rows = []
        if isinstance(data, str):
            toks = shlex.split(data) if data.strip() else []
            i = 0
            while i < len(toks):
                fl = toks[i]
                val = ""
                if i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                    val = toks[i + 1]
                    i += 1
                self.extra_args_rows.append({"flag": tk.StringVar(value=fl), "value": tk.StringVar(value=val)})
                i += 1
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self.extra_args_rows.append({"flag": tk.StringVar(value=item.get("flag", "")), "value": tk.StringVar(value=item.get("value", ""))})
                elif isinstance(item, str) and item.strip():
                    self.extra_args_rows.append({"flag": tk.StringVar(value=item), "value": tk.StringVar(value="")})
        if not self.extra_args_rows:
            self.extra_args_rows.append({"flag": tk.StringVar(value=""), "value": tk.StringVar(value="")})
        self._rebuild_extra_args()

    def _browse_workdir(self):
        path = filedialog.askdirectory(title="Select working directory")
        if path:
            self.workdir_var.set(path)

    def _build_statusbar(self):
        bar = ttk.Frame(self, relief="sunken", padding=(6, 2))
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")
        ttk.Label(bar, text=" | ").pack(side="left")
        ttk.Label(bar, textvariable=self.tunnel_status_var).pack(side="left")

    def _render_field(self, parent, f, row):
        key = f["key"]
        kind = f["kind"]
        entry_widget = None
        if kind == KIND_FLAG:
            cb = ttk.Checkbutton(parent, text=f["label"], variable=self.vars[key]["var"])
            cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
            entry_widget = cb
        elif kind == KIND_BOOL:
            self._field_label(parent, f["label"], row, 0, padx=(8, 0))
            combo = make_combobox(parent, textvariable=self.vars[key]["var"],
                                values=["default", "on", "off"], state="readonly", width=12)
            combo.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            entry_widget = combo
        elif kind == KIND_CHOICE:
            inc = ttk.Checkbutton(parent, variable=self.vars[key]["include"])
            inc.grid(row=row, column=0, sticky="w", padx=(8, 0), pady=2)
            self._field_label(parent, f["label"], row, 1)
            choices = f.get("choices", [])
            combo = make_combobox(parent, textvariable=self.vars[key]["value"],
                                values=choices, state="readonly", width=22)
            combo.grid(row=row, column=2, sticky="w", padx=4, pady=2)
            entry_widget = combo
        elif kind in (KIND_VALUE, KIND_PATH):
            inc = ttk.Checkbutton(parent, variable=self.vars[key]["include"])
            inc.grid(row=row, column=0, sticky="w", padx=(8, 0), pady=2)
            self._field_label(parent, f["label"], row, 1)
            entry = ttk.Entry(parent, textvariable=self.vars[key]["value"], width=42)
            entry.grid(row=row, column=2, sticky="we", padx=4, pady=2)
            entry_widget = entry
            if kind == KIND_PATH:
                subtype = f.get("subtype", "file")
                btn = ttk.Button(parent, text="Browse...",
                                command=lambda k=key, st=subtype: self._browse_field(k, st))
                btn.grid(row=row, column=3, sticky="w", padx=2, pady=2)
        parent.columnconfigure(2, weight=1)
        help_text = f.get("help", "")
        if help_text:
            help_lbl = ttk.Label(parent, text="(?)", foreground="#0066cc", cursor="hand2")
            help_lbl.grid(row=row, column=4, sticky="w", padx=6)
            Tooltip(help_lbl, help_text)
            if entry_widget is not None:
                Tooltip(entry_widget, help_text)
        return row + 1

    def _field_label(self, parent, text, row, col, **kw):
        label = tk.Label(parent)
        label.configure(text=text)
        grid_kwargs = {"row": row, "column": col, "sticky": "w", "padx": 4, "pady": 2}
        grid_kwargs.update(kw)
        label.grid(**grid_kwargs)
        return label

    def _browse_field(self, key, subtype):
        if subtype == "dir":
            path = filedialog.askdirectory(title="Select directory")
        else:
            path = filedialog.askopenfilename(title="Select file")
        if path:
            self.vars[key]["value"].set(path)

    def _build_command_args(self):
        args = []
        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            v = self.vars[key]
            if kind == KIND_FLAG:
                if v["var"].get(): args.append(f["flag"])
            elif kind == KIND_BOOL:
                choice = v["var"].get()
                if choice == "on": args.append(f["on_flag"])
                elif choice == "off": args.append(f["off_flag"])
            elif kind == KIND_CHOICE:
                if key == "mmproj_auto":
                    if v["include"].get():
                        val = v["value"].get()
                        if val.startswith("enabled"): args.append("--mmproj-auto")
                        elif val.startswith("disabled"): args.append("--no-mmproj")
                    continue
                if v["include"].get():
                    val = v["value"].get().strip()
                    if val:
                        args.append(f["flag"])
                        args.append(val)
            elif kind in (KIND_VALUE, KIND_PATH):
                if v["include"].get():
                    val = v["value"].get().strip()
                    if val:
                        args.append(f["flag"])
                        args.append(val)
        args.extend(self._extra_get_args())
        return args

    def _full_command(self):
        exe = LLAMA_SERVER_EXE
        return [exe] + self._build_command_args()

    def _quote(self, s):
        if not s: return '""'
        if any(c in s for c in " \t\"'"):
            return '"' + s.replace('"', '\\"') + '"'
        return s

    def _command_string(self):
        return " ".join(self._quote(p) for p in self._full_command())

    def _preview_command(self):
        cmd = self._command_string()
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", cmd)

    def _on_field_changed(self, *args):
        if self._syncing_from_preview:
            return
        if not hasattr(self, "preview_text"):
            return
        try:
            self._preview_command()
        except tk.TclError:
            pass

    def _copy_command(self):
        self._preview_command()
        cmd = self.preview_text.get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.status_var.set("Command copied to clipboard.")

    def _log(self, text):
        self.console.insert("end", text)
        self.console.see("end")

    def _activate_server_state(self, running):
        if running:
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set(f"Running (PID {self.proc.pid})")
        else:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status_var.set("Idle")

    def _start_server(self):
        if self.proc is not None:
            messagebox.showwarning("Already running", "The server is already running.")
            return
        model_val = self.vars["model"]["value"].get().strip()
        if self.vars["model"]["include"].get() and not model_val:
            if not messagebox.askyesno("No model set", "The Model field is checked but empty. Start anyway?"):
                return
        cmd = self._full_command()
        self._preview_command()
        workdir = self.workdir_var.get().strip() or None
        self._log(f"\n$ {' '.join(self._quote(p) for p in cmd)}\n")
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Starting...\n")
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.proc = subprocess.Popen(
                cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
                creationflags=creation_flags
            )
        except FileNotFoundError:
            self._log("ERROR: executable not found. Check the path at the top of the window.\n")
            self.proc = None
            return
        except Exception as e:
            self._log(f"ERROR: failed to start process: {e}\n")
            self.proc = None
            return
        self._activate_server_state(True)
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()
        self.global_manager._save_config()

    def _reader_worker(self):
        proc = self.proc
        if proc is None or proc.stdout is None:
            self.out_queue.put("__PROCESS_EXITED__:1")
            return
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                self.out_queue.put(line)
        except Exception as e:
            self.out_queue.put(f"[reader error] {e}\n")
        finally:
            try:
                code = proc.wait(timeout=2)
            except Exception:
                code = proc.poll()
            if code is None:
                code = 0
            self.out_queue.put(f"__PROCESS_EXITED__:{code}")

    def _poll_output(self):
        try:
            while True:
                line = self.out_queue.get_nowait()
                if isinstance(line, str) and line.startswith("__PROCESS_EXITED__:"):
                    code = line.split(":", 1)[1]
                    self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Process exited (code {code})\n")
                    self.proc = None
                    self._activate_server_state(False)
                else:
                    self._log(line)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_output)

    def _stop_server(self):
        if self.proc is None:
            self._activate_server_state(False)
            return
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Stopping...\n")
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        except Exception as e:
            self._log(f"ERROR while stopping: {e}\n")
        self.proc = None
        self._activate_server_state(False)

    def _start_tunnel(self):
        if self.proc_tunnel is not None:
            messagebox.showwarning("Already running", "A tunnel is already running.")
            return
        if self.proc is None:
            if not messagebox.askyesno("Server not running",
                    "The llama-server is not running. A tunnel requires a running server. Continue?"):
                return
        cloudflared = os.path.join(APP_DIR, "cloudflared.exe")
        if not os.path.exists(cloudflared):
            messagebox.showerror("Not found", "cloudflared.exe not found in the application directory.")
            return
        self._log(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Cloudflare Tunnel...\n")
        try:
            self.proc_tunnel = subprocess.Popen(
                [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{self.get_port()}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            self.tunnel_status_var.set("Running")
            self.tunnel_start_btn.config(state="disabled")
            self.tunnel_stop_btn.config(state="normal")
            threading.Thread(target=self._cloudflare_reader, daemon=True).start()
        except FileNotFoundError:
            self._log("ERROR: cloudflared.exe not found.\n")
        except Exception as e:
            self._log(f"ERROR: failed to start tunnel: {e}\n")
            self.proc_tunnel = None

    def _stop_tunnel(self):
        if self.proc_tunnel is None: return
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] Stopping tunnel...\n")
        try: self.proc_tunnel.terminate()
        except Exception: pass
        self.proc_tunnel = None
        self.tunnel_status_var.set("Stopped")
        self.tunnel_start_btn.config(state="normal")
        self.tunnel_stop_btn.config(state="disabled")

    def _open_browser(self):
        host = self.vars["host"]["value"].get().strip() or "127.0.0.1"
        port = self.vars["port"]["value"].get().strip() or "8080"
        if host in ("0.0.0.0", ""): host = "127.0.0.1"
        webbrowser.open(f"http://{host}:{port}")

    def _clear_console(self):
        self.console.delete("1.0", "end")

    def _auto_sync_from_preview(self):
        try:
            text = self.preview_text.get("1.0", "end").strip()
            if not text:
                return
            tokens = shlex.split(text, posix=(os.name != "nt"))
            if not tokens:
                return
            tokens = tokens[1:]
            self._syncing_from_preview = True
            try:
                self._apply_parsed_args(tokens)
            finally:
                self._syncing_from_preview = False
        except Exception:
            pass

    def _apply_parsed_args(self, tokens):
        i = 0
        matched_flags = set()

        while i < len(tokens):
            token = tokens[i]
            val_override = None

            if "=" in token and token.startswith("-"):
                parts = token.split("=", 1)
                token = parts[0]
                val_override = parts[1]

            if not token.startswith("-"):
                i += 1
                continue

            field_key = FLAG_TO_FIELD.get(token)
            if field_key is None:
                i += 1
                continue

            f = FIELDS_BY_KEY.get(field_key)
            if f is None:
                i += 1
                continue

            kind = f["kind"]
            v = self.vars.get(field_key)
            if v is None:
                i += 1
                continue

            if kind == KIND_FLAG:
                v["var"].set(True)
                matched_flags.add(field_key)

            elif kind == KIND_BOOL:
                on_f = f.get("on_flag", "")
                off_f = f.get("off_flag", "")
                if token == on_f:
                    v["var"].set("on")
                elif token == off_f:
                    v["var"].set("off")
                else:
                    v["var"].set("on")
                matched_flags.add(field_key)

            elif kind in (KIND_CHOICE, KIND_VALUE, KIND_PATH):
                val = ""
                if val_override is not None:
                    val = val_override
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    val = tokens[i + 1]
                    i += 1

                v["include"].set(True)
                v["value"].set(val)
                matched_flags.add(field_key)

            i += 1

        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            v = self.vars.get(key)
            if v is None or key in matched_flags:
                continue
            if kind == KIND_FLAG:
                v["var"].set(False)
            elif kind in (KIND_CHOICE, KIND_VALUE, KIND_PATH):
                if not f.get("essential", False):
                    v["include"].set(False)

    def _cloudflare_reader(self):
        for line in self.proc_tunnel.stdout:
            self.out_queue.put("[CF] " + line)
            m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
            if m:
                self.tunnel_url = m.group(0)
                self.out_queue.put(f"\n[CF] Public URL: {self.tunnel_url}\n")

    def _copy_tunnel_url(self):
        if not self.tunnel_url: return
        root = self.winfo_toplevel()
        root.clipboard_clear()
        root.clipboard_append(self.tunnel_url)

    def _reset_defaults(self):
        port_value = self.vars["port"]["value"].get()
        for f in ALL_FIELDS:
            key = f["key"]
            kind = f["kind"]
            v = self.vars[key]
            if kind == KIND_FLAG: v["var"].set(bool(f.get("default", False)))
            elif kind == KIND_BOOL: v["var"].set(f.get("default", "default"))
            else:
                v["include"].set(bool(f.get("essential", False)))
                v["value"].set(str(f.get("default", "")))
        self.vars["port"]["value"].set(port_value)
        self.global_manager.validate_ports()

    def get_port(self):
        if self.vars["port"]["include"].get():
            return self.vars["port"]["value"].get().strip()
        return "8080" 

    def set_port_error(self, is_error, msg=""):
        self.port_error_var.set(msg)
        if hasattr(self, 'start_btn'):
            if is_error:
                self.start_btn.config(state="disabled")
            elif self.proc is None:
                self.start_btn.config(state="normal")

    def _get_state(self):
        return {
            "server_name": self.server_name_var.get().strip() or f"Server {self.server_id}",
            "workdir": self.workdir_var.get(),
            "extra_args": [{"flag": rd["flag"].get(), "value": rd["value"].get()} for rd in self.extra_args_rows],
            "fields": {
                key: (
                    {"value": bool(v["var"].get())}
                    if kind == KIND_FLAG else
                    {"value": v["var"].get()}
                    if kind == KIND_BOOL else
                    {"include": bool(v["include"].get()), "value": v["value"].get()}
                )
                for f in ALL_FIELDS
                for key, kind, v in [(f["key"], f["kind"], self.vars[f["key"]])]
            },
        }

    def _set_state(self, data):
        if not data:
            return
        self.server_name_var.set(data.get("server_name", self.server_name_var.get()))
        self.workdir_var.set(data.get("workdir", ""))
        self._set_extra_args(data.get("extra_args"))
        fields = data.get("fields", {})
        for f in ALL_FIELDS:
            key = f["key"]
            if key not in fields:
                continue
            kind = f["kind"]
            v = self.vars[key]
            fv = fields[key]
            try:
                if kind == KIND_FLAG:
                    v["var"].set(bool(fv.get("value", False)))
                elif kind == KIND_BOOL:
                    v["var"].set(fv.get("value", "default"))
                else:
                    v["include"].set(bool(fv.get("include", False)))
                    v["value"].set(fv.get("value", ""))
            except Exception:
                pass
        self.global_manager._update_tab_name(self.server_id)

    @staticmethod
    def _field_search_terms(f):
        terms = [f.get("label", ""), f.get("key", ""), f.get("help", ""), f.get("category", "")]
        for k in ("flag", "on_flag", "off_flag"):
            val = f.get(k)
            if val:
                terms.append(val)
        terms.extend(f.get("aliases", []))
        return terms

    def apply_search(self, query=""):
        """Filter visible parameters in all tabs. Returns number of matched fields."""
        q = (query or "").strip().lower()
        matched = set()
        for content, fields, frame, name in self.tab_content_frames:
            if q:
                shown = [f for f in fields
                         if any(q in t.lower() for t in self._field_search_terms(f))]
                matched.update(f["key"] for f in shown)
            else:
                shown = fields
            self._refresh_tab_content(content, shown)
        return len(matched)

class MultiServerLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Llama-Server Launcher")
        self.root.geometry("1120x800")

        self.num_servers_var = tk.IntVar(value=1)  
        self.servers = []
        self.server_names = {} 

        self._build_menu()
        self._build_outer_notebook()
        self.set_server_count(self.num_servers_var.get())
        self._load_config()

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Server Count Settings", command=self._configure_server_count)
        filemenu.add_separator()
        filemenu.add_command(label="Export Settings...", command=self._export_settings)
        filemenu.add_command(label="Import Settings...", command=self._import_settings)
        filemenu.add_separator()
        filemenu.add_command(label="Reset All To Defaults", command=self._reset_defaults)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)
        
        self.root.config(menu=menubar)

    def _build_outer_notebook(self):
        search_bar = ttk.Frame(self.root, padding=(8, 4))
        search_bar.pack(side="top", fill="x")
        ttk.Label(search_bar, text="Search parameters:").pack(side="left")
        self.search_var = tk.StringVar(value="")
        self.search_var.trace_add("write", self._on_search_changed)
        self.search_entry = ttk.Entry(search_bar, textvariable=self.search_var)
        self.search_entry.pack(side="left", padx=(4, 8), fill="x", expand=True)
        self.search_count_var = tk.StringVar(value="")
        ttk.Label(search_bar, textvariable=self.search_count_var,
                  foreground="#555").pack(side="left", padx=(0, 8))
        ttk.Button(search_bar, text="Clear", command=self._clear_search).pack(side="left")

        self.outer_notebook = ttk.Notebook(self.root)
        self.outer_notebook.pack(side="top", fill="both", expand=True, padx=4, pady=4)

    def _on_search_changed(self, *args):
        q = self.search_var.get()
        if not q.strip():
            self.search_count_var.set("")
        else:
            total = 0
            for server in self.servers:
                total += server.apply_search(q)
            self.search_count_var.set(f"{total} matching parameter(s)")

    def _clear_search(self):
        self.search_var.set("")
        self.search_entry.focus_set()

    def set_server_count(self, count):
        current_count = len(self.servers)

        if count > current_count:
            for i in range(current_count, count):
                server_id = i + 1
                server_frame = ServerTabFrame(self.outer_notebook, server_id, self)
                
                server_frame.vars["port"]["value"].set(str(8080 + i))
                
                default_name = f"Server {server_id}"
                self.server_names[server_id] = default_name
                server_frame.server_name_var.set(default_name)
                server_frame.server_name_var.trace_add("write", lambda *_, sid=server_id: self._update_tab_name(sid))
                
                self.servers.append(server_frame)
                self.outer_notebook.add(server_frame, text=default_name)
                server_frame.apply_search(self.search_var.get())
        
        elif count < current_count:
            for i in range(current_count - 1, count - 1, -1):
                if self.servers[i].proc is not None:
                    messagebox.showwarning("Warning", f"Server {i+1} is currently running. Stop it first.")
                    return
                self.outer_notebook.forget(self.servers[i])
                self.server_names.pop(i + 1, None)
                self.servers.pop(i)

        self.num_servers_var.set(len(self.servers))
        self.validate_ports()
    
    def _update_tab_name(self, server_id):
        server_frame = self.servers[server_id - 1]
        new_name = server_frame.server_name_var.get()
        self.server_names[server_id] = new_name
        self.outer_notebook.tab(server_id - 1, text=new_name)

    def validate_ports(self):
        port_map = {}
        duplicates = set()

        for idx, server in enumerate(self.servers):
            port = server.get_port()
            if port in port_map:
                duplicates.add(port)
                port_map[port].append(idx)
            else:
                port_map[port] = [idx]

            if idx < len(self.servers):
                current_name = self.server_names.get(idx + 1, f"Server {idx + 1}")
                if current_name:
                    self.outer_notebook.tab(idx, text=current_name)

        for idx, server in enumerate(self.servers):
            port = server.get_port()
            if port in duplicates:
                server.set_port_error(True, f"PORT CONFLICT: Port {port} is used by another server!")
            else:
                server.set_port_error(False, "")

    def _configure_server_count(self):
        top = tk.Toplevel(self.root)
        top.title("Server Count Settings")
        top.geometry("300x100")
        top.transient(self.root)
        top.grab_set()
        
        frame = ttk.Frame(top, padding=20)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Number of servers (1-10):", font=("Segoe UI", 10)).pack(anchor="w")

        count_var = tk.StringVar(value=str(self.num_servers_var.get()))
        entry = ttk.Entry(frame, textvariable=count_var, font=("Segoe UI", 12, "bold"), width=8, justify="center")
        entry.pack(pady=10)
        entry.focus_set()
        entry.select_range(0, "end")

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)

        def on_ok():
            try:
                value = int(count_var.get().strip())
            except (ValueError, tk.TclError):
                messagebox.showerror("Invalid value", "Please enter a whole number.", parent=top)
                return
            if not (1 <= value <= 10):
                messagebox.showerror("Out of range", "Server count must be between 1 and 10.", parent=top)
                return
            self.set_server_count(value)
            top.destroy()

        top.bind("<Return>", lambda e: on_ok())
        
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=top.destroy).pack(side="left", padx=5)



    def _get_all_states(self):
        return [server._get_state() for server in self.servers]

    def _set_all_states(self, states):
        for idx, state in enumerate(states):
            if idx < len(self.servers):
                self.servers[idx]._set_state(state)

    def _load_config(self):
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                servers = data.get("servers") or data.get("server_states") or []
                count = int(data.get("server_count", max(len(servers), len(self.servers))))
                if count != len(self.servers):
                    self.set_server_count(max(1, count))
                if servers:
                    self._set_all_states(servers)
                    self.validate_ports()
            elif isinstance(data, list):
                if data:
                    self.set_server_count(max(len(data), len(self.servers)))
                    self._set_all_states(data)
                    self.validate_ports()
        except Exception:
            pass

    def _save_config(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump({
                    "server_count": len(self.servers),
                    "servers": self._get_all_states(),
                }, fh, indent=2)
        except Exception:
            pass

    def _export_settings(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Settings As..."
        )
        if not path:
            return
        try:
            data = self._get_all_states()
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            messagebox.showinfo("Success", f"Settings exported to {path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export settings: {e}")

    def _import_settings(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Import Settings From..."
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                if len(data) > len(self.servers):
                    self.set_server_count(len(data))
                self._set_all_states(data)
            else:
                messagebox.showwarning("Warning", "File format appears to be for a single server. Importing to first server only.")
                if self.servers:
                    self.servers[0]._set_state(data)
            messagebox.showinfo("Success", f"Settings imported from {path}")
        except Exception as e:
            messagebox.showerror("Import Error", f"Could not import settings: {e}")

    def _reset_defaults(self):
        if not messagebox.askyesno("Reset", "Reset all servers to their default values?"):
            return
        for server in self.servers:
            server._reset_defaults()
        messagebox.showinfo("Success", "All servers reset to defaults.")

    def _on_close(self):
        for idx, server in enumerate(self.servers):
            if server.proc is not None or server.proc_tunnel is not None:
                if not messagebox.askyesno("Exit", f"Server {idx+1} is active. Stop all and exit?"):
                    return
                break

        for server in self.servers:
            if server.proc:
                try: server.proc.terminate()
                except Exception: pass
            if server.proc_tunnel:
                try: server.proc_tunnel.terminate()
                except Exception: pass

        self._save_config()
        self.root.destroy()

def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names(): style.theme_use("vista")
        elif "clam" in style.theme_names(): style.theme_use("clam")
    except Exception: pass
    app = MultiServerLauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
