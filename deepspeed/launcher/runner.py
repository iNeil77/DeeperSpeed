# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
DeepSpeed runner is the main front-end to launching multi-worker
training jobs with DeepSpeed. By default this uses pdsh to parallel
ssh into multiple worker nodes and launch all the necessary processes
per rank for training.
"""

import os
import re
import sys
import shlex
import json
import base64
import argparse
import subprocess
from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from ..nebula.constants import NEBULA_EXPORT_ENVS
from ..utils import logger
from deepspeed.accelerator import get_accelerator

DLTS_HOSTFILE = "/job/hostfile"
EXPORT_ENVS = ['MLFLOW', 'NCCL', 'PYTHON', 'MV2', 'UCX']
EXPORT_ENVS += NEBULA_EXPORT_ENVS
DEEPSPEED_ENVIRONMENT_NAME = os.getenv("DS_ENV_FILE", ".deepspeed_env")
DEEPSPEED_ENVIRONMENT_PATHS = [os.path.expanduser("~"), '.']
PDSH_MAX_FAN_OUT = 1024

# On AISC compute, each node sets environment variables independently, want to prevent
# exporting rank-0 env variables in case of heterogeneous compute.
EXCLUDE_ENVS = {'AISC_JOB_NAME': ['NCCL_IB_HCA', 'UCX_NET_DEVICES']}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="DeepSpeed runner to help launch distributed "
                                     "multi-node/multi-gpu training jobs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--num_nodes",
                        type=int,
                        default=-1,
                        help="Total number of worker nodes to run on, this will use "
                        "the top N hosts from the given hostfile.")

    parser.add_argument("--num_gpus",
                        "--num_accelerators",
                        type=int,
                        default=-1,
                        help="Max number of GPUs to use on each node, will use "
                        "[0:N) GPU ids on each node.")

    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="(optional) Port used by PyTorch distributed for "
                        "communication during training.")

    parser.add_argument("--master_addr",
                        default="",
                        type=str,
                        help="(optional) IP address of node 0, will be "
                        "inferred via 'hostname -I' if not specified.")

    parser.add_argument("--launcher_args",
                        default="",
                        type=str,
                        help="(optional) pass launcher specific arguments as a "
                        "single quoted argument.")

    parser.add_argument("--module",
                        action="store_true",
                        help="Change each process to interpret the launch "
                        "script as a Python module, executing with the same "
                        "behavior as 'python -m'.")

    parser.add_argument("--no_python",
                        action="store_true",
                        help="Skip prepending the training script with "
                        "'python' - just execute it directly.")

    parser.add_argument("--save_pid",
                        action="store_true",
                        help="Save file containing launcher process id (pid) at /tmp/<main-pid>.ds, "
                        "where <main-pid> is the pid of the first process that invoked `deepspeed`. "
                        "Useful when launching deepspeed processes programmatically.")

    parser.add_argument("--enable_each_rank_log",
                        default="None",
                        type=str,
                        help="redirect the stdout and stderr from each rank into different log files")

    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="A comment that can be used for metadata. Used to pass --comment argument to srun in Slurm launcher"
    )

    parser.add_argument(
        "--account",
        default="",
        type=str,
        help="Used to pass --account argument to srun in Slurm launcher"
    )

    parser.add_argument("user_script", type=str, help="User script to launch, followed by any required "
                        "arguments.")

    parser.add_argument('user_args', nargs=argparse.REMAINDER)

    return parser.parse_args(args=args)


def encode_world_info(world_info):
    world_info_json = json.dumps(world_info).encode('utf-8')
    world_info_base64 = base64.urlsafe_b64encode(world_info_json).decode('utf-8')
    return world_info_base64


def main(args=None):
    args = parse_args(args)

    # For when argparse interprets remaining args as a single string
    args.user_args = shlex.split(" ".join(list(map(lambda x: x if x.startswith("-") else f'"{x}"', args.user_args))))

    # respect CUDA_VISIBLE_DEVICES for a single node and no explicit resource filters
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    active_resources = {}
    device_count = get_accelerator().device_count()
    if device_count == 0:
        raise RuntimeError("Unable to proceed, no GPU resources available")
    elif device_count < args.num_gpus:
        raise ValueError(f"num_gpus ({args.num_gpus}) cannot be more than the number of GPUs present ({device_count})")
    
    if len(cuda_visible_devices):
        visible_devices = cuda_visible_devices.split(",")
    else:
        visible_devices = list(range(device_count))
    visible_devices = [int(item) for item in visible_devices]
    active_resources['localhost'] = visible_devices[:args.num_gpus]
    args.master_addr = "127.0.0.1"

    env = os.environ.copy()

    # encode world info as base64 to make it easier to pass via command line
    world_info_base64 = encode_world_info(active_resources)

    deepspeed_launch = [
        sys.executable, "-u", "-m", "deepspeed.launcher.launch", f"--world_info={world_info_base64}",
        f"--master_addr={args.master_addr}", f"--master_port={args.master_port}"
    ]
    if args.no_python:
        deepspeed_launch.append("--no_python")
    if args.module:
        deepspeed_launch.append("--module")
    if args.save_pid:
        deepspeed_launch += ["--save_pid", f"{os.getpid()}"]
    if args.enable_each_rank_log:
        deepspeed_launch.append(f"--enable_each_rank_log={args.enable_each_rank_log}")
    cmd = deepspeed_launch + [args.user_script] + args.user_args

    logger.info(f"cmd = {' '.join(cmd)}")

    result = subprocess.Popen(cmd, env=env)
    result.wait()

    # In case of failure must propagate the error-condition back to the caller (usually shell). The
    # actual error and traceback should have been printed in the subprocess, so in order to avoid
    # unnecessary noise we just quietly exit here with the same code as the subprocess
    if result.returncode > 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
