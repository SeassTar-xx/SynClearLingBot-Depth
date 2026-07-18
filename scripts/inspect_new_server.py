#!/usr/bin/env python3
import os, pathlib, platform, shutil, subprocess
R=pathlib.Path('/home/liuke/xuxin/SynClearLingBot-Depth').resolve(); out=R/'logs/new_server_environment.txt'
def run(*c):
 try:return subprocess.run(c,text=True,capture_output=True,timeout=30).stdout+subprocess.run(c,text=True,capture_output=True,timeout=30).stderr
 except Exception as e:return repr(e)+'\n'
lines=[f'hostname={platform.node()}',run('date','-Is'),f'pwd={os.getcwd()}',run('df','-h','/home/liuke/xuxin'),run('du','-sh','/home/liuke/xuxin'),run('nvidia-smi')]
for x in ['python','git','curl','wget']:
 p=shutil.which(x); lines.append(f'{x}={p or "NOT_FOUND"}'); lines.append(run(p,'--version') if p else '')
try:
 import torch
 lines += [f'torch={torch.__version__}',f'torch_cuda={torch.version.cuda}',f'cuda_available={torch.cuda.is_available()}',f'device_count={torch.cuda.device_count()}']
 for i in range(torch.cuda.device_count()):
  d=torch.cuda.get_device_properties(i); lines.append(f'gpu[{i}]={d.name}; total_memory={d.total_memory}')
except Exception as e: lines.append('torch_error='+repr(e))
out.write_text('\n'.join(lines),encoding='utf-8')
print(out)
