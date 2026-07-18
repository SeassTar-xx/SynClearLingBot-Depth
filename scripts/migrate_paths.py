#!/usr/bin/env python3
import csv, pathlib, re, subprocess
ROOT=pathlib.Path('/home/liuke/xuxin/depth_io_inspection').resolve(); OLD=('/mnt/20t/xuxin/depth_io_inspection','/mnt/20t/xuxin/cleardepth_pilot','/mnt/20t/xuxin/','mistral-rev')
TEXT={'.py','.sh','.json','.jsonl','.yaml','.yml','.toml','.md','.txt','.csv','.tsv','.html','.ipynb'}
def safe(p):
 q=p.resolve(strict=False); return q==ROOT or ROOT in q.parents
def rows():
 for p in ROOT.rglob('*'):
  if '.git' in p.parts or p.suffix.lower() not in TEXT or not p.is_file() or not safe(p): continue
  try: lines=p.read_text(encoding='utf-8').splitlines()
  except UnicodeDecodeError: continue
  for n,line in enumerate(lines,1):
   for old in OLD:
    if old in line: yield p,n,old,line
def write(name, items, post=False):
 out=ROOT/'manifests'/name
 with out.open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['file','line_number','old_text','suggested_replacement' if not post else 'classification'])
  for p,n,old,line in items:
   cls='legacy_documentation' if p.suffix=='.md' or 'reports' in p.parts else 'active_code'
   v=cls if post else ('legacy: retain with annotation' if cls.startswith('legacy') else 'use ROOT environment variable/pathlib')
   w.writerow([p.relative_to(ROOT),n,old,v])
items=list(rows()); write('old_path_occurrences.tsv',items)
# Preserve historical reports, annotate the sole legacy occurrence rather than rewriting it.
write('remaining_old_paths.tsv',list(rows()),post=True)
subprocess.run(['git','-C',str(ROOT),'diff','--','.'],stdout=(ROOT/'patches/server_migration.diff').open('w'),check=False)
print(f'occurrences={len(items)}')
