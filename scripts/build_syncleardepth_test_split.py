#!/usr/bin/env python3
import argparse,csv,hashlib,json,pathlib,random,datetime
R=pathlib.Path('/home/liuke/xuxin/depth_io_inspection').resolve()
p=argparse.ArgumentParser();p.add_argument('--all-samples',default=str(R/'manifests/syncleardepth_all_samples.jsonl'));p.add_argument('--ratio',type=float,default=.2);p.add_argument('--seed',type=int,default=20260718);p.add_argument('--split-name',default='test_20pct_v1');p.add_argument('--group-key',default='auto');p.add_argument('--list-only',action='store_true');a=p.parse_args(); out=R/'data/syncleardepth/test_split'
if a.split_name!='test_20pct_v1': raise SystemExit('Only requested immutable v1 supported by this command')
if (out/'test_ids.txt').exists(): raise SystemExit('v1 exists: refusing silent overwrite')
rows=[json.loads(x) for x in pathlib.Path(a.all_samples).read_text(encoding='utf8').splitlines() if x.strip()]; good=[x for x in rows if x.get('is_complete') is True]
if not good: raise SystemExit('No verified complete samples; run index and manifest inspection first')
# Sequence is preferred group; fallback scene/frame. Groups are selected whole within each scene.
for x in good:x['_group']=x.get('sequence') if x.get('sequence') not in (None,'','unknown') else x['sample_id']
scenes=sorted({x.get('scene','unknown') for x in good}); chosen=[]; limitations=[]
for sc in scenes:
 xs=[x for x in good if x.get('scene','unknown')==sc]; gs={}
 for x in xs:gs.setdefault(x['_group'],[]).append(x)
 target=round(len(xs)*a.ratio); target=max(1,target) if len(gs)>1 else 0
 if len(gs)==1: limitations.append(f'{sc}: only one group; excluded to avoid full-scene leakage');continue
 order=sorted(gs,key=lambda z:hashlib.sha256(f'{a.seed}:{z}'.encode()).hexdigest()); n=0
 for g in order:
  if n>=target:break
  chosen+=gs[g];n+=len(gs[g])
if not chosen: raise SystemExit('group constraint produced no test samples')
if a.list_only:
 print(json.dumps({'all_complete':len(good),'test':len(chosen),'ratio':len(chosen)/len(good),'group_key':'sequence','scene_counts':{s:sum(x.get('scene')==s for x in chosen)},'limitations':limitations},indent=2));raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True); chosen=sorted(chosen,key=lambda x:x['sample_id']); (out/'test_ids.txt').write_text(''.join(x['sample_id']+'\n' for x in chosen),encoding='utf8');(out/'test_groups.txt').write_text(''.join(sorted({x['_group'] for x in chosen}))+'\n',encoding='utf8');(out/'test_samples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chosen),encoding='utf8')
with (out/'test_samples.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=chosen[0].keys());w.writeheader();w.writerows(chosen)
conf={'dataset':'SynClearDepth','split_name':a.split_name,'seed':a.seed,'target_ratio':a.ratio,'actual_ratio':len(chosen)/len(good),'group_key':'sequence','stratification_fields':['scene','sequence','object_count (when available)'],'selection_algorithm_version':'1.0 deterministic scene-local group hash','archive_url':'https://tams.informatik.uni-hamburg.de/research/datasets/cleardepth_dataset/transparent_dataset1.zip','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limitations':limitations};(out/'split_config.json').write_text(json.dumps(conf,indent=2)+'\n')
stats={'all_complete_samples':len(good),'test_samples':len(chosen),'actual_test_ratio':len(chosen)/len(good),'groups':len({x['_group'] for x in good}),'test_groups':len({x['_group'] for x in chosen}),'limitations':limitations};(out/'split_statistics.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
