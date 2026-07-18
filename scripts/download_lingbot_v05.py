#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,sys,datetime,os
R=pathlib.Path('/home/liuke/xuxin/depth_io_inspection').resolve(); T=R/'models/lingbot-depth-v0.5'; log=R/'logs/lingbot_v05_download.log'; expected=1284837952; known='b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
if not T.exists():
 cmd=['git','-c','http.proxy=http://127.0.0.1:7897','clone','https://www.modelscope.cn/Robbyant/lingbot-depth-pretrain-vitl-14-v0.5.git',str(T)]
 with log.open('a') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
p=next(T.rglob('model.pt'),None)
data={'model_name':'LingBot-Depth-v0.5','model_id':'Robbyant/lingbot-depth-pretrain-vitl-14-v0.5','source':'https://modelscope.cn/models/Robbyant/lingbot-depth-pretrain-vitl-14-v0.5','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':str(p) if p else None}
if p:
 data.update(size_bytes=p.stat().st_size,sha256=sha(p),expected_size_bytes=expected,reference_sha256=known,is_lfs_pointer=p.read_bytes()[:64].startswith(b'version https://git-lfs.github.com/spec'),size_matches=p.stat().st_size==expected)
 try:
  import torch; x=torch.load(p,map_location='cpu',weights_only=True); data['torch_load_weights_only']=True; data['contains_model']=isinstance(x,dict) and 'model' in x; data['contains_model_config']=isinstance(x,dict) and 'model_config' in x
 except Exception as e:data['torch_load_weights_only_error']=repr(e)
 (R/'models/ACTIVE_MODEL.txt').write_text('\n'.join(f'{k}={v}' for k,v in data.items() if k in ['model_name','model_id','checkpoint','sha256'])+'\n',encoding='utf-8')
(R/'manifests/lingbot_v05_checkpoint.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps(data,indent=2))
