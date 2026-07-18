#!/usr/bin/env python3
import argparse,csv,json,os,pathlib,shutil,struct,sys,time,urllib.request,zipfile,zlib,io,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
R=pathlib.Path('/home/liuke/xuxin/SynClearLingBot-Depth').resolve();URL='https://tams.informatik.uni-hamburg.de/research/datasets/cleardepth_dataset/transparent_dataset1.zip';LOG=R/'logs/syncleardepth_test_download.log';LOCK=threading.Lock();PROXY={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
class RR(io.RawIOBase):
 def __init__(self,n):self.n=n;self.p=0;self.op=urllib.request.build_opener(urllib.request.ProxyHandler(PROXY))
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.p
 def seek(self,x,w=0):self.p=max(0,min(self.n,x if w==0 else self.p+x if w==1 else self.n+x));return self.p
 def readinto(self,b):
  n=min(len(b),self.n-self.p)
  if n<=0:return 0
  a=self.p;q=urllib.request.Request(URL,headers={'Range':f'bytes={a}-{a+n-1}'})
  with self.op.open(q,timeout=120) as r:
   if r.status!=206:raise RuntimeError(f'Range HTTP {r.status}')
   x=r.read(n)
  with LOCK:
   with LOG.open('a') as f:f.write(f'{time.time()}\t{a}\t{a+len(x)-1}\t{len(x)}\n')
  b[:len(x)]=x;self.p+=len(x);return len(x)
def crc(p):
 c=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):c=zlib.crc32(b,c)
 return c&0xffffffff
def dst(root,name):
 q=pathlib.PurePosixPath(name)
 if q.is_absolute() or '..' in q.parts:raise ValueError(name)
 p=root.joinpath(*q.parts)
 if root not in p.resolve(strict=False).parents:raise ValueError(name)
 cur=root
 for x in q.parts[:-1]:
  cur/=x
  if cur.exists() and cur.is_symlink():raise ValueError('symlink parent')
 return p
def get(name,info,root,size):
 d=dst(root,name);d.parent.mkdir(parents=True,exist_ok=True)
 if d.exists() and d.is_file() and d.stat().st_size==info.file_size and crc(d)==info.CRC:return name,'resume_valid'
 part=d.with_name(d.name+'.part')
 try:
  with zipfile.ZipExtFile(RR(size),'r',info,None,True) as src,part.open('wb') as out:shutil.copyfileobj(src,out,8<<20)
  if part.stat().st_size!=info.file_size or crc(part)!=info.CRC:raise RuntimeError('size/CRC mismatch')
  os.replace(part,d);return name,'downloaded'
 except Exception as e:return name,'failed:'+repr(e)
p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--source',default='auto');p.add_argument('--resume',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=R/'data/syncleardepth/raw_subset';out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in pathlib.Path(a.split).read_text().splitlines() if x];need={v for x in rows for k,v in x.items() if k in {'left_rgb_member','right_rgb_member','gt_depth_member','segmentation_member','instance_mask_member','normal_member','scene_info_member'} and v};size=json.loads((R/'manifests/syncleardepth_archive_summary.json').read_text())['archive_content_length'];z=zipfile.ZipFile(RR(size));infos={x.filename:x for x in z.infolist()};missing=need-set(infos)
if missing:raise SystemExit(f'missing index entries: {len(missing)}')
res=[]
with ThreadPoolExecutor(max_workers=a.workers) as ex:
 for f in as_completed([ex.submit(get,n,infos[n],out,size) for n in sorted(need)]):
  res.append(f.result())
  if len(res)%100==0:print(f'completed={len(res)}/{len(need)}',flush=True)
with (R/'manifests/syncleardepth_downloaded_members.tsv').open('w') as f:
 f.write('archive_path\tstatus\n');f.writelines(f'{n}\t{s}\n' for n,s in sorted(res))
failed=[x for x in res if x[1].startswith('failed:')];print(json.dumps({'done':len(res)-len(failed),'failed':len(failed)}));sys.exit(1 if failed else 0)
