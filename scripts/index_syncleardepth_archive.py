#!/usr/bin/env python3
"""Range-only SynClearDepth ZIP central-directory indexer; never downloads the ZIP body."""
import argparse,collections,csv,json,pathlib,time,urllib.request,zipfile,io
R=pathlib.Path('/home/liuke/xuxin/depth_io_inspection').resolve(); URL='https://tams.informatik.uni-hamburg.de/research/datasets/cleardepth_dataset/transparent_dataset1.zip'
class RR(io.RawIOBase):
 def __init__(self,url,size): self.url,self.size,self.pos,self.bytes=url,size,0,0
 def seekable(self): return True
 def readable(self): return True
 def tell(self): return self.pos
 def seek(self,o,w=0): self.pos=max(0,min(self.size,o if w==0 else self.pos+o if w==1 else self.size+o)); return self.pos
 def readinto(self,b):
  n=min(len(b),self.size-self.pos)
  if n<=0:return 0
  q=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'})
  with OP.open(q,timeout=90) as r:
   if getattr(r,'status',None)!=206: raise RuntimeError(f'Range unsupported: HTTP {getattr(r,"status",None)}')
   x=r.read(n)
  b[:len(x)]=x; self.pos+=len(x); self.bytes+=len(x); return len(x)
def meta(path):
 parts=path.split('/'); base=parts[-1].lower(); stem=pathlib.PurePosixPath(path).stem
 m='depth' if any(x in base for x in ['depth','disp','distance']) else 'segmentation' if any(x in base for x in ['seg','mask','label','instance']) else 'rgb' if any(x in base for x in ['left','right','rgb','color','image']) else 'metadata' if pathlib.PurePosixPath(path).suffix.lower() in ['.json','.xml','.txt','.csv'] else 'other'
 return (parts[0] if parts else '',parts[1] if len(parts)>1 else '',parts[2] if len(parts)>2 else '',stem,m)
p=argparse.ArgumentParser(); p.add_argument('--url',default=URL); p.add_argument('--out',default=str(R/'manifests')); a=p.parse_args(); out=pathlib.Path(a.out).resolve(); assert out==R/'manifests'
OP=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}))
req=urllib.request.Request(a.url,method='HEAD')
try:
 with OP.open(req,timeout=60) as h: hdr=dict(h.headers); size=int(hdr.get('Content-Length','0'))
except Exception as e: raise SystemExit(f'HEAD failed: {e}')
if not size: raise SystemExit('Missing Content-Length; refusing full download')
rr=RR(a.url,size)
try: z=zipfile.ZipFile(rr)
except Exception as e: raise SystemExit(f'central directory unavailable via range: {e}')
infos=z.infolist(); rows=[]; ext=collections.Counter(); top=collections.Counter(); scenes=collections.Counter(); mods=collections.Counter()
for i in infos:
 t,s,q,sid,m=meta(i.filename); ext[pathlib.PurePosixPath(i.filename).suffix.lower()]+=1; top[t]+=1; scenes[s]+=1; mods[m]+=1
 rows.append([i.filename,i.compress_size,i.file_size,f'{i.CRC:08x}',pathlib.PurePosixPath(i.filename).suffix.lower(),t,s,q,sid,m])
with (out/'syncleardepth_archive_listing.tsv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['archive_path','compressed_size','uncompressed_size','crc','extension','top_level_dir','scene','sequence','sample_id_candidate','modality_candidate']);w.writerows(rows)
s={'archive_url':a.url,'archive_member_count':len(rows),'archive_content_length':size,'total_compressed_member_bytes':sum(x[1] for x in rows),'total_uncompressed_member_bytes':sum(x[2] for x in rows),'top_level_dirs':top,'extensions':ext,'scene_counts':scenes,'modality_counts':mods,'zip64':any(i.header_offset>0xffffffff or i.file_size>0xffffffff for i in infos),'central_directory_range_bytes':rr.bytes}
(out/'syncleardepth_archive_summary.json').write_text(json.dumps(s,indent=2,default=dict)+'\n');print(json.dumps(s,indent=2,default=dict))
