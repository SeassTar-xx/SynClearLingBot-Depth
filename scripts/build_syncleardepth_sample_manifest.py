import csv,json,pathlib,collections,re
R=pathlib.Path("/home/liuke/xuxin/depth_io_inspection").resolve();L=R/"manifests/syncleardepth_archive_listing.tsv";O=R/"manifests"
rx=re.compile(r"^(Image|depth|segmentation|normal|disparity)(\d+)(?:_([LR]))?\.(png|exr|npy)$",re.I);ss=collections.defaultdict(dict);mm={}
for x in csv.DictReader(L.open(),delimiter="\t"):
 p=pathlib.PurePosixPath(x["archive_path"]);q=p.parts
 if len(q)<3:continue
 m=rx.match(p.name);sc=q[1]
 if m:ss[sc,m.group(2)][m.group(1).lower()+(("_"+m.group(3).lower()) if m.group(3) else "")]=x
 elif p.suffix.lower() in [".json",".txt",".csv"]:mm.setdefault(sc,x)
rows=[]
for (sc,fr),d in sorted(ss.items()):
 get=lambda k:d.get(k,{}).get("archive_path")
 vals={"left_rgb_member":get("image_l"),"right_rgb_member":get("image_r"),"gt_depth_member":get("depth_l"),"segmentation_member":get("segmentation_l"),"scene_info_member":mm.get(sc,{}).get("archive_path")}
 mis=[k for k,v in vals.items() if not v]
 rows.append(dict(sample_id=f"{sc}/{fr}",scene=re.match(r"[a-z]+",sc).group(0),sequence=sc,frame_id=fr,**vals,instance_mask_member=None,normal_member=get("normal_l"),camera_pose_available=False,object_count="unknown",object_ids=[],object_categories=[],width="unknown",height="unknown",compressed_bytes=sum(int(v.get("compressed_size",0)) for v in d.values()),missing_modalities=mis,is_complete=not mis,notes="parsed actual member names"))
(O/"syncleardepth_all_samples.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
import sys;w=csv.DictWriter((O/"syncleardepth_all_samples.csv").open("w",newline=""),fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(len(rows),sum(x["is_complete"] for x in rows))
