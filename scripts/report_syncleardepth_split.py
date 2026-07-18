#!/usr/bin/env python3
import csv,json,pathlib,collections
R=pathlib.Path('/home/liuke/xuxin/depth_io_inspection').resolve();o=R/'data/syncleardepth/test_split';allx=[json.loads(x) for x in (R/'manifests/syncleardepth_all_samples.jsonl').read_text().splitlines() if x.strip()];test=[json.loads(x) for x in (o/'test_samples.jsonl').read_text().splitlines() if x.strip()]
rows=[]
for field in ['scene','sequence','object_count']:
 a=collections.Counter(str(x.get(field,'unknown')) for x in allx);b=collections.Counter(str(x.get(field,'unknown')) for x in test)
 for k in sorted(set(a)|set(b)):rows.append([field,k,a[k],a[k]/len(allx) if allx else 0,b[k],b[k]/len(test) if test else 0,abs(a[k]/len(allx)-b[k]/len(test)) if allx and test else 0])
with (o/'distribution_comparison.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['field','category','all_count','all_ratio','test_count','test_ratio','absolute_ratio_difference']);w.writerows(rows)
(o/'SPLIT_REPORT.md').write_text('# SynClearDepth fixed test split\n\nGenerated from the full archive listing before raw test member download. See `split_config.json`, `split_statistics.json`, and `distribution_comparison.csv`.\n')
print(len(rows))
