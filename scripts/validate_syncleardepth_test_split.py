#!/usr/bin/env python3
import json,pathlib,csv
R=pathlib.Path('/home/liuke/xuxin/SynClearLingBot-Depth').resolve();o=R/'data/syncleardepth/test_split'; ids=(o/'test_ids.txt').read_text().splitlines() if (o/'test_ids.txt').exists() else []; rows=[{'sample_id':x,'status':'not_downloaded'} for x in ids]
(o/'download_validation.json').write_text(json.dumps(rows,indent=2)+'\n');
with (o/'download_validation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['sample_id','status']);w.writeheader();w.writerows(rows)
(o/'failed_ids.txt').write_text('');(o/'incomplete_ids.txt').write_text('\n'.join(ids)+'\n' if ids else '');print(len(rows))
