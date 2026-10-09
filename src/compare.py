"""Collect experiment summaries; compare only matching evaluation protocols."""
import argparse
import csv
import json
from pathlib import Path
from .common import run_cli


def main():
    p=argparse.ArgumentParser()
    p.add_argument('summaries',nargs='+')
    p.add_argument('--output',default='runs/model_comparison.csv')
    args=p.parse_args()
    rows=[]
    protocols=set()
    for path in args.summaries:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        protocols.add((data['split_label'],data['image_count'],data['pair_list_sha256'],data['protocol'],data['tile_size'],data['overlap']))
        rows.append({'summary':path,**data['means']})
    if len(protocols)!=1:
        raise ValueError('Different evaluation pair lists/protocols/counts; rerun on the same split, tile size, and overlap')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved comparison; verify all models used the same saved split:',args.output)


if __name__=='__main__':
    run_cli(main)
