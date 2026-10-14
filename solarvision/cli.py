import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from .data import prepare
from .pipeline import run
from .features import batch
from .model import predict
from .review import ReviewQueue

def main():
    p = argparse.ArgumentParser(description='Solar-cell research inspection desk')
    s = p.add_subparsers(dest='command', required=True)
    d = s.add_parser('download'); d.add_argument('--data',default='data'); d.add_argument('--archive')
    r = s.add_parser('run'); r.add_argument('--data',default='data'); r.add_argument('--out',default='reports/run')
    i = s.add_parser('predict'); i.add_argument('image'); i.add_argument('--model',default='reports/benchmark/model.json')
    q = s.add_parser('review'); q.add_argument('--db',required=True); q.add_argument('--id',required=True)
    q.add_argument('--decision',choices=['defect','clear','uncertain'],required=True); q.add_argument('--note',required=True)
    args = p.parse_args()
    if args.command == 'download':
        m = prepare(args.data,args.archive); print(json.dumps({k:v for k,v in m.items() if k!='images'},indent=2))
    elif args.command == 'run':
        m = run(args.data,args.out); print(json.dumps({'model':m['model'],'test':m['test'],'counts':m['counts']},indent=2))
    elif args.command == 'predict':
        m = json.loads(Path(args.model).read_text())
        with Image.open(args.image) as im:
            if im.mode != 'L' or min(im.size) < 16:
                raise ValueError('Use a grayscale electroluminescence cell crop, not a color photograph')
            score = float(predict(m,batch([np.asarray(im)]))[0])
        print(json.dumps({'score':score,'flagged':score>=m['threshold'],'threshold':m['threshold'],
                          'note':'Research screening score; human inspection required'},indent=2))
    else:
        desk = ReviewQueue(args.db)
        try: desk.review(args.id,args.decision,args.note)
        finally: desk.close()
        print('Review saved. Existing HTML/CSV benchmark snapshots are unchanged.')

if __name__ == '__main__': main()
