"""Small grayscale vision pipeline; not a trained detector or surveillance system."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

def components(image, threshold=127, min_area=4):
    if not image or not image[0] or any(len(row)!=len(image[0]) for row in image):
        raise ValueError('expected a nonempty rectangular image')
    if not 0<=threshold<=255 or min_area<1:
        raise ValueError('invalid component parameters')
    if any(not isinstance(x,int) or not 0<=x<=255 for row in image for x in row):
        raise ValueError('pixels must be integer grayscale values 0..255')
    height,width=len(image),len(image[0]); seen=set(); boxes=[]
    for y in range(height):
        for x in range(width):
            if (x,y) in seen or image[y][x]<=threshold: continue
            stack=[(x,y)]; seen.add((x,y)); points=[]
            while stack:
                px,py=stack.pop(); points.append((px,py))
                for nx,ny in [(px-1,py),(px+1,py),(px,py-1),(px,py+1)]:
                    if 0<=nx<width and 0<=ny<height and (nx,ny) not in seen and image[ny][nx]>threshold:
                        seen.add((nx,ny)); stack.append((nx,ny))
            if len(points)>=min_area:
                xs,ys=zip(*points); boxes.append((min(xs),min(ys),max(xs)+1,max(ys)+1))
    return boxes

def iou(a,b):
    area=lambda v:max(0,v[2]-v[0])*max(0,v[3]-v[1])
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=area(a)+area(b)-intersection
    return intersection/union if union else 0.0

@dataclass
class Track:
    id:int
    box:tuple
    missed:int=0
    counted:bool=False

class Tracker:
    def __init__(self, line_x=16, min_iou=.1, max_missed=2):
        if not 0<min_iou<=1 or max_missed<0:
            raise ValueError('invalid tracker settings')
        self.line_x,self.min_iou,self.max_missed=line_x,min_iou,max_missed
        self.tracks={}; self.next_id=1; self.crossings=0

    def update(self, boxes):
        for box in boxes:
            if len(box)!=4 or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError('boxes use positive-area x1,y1,x2,y2 coordinates')
        candidates=sorted((-iou(track.box,box),ident,j) for ident,track in self.tracks.items() for j,box in enumerate(boxes))
        matched_tracks,matched_boxes=set(),set()
        for neg_score,ident,j in candidates:
            if -neg_score<self.min_iou or ident in matched_tracks or j in matched_boxes: continue
            track=self.tracks[ident]; box=tuple(boxes[j])
            old=(track.box[0]+track.box[2])/2; new=(box[0]+box[2])/2
            if old<self.line_x<=new and not track.counted:
                self.crossings+=1; track.counted=True
            track.box=box; track.missed=0
            matched_tracks.add(ident); matched_boxes.add(j)
        for ident in list(self.tracks):
            if ident not in matched_tracks:
                self.tracks[ident].missed+=1
                if self.tracks[ident].missed>self.max_missed: del self.tracks[ident]
        for j,box in enumerate(boxes):
            if j not in matched_boxes:
                self.tracks[self.next_id]=Track(self.next_id,tuple(box)); self.next_id+=1
        return [dict(id=t.id,box=t.box,missed=t.missed) for t in self.tracks.values()]

def render_frame(width,height,boxes):
    if width<=0 or height<=0 or width*height>1_000_000:
        raise ValueError('invalid image dimensions')
    image=[[0]*width for _ in range(height)]
    for x1,y1,x2,y2 in boxes:
        if not 0<=x1<x2<=width or not 0<=y1<y2<=height:
            raise ValueError('box outside image')
        for y in range(y1,y2):
            for x in range(x1,x2): image[y][x]=255
    return image

def run(sequence, output):
    output.mkdir(parents=True,exist_ok=True)
    tracker=Tracker(line_x=sequence['line_x']); results=[]
    for index,boxes in enumerate(sequence['frames']):
        image=render_frame(sequence['width'],sequence['height'],boxes)
        pgm=f"P2\n{sequence['width']} {sequence['height']}\n255\n"+'\n'.join(' '.join(map(str,row)) for row in image)
        (output/f'frame-{index:03d}.pgm').write_text(pgm+'\n')
        tracks=tracker.update(components(image))
        results.append(dict(frame=index,tracks=tracks,crossings=tracker.crossings))
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).parent/'samples/sequence.json')
    parser.add_argument('--output',type=Path,default=Path('output'))
    args=parser.parse_args()
    print(json.dumps(run(json.loads(args.input.read_text()),args.output),indent=2))

if __name__=='__main__': main()
