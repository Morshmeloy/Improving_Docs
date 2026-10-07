"""Explicit black tracing and short endpoint connections. This adds drawn pixels."""
import cv2
import numpy as np
from cv_pipeline import paper_background
from trace_pipeline import trace_strokes


def skeletonize(mask):
    """Zhang-Suen thinning, preserves one-pixel skeletons and endpoint topology."""
    a=mask.astype(bool).copy()
    for _ in range(100):
        changed=False
        for step in (0,1):
            p=np.pad(a,1)
            n=[p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],
               p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2]]
            count=sum(x.astype(np.uint8) for x in n)
            transitions=sum((~n[i]&n[(i+1)%8]).astype(np.uint8) for i in range(8))
            if step==0:triples=(n[0]&n[2]&n[4])|(n[2]&n[4]&n[6])
            else:triples=(n[0]&n[2]&n[6])|(n[0]&n[4]&n[6])
            remove=a&(count>=2)&(count<=6)&(transitions==1)&~triples
            if remove.any():a[remove]=False;changed=True
        if not changed:break
    return a


def connect_endpoints(skeleton,max_gap):
    """Pair nearby endpoints whose outgoing tangents face each other."""
    a=skeleton.astype(np.uint8)
    counts=cv2.filter2D(a,cv2.CV_16S,np.ones((3,3),np.int16),borderType=cv2.BORDER_CONSTANT)-a
    endpoints=np.argwhere((a==1)&(counts==1))
    directions=[]
    for y,x in endpoints:
        curr=(int(y),int(x));visited={curr};last=curr
        for _ in range(6):
            nexts=[(yy,xx) for yy in range(max(0,curr[0]-1),min(a.shape[0],curr[0]+2))
                   for xx in range(max(0,curr[1]-1),min(a.shape[1],curr[1]+2))
                   if a[yy,xx] and (yy,xx) not in visited]
            if len(nexts)!=1:break
            last=nexts[0];visited.add(last);curr=last
        vector=np.array([y-last[0],x-last[1]],dtype=np.float32)
        norm=np.linalg.norm(vector)
        directions.append(vector/norm if norm>0 else np.zeros(2))
    buckets={};size=max(1,max_gap)
    for i,(y,x) in enumerate(endpoints):buckets.setdefault((int(y)//size,int(x)//size),[]).append(i)
    candidates=[]
    for i,(y,x) in enumerate(endpoints):
        by,bx=int(y)//size,int(x)//size
        for gy in range(by-1,by+2):
            for gx in range(bx-1,bx+2):
                for j in buckets.get((gy,gx),[]):
                    if j<=i:continue
                    vector=(endpoints[j]-endpoints[i]).astype(np.float32)
                    dist=float(np.linalg.norm(vector))
                    if not 1.5<dist<=max_gap:continue
                    direction=vector/dist
                    if np.dot(directions[i],direction)>=.5 and np.dot(directions[j],-direction)>=.5:
                        candidates.append((dist,i,j))
    bridges=np.zeros_like(a);used=set();connections=[]
    for dist,i,j in sorted(candidates):
        if i in used or j in used:continue
        y,x=endpoints[i];yy,xx=endpoints[j]
        cv2.line(bridges,(int(x),int(y)),(int(xx),int(yy)),255,1)
        used.update((i,j));connections.append({'from':[int(x),int(y)],'to':[int(xx),int(yy)],'distance':dist})
    return bridges>0,connections


def redraw(rgb,areas,max_gap=6,line_width=1,threshold=.025,locked=None):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected RGB uint8')
    if not 0<=max_gap<=20 or not 1<=line_width<=3 or not .005<=threshold<=.2:
        raise ValueError('gap 0..20, width 1..3, threshold .005..0.2')
    if areas.shape!=rgb.shape[:2]:raise ValueError('Area mask shape mismatch')
    h,w=areas.shape
    out,_=trace_strokes(rgb,.55)
    density=1-np.clip(rgb.astype(np.float32)/np.maximum(paper_background(rgb),32),0,1)
    evidence=density.max(axis=2)
    # Filter only the drawing layer. Original observed pixels are not removed.
    weak=(evidence>=threshold)&areas
    count,labels,stats,_=cv2.connectedComponentsWithStats(weak.astype(np.uint8),8)
    keep=np.zeros(count,bool)
    if count>1:keep[1:]=(stats[1:,cv2.CC_STAT_AREA]>=7)
    weak=keep[labels]
    skeleton=skeletonize(weak)
    bridges,connections=connect_endpoints(skeleton,max_gap) if max_gap else (np.zeros_like(areas),[])
    strokes=skeleton|bridges
    if line_width>1:
        strokes=cv2.dilate(strokes.astype(np.uint8),np.ones((line_width,line_width),np.uint8))>0
    strokes&=areas
    locked=np.zeros_like(areas) if locked is None else locked
    strokes&=~locked;bridges&=areas&~locked
    out[strokes]=0
    out[locked]=rgb[locked]
    # Red marks generated pixels; green marks traced observed strokes.
    overlay=rgb.copy();overlay[strokes]=[0,170,0];overlay[bridges]=[255,0,0]
    return out,overlay,{'drawn_pixels':int(strokes.sum()),'bridge_pixels':int(bridges.sum()),
                        'candidate_connections':connections,'max_gap_pixels':max_gap,
                        'line_width':line_width,'threshold':threshold,
                        'notice':'Black drawing layer and inferred short connections; pixels are edited.'}
