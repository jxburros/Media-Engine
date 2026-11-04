"""Owned, deterministic Canny edges: Gaussian5, Sobel, nonmaximum suppression, hysteresis."""
from array import array
import math
from . import png


def generate(image,low,high,cancelled=None):
    from .backend import Cancelled
    w,h,rgb=png.read_rgb(image)
    def check():
        if cancelled is not None and cancelled.is_set():raise Cancelled()
    gray=array('f',((77*rgb[i]+150*rgb[i+1]+29*rgb[i+2])/256 for i in range(0,len(rgb),3)))
    kernel=(1,4,6,4,1)
    horizontal=array('f',[0])*(w*h);smooth=array('f',[0])*(w*h)
    for y in range(h):
        check()
        for x in range(w):horizontal[y*w+x]=sum(k*gray[y*w+min(w-1,max(0,x+dx-2))] for dx,k in enumerate(kernel))/16
    for y in range(h):
        check()
        for x in range(w):smooth[y*w+x]=sum(k*horizontal[min(h-1,max(0,y+dy-2))*w+x] for dy,k in enumerate(kernel))/16
    del gray,horizontal
    magnitude=array('f',[0])*(w*h);direction=bytearray(w*h)
    for y in range(1,h-1):
        check()
        for x in range(1,w-1):
            i=y*w+x
            gx=-smooth[i-w-1]+smooth[i-w+1]-2*smooth[i-1]+2*smooth[i+1]-smooth[i+w-1]+smooth[i+w+1]
            gy=-smooth[i-w-1]-2*smooth[i-w]-smooth[i-w+1]+smooth[i+w-1]+2*smooth[i+w]+smooth[i+w+1]
            magnitude[i]=math.hypot(gx,gy)
            direction[i]=int((math.degrees(math.atan2(gy,gx))%180+22.5)//45)%4
    del smooth
    edges=bytearray(w*h);stack=[];offsets=(1,w+1,w,w-1)
    for y in range(1,h-1):
        check()
        for x in range(1,w-1):
            i=y*w+x;v=magnitude[i];delta=offsets[direction[i]]
            if v>0 and v>=magnitude[i-delta] and v>=magnitude[i+delta]:
                if v>=high*1020:edges[i]=255;stack.append(i)
                elif v>=low*1020:edges[i]=1
    while stack:
        check();i=stack.pop();y,x=divmod(i,w)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                ny,nx=y+dy,x+dx
                if 0<=ny<h and 0<=nx<w and edges[ny*w+nx]==1:
                    edges[ny*w+nx]=255;stack.append(ny*w+nx)
    return png.write(w,h,bytes(v if v==255 else 0 for v in edges for _ in range(3)))
