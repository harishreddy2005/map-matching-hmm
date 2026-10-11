"""Offline plot and interactive canvas replay; no map tiles or internet required."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import Point
from .data_loader import ROOT

def render(data, prediction, start, html_name='map.html', png_name='map.png'):
    xy=data.xy[start:start+len(prediction)]
    truth=data.truth[start:start+len(prediction)]
    wrong=prediction!=truth
    true_lines=[np.array(data.lines[a].coords) for a in dict.fromkeys(truth.tolist())]
    matched_lines=[np.array(data.lines[a].coords) for a in dict.fromkeys(prediction.tolist())]
    snapped=np.array([data.lines[a].interpolate(data.lines[a].project(Point(p))).coords[0] for a,p in zip(prediction,xy)])
    fig,ax=plt.subplots(figsize=(8,5))
    ax.add_collection(LineCollection(true_lines,colors='#16a34a',linewidths=3,label='Reference route'))
    ax.add_collection(LineCollection(matched_lines,colors='#2563eb',linewidths=1.4,label='Matched arcs'))
    ax.scatter(xy[:,0],xy[:,1],s=2,c='#64748b',label='GPS')
    ax.scatter(xy[wrong,0],xy[wrong,1],s=8,c='#dc2626',label='Proxy-label mismatch')
    ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='UTM easting (m)',ylabel='UTM northing (m)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'results'/png_name,dpi=180,bbox_inches='tight');plt.close(fig)
    payload={'gps':xy.tolist(),'truth':truth.tolist(),'prediction':prediction.tolist(),
             'wrong':wrong.tolist(),'snapped':snapped.tolist(),'reference': [x.tolist() for x in true_lines],
             'matched':[x.tolist() for x in matched_lines],'start':start}
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Map matching replay</title>
<style>body{font:16px system-ui;background:#f1f5f9;color:#0f172a;margin:24px}main{max-width:1100px;margin:auto}canvas{width:100%;height:64vh;background:white;border:1px solid #cbd5e1;border-radius:12px}input{width:75%}button{padding:8px 18px}p{line-height:1.5}.green{color:#15803d}.blue{color:#2563eb}.red{color:#dc2626}</style>
<main><h1>Real-time dense map matching</h1><p>Offline UTM road view • <span class="green">Reference route</span> • <span class="blue">Matched arcs</span> • Grey GPS • <span class="red">Proxy-label mismatches</span></p>
<canvas id="map"></canvas><p><button id="play">Play</button> <input id="step" type="range" min="0" value="0"> <span id="status"></span></p>
<p>Reference labels are derived by monotone alignment to the published route. Blue lines show predicted arcs; they can be disconnected because this model does not enforce network connectivity. No external map tiles are loaded.</p></main>
<script>const D=PAYLOAD;const canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),slider=document.getElementById('step');slider.max=D.gps.length-1;
let all=D.gps.concat(...D.reference,...D.matched);let minx=Math.min(...all.map(p=>p[0])),maxx=Math.max(...all.map(p=>p[0])),miny=Math.min(...all.map(p=>p[1])),maxy=Math.max(...all.map(p=>p[1]));
function draw(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);let w=canvas.clientWidth,h=canvas.clientHeight,s=Math.min((w-40)/(maxx-minx||1),(h-40)/(maxy-miny||1));function p(a){return [(a[0]-(maxx+minx)/2)*s+w/2,h/2-(a[1]-(maxy+miny)/2)*s]};function lines(arr,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;for(let line of arr){ctx.beginPath();line.forEach((a,i)=>{let b=p(a);i?ctx.lineTo(...b):ctx.moveTo(...b)});ctx.stroke()}}lines(D.reference,'#16a34a',4);lines(D.matched,'#2563eb',1.6);D.gps.forEach((a,i)=>{let b=p(a);ctx.fillStyle=D.wrong[i]?'#dc2626':'#94a3b8';ctx.beginPath();ctx.arc(...b,D.wrong[i]?2.2:1,0,Math.PI*2);ctx.fill()});let i=+slider.value;for(let [a,c] of [[D.gps[i],'#0f172a'],[D.snapped[i],'#2563eb']]){ctx.fillStyle=c;ctx.beginPath();ctx.arc(...p(a),6,0,Math.PI*2);ctx.fill()}document.getElementById('status').textContent=`Point ${D.start+i} · ${D.wrong[i]?'mismatch':'match'}`};slider.oninput=draw;window.onresize=draw;let timer;document.getElementById('play').onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent='Play'}else{this.textContent='Pause';timer=setInterval(()=>{slider.value=(+slider.value+1)%D.gps.length;draw()},50)}};draw();</script></html>'''
    (ROOT/'results'/html_name).write_text(page.replace('PAYLOAD',json.dumps(payload,separators=(',',':'))),encoding='utf-8')
