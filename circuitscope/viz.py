"""Render a discovered circuit as a self-contained interactive HTML diagram.

Nodes are laid out by depth (input at the bottom, logits at the top, layers in
between; heads spread horizontally, the MLP to the right). Edges are drawn as
curves whose width encodes ``|EAP score|`` and whose color encodes sign (blue =
supports the behavior, red = opposes). Hovering a node shows its automated label
(DLA tokens, attention target, SAE features). The output is a single HTML string
with no external dependencies, plus a JSON serialization of the circuit.
"""

from __future__ import annotations

import json

from circuitscope.acdc import Circuit
from circuitscope.graph import NodeType


def _node_depth(name: str, n_layers: int) -> float:
    if name == "input":
        return -1.0
    if name == "logits":
        return n_layers
    # "a{L}.h{H}" or "mlp{L}"
    if name.startswith("mlp"):
        return int(name[3:]) + 0.5
    layer = int(name[1:].split(".")[0])
    return float(layer)


def _node_kind(name: str) -> str:
    if name == "input":
        return "input"
    if name == "logits":
        return "logits"
    if name.startswith("mlp"):
        return "mlp"
    return "head"


def circuit_to_dict(circuit: Circuit, labels: dict | None = None, n_layers: int = 12) -> dict:
    labels = labels or {}
    ni = getattr(circuit, "node_importance", {}) or {}
    nodes = []
    for name in sorted(circuit.nodes, key=lambda n: _node_depth(n, n_layers)):
        lab = labels.get(name)
        nodes.append({
            "id": name,
            "kind": _node_kind(name),
            "depth": _node_depth(name, n_layers),
            "importance": round(float(ni.get(name, 0.0)), 4),
            "label": (lab.summary if lab else ""),
            "promotes": (lab.promotes[:5] if lab else []),
            "attends_to": (lab.attends_to if lab else None),
            "sae_features": (lab.sae_features if lab else []),
            "method": (lab.method if lab else None),
        })
    edges = []
    for e in circuit.edges:
        edges.append({
            "src": e.src.name,
            "dst": e.dst.name,
            "qkv": e.qkv,
            "score": round(circuit.edge_scores[e.name], 5),
        })
    return {
        "behavior_metrics": {
            "clean": round(circuit.clean_metric, 4),
            "corrupt": round(circuit.corrupt_metric, 4),
            "circuit": round(circuit.metric_value, 4),
        },
        "faithfulness": round(circuit.faithfulness, 4),
        "completeness": round(circuit.completeness, 4),
        "target_faithfulness": circuit.target_faithfulness,
        "faithfulness_curve": getattr(circuit, "faithfulness_curve", []),
        "n_edges": len(circuit.edges),
        "n_edges_considered": circuit.n_edges_considered,
        "n_nodes": len(circuit.nodes),
        "nodes": nodes,
        "edges": edges,
    }


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>circuitscope — {title}</title>
<style>
  body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0d1117; color:#e6edf3; }}
  #wrap {{ display:flex; height:100vh; }}
  #stage {{ flex:1; }}
  #side {{ width:330px; padding:18px; border-left:1px solid #30363d; overflow:auto; font-size:13px; }}
  h1 {{ font-size:16px; margin:0 0 4px; }}
  .sub {{ color:#8b949e; font-size:12px; margin-bottom:14px; }}
  .stat {{ display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid #21262d; }}
  .stat b {{ color:#58a6ff; }}
  .legend span {{ display:inline-block; margin-right:10px; }}
  .swatch {{ display:inline-block; width:11px; height:11px; border-radius:2px; vertical-align:middle; margin-right:4px; }}
  #detail {{ margin-top:16px; padding:10px; background:#161b22; border-radius:8px; min-height:60px; }}
  #detail h3 {{ margin:0 0 6px; font-size:14px; color:#79c0ff; }}
  .tok {{ display:inline-block; background:#21262d; border-radius:4px; padding:1px 6px; margin:2px; font-family:monospace; font-size:12px; }}
  text {{ fill:#e6edf3; font-size:11px; pointer-events:none; }}
  .node {{ cursor:pointer; }}
  .node:hover circle, .node:hover rect {{ stroke:#fff; stroke-width:2px; }}
</style></head>
<body><div id="wrap">
<svg id="stage"></svg>
<div id="side">
  <h1>circuitscope</h1>
  <div class="sub">{title}</div>
  <div class="stat"><span>Clean metric</span><b>{clean}</b></div>
  <div class="stat"><span>Corrupt metric</span><b>{corrupt}</b></div>
  <div class="stat"><span>Circuit metric</span><b>{cmetric}</b></div>
  <div class="stat"><span>Faithfulness</span><b>{faith}</b></div>
  <div class="stat"><span>Completeness</span><b>{complete}</b></div>
  <div class="stat"><span>Edges (of {considered})</span><b>{nedges}</b></div>
  <div class="stat"><span>Nodes</span><b>{nnodes}</b></div>
  <div class="legend" style="margin-top:12px;">
    <span><span class="swatch" style="background:#58a6ff"></span>supports</span>
    <span><span class="swatch" style="background:#f85149"></span>opposes</span>
    <span><span class="swatch" style="background:#3fb950"></span>head</span>
    <span><span class="swatch" style="background:#d29922"></span>mlp</span>
  </div>
  <div id="detail">Hover a node for its automated label.</div>
</div></div>
<script>
const DATA = {data};
const svg = document.getElementById('stage');
const W = svg.clientWidth || 900, H = svg.clientHeight || 800;
const NS = 'http://www.w3.org/2000/svg';
const depths = [...new Set(DATA.nodes.map(n=>n.depth))].sort((a,b)=>a-b);
const dY = d => H - 60 - (depths.indexOf(d))*( (H-120)/Math.max(1,depths.length-1) );
// x by spreading nodes that share a depth
const byDepth = {{}};
DATA.nodes.forEach(n=>{{ (byDepth[n.depth]=byDepth[n.depth]||[]).push(n); }});
const pos = {{}};
Object.values(byDepth).forEach(arr=>{{
  arr.forEach((n,i)=>{{ pos[n.id] = {{x: 80 + (i+1)*( (W-160)/(arr.length+1) ), y: dY(n.depth)}}; }});
}});
const maxAbs = Math.max(1e-6, ...DATA.edges.map(e=>Math.abs(e.score)));
DATA.edges.forEach(e=>{{
  const a=pos[e.src], b=pos[e.dst]; if(!a||!b) return;
  const p=document.createElementNS(NS,'path');
  const mx=(a.x+b.x)/2, my=(a.y+b.y)/2-30;
  p.setAttribute('d',`M${{a.x}},${{a.y}} Q${{mx}},${{my}} ${{b.x}},${{b.y}}`);
  p.setAttribute('fill','none');
  p.setAttribute('stroke', e.score>=0 ? '#58a6ff' : '#f85149');
  p.setAttribute('stroke-width', 0.6 + 4*Math.abs(e.score)/maxAbs);
  p.setAttribute('stroke-opacity','0.5');
  svg.appendChild(p);
}});
const colors = {{head:'#3fb950', mlp:'#d29922', input:'#8b949e', logits:'#bc8cff'}};
DATA.nodes.forEach(n=>{{
  const p=pos[n.id];
  const g=document.createElementNS(NS,'g'); g.setAttribute('class','node');
  if(n.kind==='head'){{
    const c=document.createElementNS(NS,'circle');
    c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',13);
    c.setAttribute('fill',colors[n.kind]);g.appendChild(c);
  }} else {{
    const r=document.createElementNS(NS,'rect');
    r.setAttribute('x',p.x-16);r.setAttribute('y',p.y-12);
    r.setAttribute('width',32);r.setAttribute('height',24);r.setAttribute('rx',5);
    r.setAttribute('fill',colors[n.kind]);g.appendChild(r);
  }}
  const t=document.createElementNS(NS,'text');
  t.setAttribute('x',p.x);t.setAttribute('y',p.y+25);t.setAttribute('text-anchor','middle');
  t.textContent=n.id; g.appendChild(t);
  g.addEventListener('mouseenter',()=>showDetail(n));
  svg.appendChild(g);
}});
function showDetail(n){{
  let h=`<h3>${{n.id}}</h3>`;
  if(n.method) h+=`<div class="sub">label via ${{n.method}}</div>`;
  if(n.attends_to) h+=`<div>${{n.attends_to}}</div>`;
  if(n.promotes&&n.promotes.length){{ h+=`<div style="margin-top:6px">promotes:</div>`; n.promotes.forEach(t=>h+=`<span class="tok">${{t.replace(/</g,'&lt;')}}</span>`); }}
  if(n.sae_features&&n.sae_features.length){{ h+=`<div style="margin-top:8px">SAE features:</div>`; n.sae_features.forEach(f=>h+=`<span class="tok">L${{f.layer}} #${{f.feature}} (${{f.activation}})</span>`); }}
  document.getElementById('detail').innerHTML=h;
}}
</script></body></html>"""


def feature_circuit_to_dict(fc, n_layers: int = 12) -> dict:
    feats = [n for n in fc.nodes if not n.is_error]
    nodes = []
    for n in sorted(feats, key=lambda n: (n.layer, -abs(n.ie))):
        lab = fc.labels.get(n.name, {})
        nodes.append({
            "id": n.name,
            "layer": n.layer,
            "ie": round(n.ie, 4),
            "promotes": lab.get("promotes", [])[:5],
        })
    edges = [{"src": e.src, "dst": e.dst, "weight": e.weight} for e in fc.edges]
    return {
        "behavior_metrics": {
            "clean": round(fc.clean_metric, 4),
            "corrupt": round(fc.corrupt_metric, 4),
            "circuit": round(fc.metric_value, 4),
        },
        "faithfulness": round(fc.faithfulness, 4),
        "completeness": round(fc.completeness, 4),
        "errors_only_baseline": round(fc.errors_only_baseline, 4),
        "target_faithfulness": fc.target_faithfulness,
        "include_errors": fc.include_errors,
        "faithfulness_curve": fc.faithfulness_curve,
        "n_features": fc.n_features,
        "nodes": nodes,
        "edges": edges,
    }


_FEATURE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>circuitscope features — {title}</title>
<style>
  body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0d1117; color:#e6edf3; }}
  #wrap {{ display:flex; height:100vh; }}
  #stage {{ flex:1; }}
  #side {{ width:340px; padding:18px; border-left:1px solid #30363d; overflow:auto; font-size:13px; }}
  h1 {{ font-size:16px; margin:0 0 4px; }}
  .sub {{ color:#8b949e; font-size:12px; margin-bottom:14px; }}
  .stat {{ display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid #21262d; }}
  .stat b {{ color:#58a6ff; }}
  #detail {{ margin-top:16px; padding:10px; background:#161b22; border-radius:8px; min-height:60px; }}
  #detail h3 {{ margin:0 0 6px; font-size:14px; color:#79c0ff; font-family:monospace; }}
  .tok {{ display:inline-block; background:#21262d; border-radius:4px; padding:1px 6px; margin:2px; font-family:monospace; font-size:12px; }}
  text {{ fill:#e6edf3; font-size:10px; pointer-events:none; }}
  .lyr {{ fill:#8b949e; font-size:11px; }}
  .node {{ cursor:pointer; }}
  .node:hover circle {{ stroke:#fff; stroke-width:2px; }}
</style></head>
<body><div id="wrap">
<svg id="stage"></svg>
<div id="side">
  <h1>circuitscope · features</h1>
  <div class="sub">{title}</div>
  <div class="stat"><span>Clean metric</span><b>{clean}</b></div>
  <div class="stat"><span>Corrupt metric</span><b>{corrupt}</b></div>
  <div class="stat"><span>Faithfulness (feats+err)</span><b>{faith}</b></div>
  <div class="stat"><span>Errors-only baseline</span><b>{errbase}</b></div>
  <div class="stat"><span>Completeness</span><b>{complete}</b></div>
  <div class="stat"><span>SAE features in circuit</span><b>{nfeat}</b></div>
  <div class="sub" style="margin-top:12px;">Node = SAE feature, placed by layer (bottom→top).
  Size = |indirect effect|; blue supports, red opposes. Hover for promoted tokens.</div>
  <div id="detail">Hover a feature node.</div>
</div></div>
<script>
const DATA = {data};
const svg=document.getElementById('stage'), NS='http://www.w3.org/2000/svg';
const W=svg.clientWidth||900, H=svg.clientHeight||800, NL={n_layers};
const layers=[...new Set(DATA.nodes.map(n=>n.layer))].sort((a,b)=>a-b);
const yOf=l=> H-50 - (layers.indexOf(l))*((H-100)/Math.max(1,layers.length-1));
const lane={{}}; DATA.nodes.forEach(n=>{{(lane[n.layer]=lane[n.layer]||[]).push(n);}});
const pos={{}};
Object.entries(lane).forEach(([l,arr])=>{{arr.forEach((n,i)=>{{pos[n.id]={{x:90+(i+1)*((W-180)/(arr.length+1)), y:yOf(+l)}};}});}});
layers.forEach(l=>{{const t=document.createElementNS(NS,'text');t.setAttribute('class','lyr');t.setAttribute('x',12);t.setAttribute('y',yOf(l)+3);t.textContent='L'+l;svg.appendChild(t);}});
const maxW=Math.max(1e-6,...DATA.edges.map(e=>Math.abs(e.weight)));
DATA.edges.forEach(e=>{{const a=pos[e.src],b=pos[e.dst];if(!a||!b)return;const p=document.createElementNS(NS,'path');
  const mx=(a.x+b.x)/2,my=(a.y+b.y)/2; p.setAttribute('d',`M${{a.x}},${{a.y}} Q${{mx+25}},${{my}} ${{b.x}},${{b.y}}`);
  p.setAttribute('fill','none');p.setAttribute('stroke',e.weight>=0?'#58a6ff':'#f85149');
  p.setAttribute('stroke-width',(0.5+3*Math.abs(e.weight)/maxW).toFixed(2));p.setAttribute('stroke-opacity','0.4');svg.appendChild(p);}});
const maxIe=Math.max(1e-6,...DATA.nodes.map(n=>Math.abs(n.ie)));
DATA.nodes.forEach(n=>{{const p=pos[n.id];const g=document.createElementNS(NS,'g');g.setAttribute('class','node');
  const c=document.createElementNS(NS,'circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);
  c.setAttribute('r',(4+12*Math.abs(n.ie)/maxIe).toFixed(1));c.setAttribute('fill',n.ie>=0?'#3fb950':'#f85149');
  c.setAttribute('fill-opacity','0.85');g.appendChild(c);
  g.addEventListener('mouseenter',()=>{{let h=`<h3>${{n.id}}</h3><div class="sub">indirect effect ${{n.ie>=0?'+':''}}${{n.ie}}</div>`;
    if(n.promotes&&n.promotes.length){{h+='promotes:';n.promotes.forEach(t=>h+=`<span class="tok">${{(t+'').replace(/</g,'&lt;')}}</span>`);}}
    document.getElementById('detail').innerHTML=h;}});
  svg.appendChild(g);}});
</script></body></html>"""


def render_feature_html(fc, n_layers: int, title: str) -> str:
    data = feature_circuit_to_dict(fc, n_layers)
    return _FEATURE_HTML.format(
        title=title, data=json.dumps(data), n_layers=n_layers,
        clean=data["behavior_metrics"]["clean"],
        corrupt=data["behavior_metrics"]["corrupt"],
        faith=f'{data["faithfulness"]:.2%}',
        errbase=f'{data["errors_only_baseline"]:.2%}',
        complete=f'{data["completeness"]:.2%}',
        nfeat=data["n_features"],
    )


def render_html(circuit: Circuit, labels: dict | None, n_layers: int, title: str) -> str:
    data = circuit_to_dict(circuit, labels, n_layers)
    return _HTML_TEMPLATE.format(
        title=title,
        data=json.dumps(data),
        clean=data["behavior_metrics"]["clean"],
        corrupt=data["behavior_metrics"]["corrupt"],
        cmetric=data["behavior_metrics"]["circuit"],
        faith=f'{data["faithfulness"]:.2%}',
        complete=f'{data["completeness"]:.2%}',
        considered=data["n_edges_considered"],
        nedges=data["n_edges"],
        nnodes=data["n_nodes"],
    )
