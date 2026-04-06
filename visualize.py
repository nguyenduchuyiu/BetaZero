import json
from pyvis.network import Network
from flask import Flask, request, render_template_string, jsonify
from expr_parser import get_lean_expr_tree

# ── Color palette ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "color_map": {
        "forallE": "#38BDF8",   # sky blue
        "lam":     "#4ADE80",   # emerald
        "letE":    "#FB7185",   # rose
        "app":     "#FACC15",   # amber
        "const":   "#A78BFA",   # violet
        "bvar":    "#FB923C",   # orange
        "lit":     "#94A3B8",   # slate
        "mdata":   "#2DD4BF",   # teal
        "proj":    "#F472B6",   # pink
        "default": "#64748B"    # muted
    },
    "edge_color":          "#334155",
    "node_size_expr":      22,
    "node_size_root":      34,
    "theorem_node_color":  "#F8FAFC",
    "theorem_node_shape":  "star",
    "theorem_node_size":   44,
    "bgcolor":             "#0D1117",
    "font_color":          "#E2E8F0",
    "font_size":           11,
    "physics_solver":      "forceAtlas2Based"
}

EDGE_RELATIONS = {
    "fn":       "fn",
    "arg":      "arg",
    "body":     "body",
    "var_type": "type",
    "val":      "val",
    "inner":    "inner",
    "type":     "type",
}


def _patch_pyvis_html_full_height(html_path: str) -> None:
    """PyVis sets #mynetwork height:100% but body/.card have no height, so the canvas stays tiny."""
    marker = "betazero-pyvis-fullviewport"
    try:
        with open(html_path, encoding="utf-8") as f:
            s = f.read()
    except OSError:
        return
    if marker in s:
        return
    inject = (
        f'<style id="{marker}">'
        "html,body{height:100%;margin:0;overflow:hidden;}"
        ".card{height:100%!important;border:none!important;margin:0!important;box-shadow:none!important;}"
        ".card-body{height:100%!important;padding:0!important;min-height:0!important;}"
        "#mynetwork{height:100%!important;width:100%!important;min-height:0!important;border:none!important;box-sizing:border-box;}"
        ".node-tooltip{font-family:JetBrains Mono,monospace;font-size:11px;max-width:420px;white-space:pre-wrap;margin:0;}"
        "</style>\n"
    )
    if "</head>" in s:
        s = s.replace("</head>", inject + "</head>", 1)
    else:
        s = inject + s
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(s)


# ── Visualizer ─────────────────────────────────────────────────────────────────
class LeanExprVisualizer:
    def __init__(self, config=None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg
        self.COLOR_MAP = cfg["color_map"]
        self._counter = 0

        self.net = Network(
            height="100%",
            width="100%",
            bgcolor=cfg["bgcolor"],
            font_color=cfg["font_color"],
            directed=True,
            select_menu=False,
            filter_menu=False,
        )
        self._setup_physics()

    # ── Physics ──────────────────────────────────────────────────────────────
    def _setup_physics(self):
        solver = self.cfg.get("physics_solver", "forceAtlas2Based")
        options = {
            "physics": {
                "enabled": True,
                "solver": solver,
                "forceAtlas2Based": {
                    "gravitationalConstant": -80,
                    "centralGravity": 0.008,
                    "springLength": 100,
                    "springConstant": 0.05,
                    "damping": 0.4,
                    "avoidOverlap": 0.8,
                },
                "stabilization": {"iterations": 200},
            },
            "edges": {
                "color": {"color": self.cfg["edge_color"], "highlight": "#94A3B8"},
                "font": {"size": 9, "color": "#64748B", "strokeWidth": 0},
                "smooth": {"type": "curvedCW", "roundness": 0.15},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                "width": 1.2,
            },
            "nodes": {
                "borderWidth": 1.5,
                "borderWidthSelected": 2.5,
                "font": {"size": self.cfg.get("font_size", 11), "face": "JetBrains Mono, monospace"},
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True,
                "keyboard": True,
                "tooltipDelay": 150,
                "multiselect": True,
            },
        }
        self.net.set_options(json.dumps(options))

    # ── Node helpers ──────────────────────────────────────────────────────────
    def _node_label(self, d):
        kind = d.get("expr", "?")
        parts = [kind]
        if kind == "const":  parts.append(d.get("name", ""))
        elif kind == "bvar": parts.append(f"#{d.get('idx', '?')}")
        elif kind == "lit":  parts.append(str(d.get("val", "")))
        elif kind in ("forallE", "lam", "letE"): parts.append(d.get("var_name", ""))
        return "\n".join(p for p in parts if p)

    def _node_color(self, kind):
        hex_color = self.COLOR_MAP.get(kind, self.COLOR_MAP["default"])
        return {
            "background": hex_color + "22",   # translucent fill
            "border":     hex_color,
            "highlight":  {"background": hex_color + "44", "border": hex_color},
            "hover":      {"background": hex_color + "33", "border": hex_color},
        }

    # ── Recursive build ───────────────────────────────────────────────────────
    def _build(self, d, is_root=False):
        if not isinstance(d, dict):
            return None
        nid = self._counter
        self._counter += 1
        kind = d.get("expr", "unknown")
        size = self.cfg["node_size_root"] if is_root else self.cfg["node_size_expr"]
        self.net.add_node(
            nid,
            label=self._node_label(d),
            title=json.dumps(d, indent=2, ensure_ascii=False),
            color=self._node_color(kind),
            shape="dot",
            size=size,
            borderWidth=2 if is_root else 1.5,
        )
        for key, edge_label in EDGE_RELATIONS.items():
            child = d.get(key)
            if isinstance(child, dict):
                cid = self._build(child)
                if cid is not None:
                    self.net.add_edge(nid, cid, label=edge_label)
        return nid

    # ── Public API ────────────────────────────────────────────────────────────
    def visualize(self, expr_json_list, filename="lean_expr_graph.html"):
        self._counter = 0
        thm_color = self.cfg["theorem_node_color"]
        for item in expr_json_list:
            name = item.get("theorem", "theorem")
            tree = item.get("expr_tree")
            if not tree:
                continue
            root_id = self._build(tree, is_root=True)
            thm_id  = f"thm_{name}"
            self.net.add_node(
                thm_id,
                label=f"⊢ {name}",
                color={"background": "#1E293B", "border": thm_color,
                       "highlight": {"background": "#334155", "border": thm_color}},
                shape=self.cfg["theorem_node_shape"],
                size=self.cfg["theorem_node_size"],
                font={"size": 13, "bold": True, "color": thm_color},
            )
            if root_id is not None:
                self.net.add_edge(thm_id, root_id,
                                  label="stmt",
                                  color={"color": thm_color + "88"},
                                  width=2)
        # write bare graph HTML (no physics buttons — we handle UI ourselves)
        self.net.save_graph(filename)
        _patch_pyvis_html_full_height(filename)
        return filename


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── HTML template (redesigned) ─────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lean Expr · Visualizer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  /* ── reset + base ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:         #0D1117;
    --surface-1:  #111827;
    --surface-2:  #1E2D3D;
    --surface-3:  #243447;
    --border:     #1E3A5F;
    --border-dim: #1A2B3C;
    --text-1:     #E2E8F0;
    --text-2:     #94A3B8;
    --text-3:     #64748B;
    --accent:     #38BDF8;
    --accent-glow:#38BDF822;
    --green:      #4ADE80;
    --violet:     #A78BFA;
    --rose:       #FB7185;
    --amber:      #FACC15;
    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  14px;
    --font-mono:  'JetBrains Mono', monospace;
    --font-ui:    'Outfit', sans-serif;
    --sidebar-w:  340px;
    --topbar-h:   52px;
    --status-h:   30px;
  }
  html, body { height: 100%; overflow: hidden; }
  body {
    background: var(--bg);
    color: var(--text-1);
    font-family: var(--font-ui);
    font-size: 14px;
    display: flex;
    flex-direction: column;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* ── Top bar ── */
  #topbar {
    height: var(--topbar-h);
    background: var(--surface-1);
    border-bottom: 1px solid var(--border-dim);
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 16px;
    flex-shrink: 0;
    z-index: 10;
  }
  .logo {
    font-family: var(--font-mono);
    font-size: 15px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .logo-icon {
    width: 26px; height: 26px;
    background: var(--accent-glow);
    border: 1px solid var(--accent);
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
  }
  .logo-sep { color: var(--text-3); font-weight: 300; }
  .logo-sub { font-family: var(--font-ui); font-size: 12px; font-weight: 400; color: var(--text-3); }
  .topbar-spacer { flex: 1; }
  .badge {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 20px;
    border: 1px solid;
    font-weight: 500;
  }
  .badge-blue { color: var(--accent); border-color: var(--accent); background: var(--accent-glow); }
  .badge-green { color: var(--green); border-color: var(--green); background: #4ADE8011; }
  .badge-dim { color: var(--text-3); border-color: var(--border); }

  /* ── Main layout ── */
  #main {
    flex: 1;
    min-height: 0;
    display: flex;
    overflow: hidden;
  }

  /* ── Sidebar ── */
  #sidebar {
    width: var(--sidebar-w);
    flex-shrink: 0;
    background: var(--surface-1);
    border-right: 1px solid var(--border-dim);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Sidebar section ── */
  .section {
    border-bottom: 1px solid var(--border-dim);
    padding: 14px 16px;
  }
  .section-head {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-3);
  }
  .section-head-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
  }

  /* ── Editor ── */
  #editor-wrap {
    position: relative;
    flex: 1;
    min-height: 0;
    padding: 0 16px 0 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .editor-head {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 12px 16px 8px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-3);
    border-bottom: 1px solid var(--border-dim);
    flex-shrink: 0;
  }
  .editor-inner {
    flex: 1;
    min-height: 0;
    display: flex;
    overflow: hidden;
  }
  #line-nums {
    width: 36px;
    flex-shrink: 0;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1.7;
    color: var(--text-3);
    text-align: right;
    padding: 12px 8px 12px 0;
    background: var(--surface-1);
    overflow: hidden;
    user-select: none;
  }
  #lean-code {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    resize: none;
    font-family: var(--font-mono);
    font-size: 12.5px;
    line-height: 1.7;
    color: var(--text-1);
    padding: 12px 16px 12px 8px;
    tab-size: 2;
    overflow-y: auto;
    caret-color: var(--accent);
    white-space: pre;
    overflow-x: auto;
  }
  #lean-code::placeholder { color: var(--text-3); }
  #lean-code:focus { outline: none; }

  /* ── Config ── */
  #config-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border-dim);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-3);
    user-select: none;
    transition: color .15s;
  }
  #config-toggle:hover { color: var(--text-2); }
  .toggle-arrow { font-size: 10px; transition: transform .2s; }
  .toggle-arrow.open { transform: rotate(180deg); }
  #config-panel {
    display: none;
    padding: 10px 16px 12px;
    border-bottom: 1px solid var(--border-dim);
  }
  #config-panel.open { display: block; }
  #config-json {
    width: 100%;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-2);
    font-family: var(--font-mono);
    font-size: 10.5px;
    line-height: 1.6;
    padding: 8px 10px;
    resize: vertical;
    min-height: 80px;
    max-height: 160px;
    outline: none;
  }
  #config-json:focus { border-color: var(--accent); color: var(--text-1); }

  /* ── Legend ── */
  .legend-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-2);
  }
  .legend-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* ── Run button ── */
  #btn-run {
    margin: 14px 16px;
    background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
    border: none;
    border-radius: var(--radius-md);
    color: #fff;
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
    padding: 10px 0;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: opacity .15s, transform .12s;
    flex-shrink: 0;
  }
  #btn-run:hover { opacity: 0.88; }
  #btn-run:active { transform: scale(0.98); }
  #btn-run:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-spinner {
    width: 14px; height: 14px;
    border: 2px solid rgba(255,255,255,.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin .7s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Graph pane ── */
  #graph-pane {
    flex: 1;
    min-width: 0;
    min-height: 0;
    position: relative;
    display: flex;
    flex-direction: column;
  }
  #graph-frame {
    flex: 1;
    min-height: 0;
    width: 100%;
    height: 100%;
    border: none;
    background: #0D1117;
    display: block;
  }

  /* ── Empty state ── */
  #empty-state {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    pointer-events: none;
  }
  .empty-icon {
    font-size: 48px;
    opacity: 0.12;
  }
  .empty-title {
    font-size: 15px;
    font-weight: 500;
    color: var(--text-3);
  }
  .empty-sub {
    font-size: 12px;
    color: var(--text-3);
    opacity: 0.7;
  }

  /* ── Status bar ── */
  #status-bar {
    height: var(--status-h);
    background: var(--surface-1);
    border-top: 1px solid var(--border-dim);
    display: flex;
    align-items: center;
    padding: 0 14px;
    gap: 18px;
    flex-shrink: 0;
  }
  .status-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-3);
  }
  .status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--text-3);
  }
  .status-dot.ready { background: var(--green); }
  .status-dot.loading { background: var(--amber); animation: pulse 1s ease-in-out infinite; }
  .status-dot.error { background: var(--rose); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── Error banner ── */
  #error-bar {
    background: #1A0F0F;
    border-top: 1px solid #5B2121;
    color: var(--rose);
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 8px 16px;
    display: none;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 80px;
    overflow-y: auto;
  }
  #error-bar.show { display: block; }

  /* ── Resizer ── */
  #resizer {
    width: 4px;
    cursor: col-resize;
    background: var(--border-dim);
    flex-shrink: 0;
    transition: background .15s;
  }
  #resizer:hover, #resizer.dragging { background: var(--accent); }

  /* ── Keyboard hint ── */
  kbd {
    font-family: var(--font-mono);
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    border: 1px solid var(--border);
    color: var(--text-3);
    background: var(--surface-2);
  }
</style>
</head>
<body>

<!-- Top bar -->
<header id="topbar">
  <div class="logo">
    <div class="logo-icon">λ</div>
    <span>lean</span>
    <span class="logo-sep">/</span>
    <span style="color:var(--text-2)">expr</span>
  </div>
  <span class="logo-sub">AST Visualizer</span>
  <div class="topbar-spacer"></div>
  <span class="badge badge-blue">pyvis</span>
  <span class="badge badge-green">flask</span>
  <span class="badge badge-dim">v2.0</span>
</header>

<!-- Main -->
<div id="main">

  <!-- Sidebar -->
  <aside id="sidebar">

    <!-- Editor -->
    <div class="editor-head">
      <div class="section-head-dot"></div>
      theorem input
      <span style="margin-left:auto"><kbd>Ctrl</kbd>+<kbd>Enter</kbd></span>
    </div>
    <div id="editor-wrap">
      <div class="editor-inner">
        <div id="line-nums">1</div>
        <textarea
          id="lean-code"
          spellcheck="false"
          autocomplete="off"
          autocorrect="off"
          placeholder="-- Paste your Lean theorem or expression here
theorem my_lemma (n : Nat) : n + 0 = n := by
  simp"
        >{{ lean_code }}</textarea>
      </div>
    </div>

    <!-- Config toggle -->
    <div id="config-toggle" onclick="toggleConfig()">
      <span style="display:flex;align-items:center;gap:6px">
        <span class="section-head-dot" style="background:var(--violet)"></span>
        config (JSON)
      </span>
      <span class="toggle-arrow" id="config-arrow">▼</span>
    </div>
    <div id="config-panel">
      <textarea id="config-json" rows="6">{{ config }}</textarea>
    </div>

    <!-- Legend -->
    <div class="section">
      <div class="section-head">
        <div class="section-head-dot" style="background:var(--amber)"></div>
        node types
      </div>
      <div class="legend-grid" id="legend"></div>
    </div>

    <!-- Run button -->
    <button id="btn-run" onclick="runVisualize()">
      <div class="btn-spinner" id="spinner"></div>
      <span id="btn-label">▶ Parse &amp; Visualize</span>
    </button>

  </aside>

  <!-- Resizer -->
  <div id="resizer"></div>

  <!-- Graph pane -->
  <div id="graph-pane">
    {% if graph_url %}
    <iframe id="graph-frame" src="{{ graph_url }}" title="Lean expression graph"></iframe>
    {% else %}
    <div id="empty-state">
      <div class="empty-icon">λ</div>
      <div class="empty-title">No graph yet</div>
      <div class="empty-sub">Paste a Lean theorem and click Visualize</div>
    </div>
    {% endif %}
  </div>
</div>

<!-- Status + error -->
<div id="error-bar"></div>
<footer id="status-bar">
  <div class="status-item">
    <div class="status-dot ready" id="status-dot"></div>
    <span id="status-text">ready</span>
  </div>
  <div class="status-item" id="status-nodes" style="display:none">
    nodes: <span id="node-count" style="color:var(--accent)">—</span>
  </div>
  <div class="status-item" id="status-edges" style="display:none">
    edges: <span id="edge-count" style="color:var(--violet)">—</span>
  </div>
  <div style="margin-left:auto" class="status-item">
    lean expr · ast visualizer
  </div>
</footer>

<script>
/* ── Legend colors ── */
const COLOR_MAP = {{ color_map_json | safe }};
const legend = document.getElementById('legend');
Object.entries(COLOR_MAP).filter(([k]) => k !== 'default').forEach(([kind, hex]) => {
  legend.innerHTML += `<div class="legend-item">
    <div class="legend-dot" style="background:${hex}"></div>${kind}
  </div>`;
});

/* ── Line numbers ── */
const ta = document.getElementById('lean-code');
const ln = document.getElementById('line-nums');
function updateLineNums() {
  const lines = ta.value.split('\n').length;
  ln.innerHTML = Array.from({length: lines}, (_, i) => i + 1).join('<br>');
  ln.scrollTop = ta.scrollTop;
}
ta.addEventListener('input', updateLineNums);
ta.addEventListener('scroll', () => { ln.scrollTop = ta.scrollTop; });
updateLineNums();

/* ── Config toggle ── */
function toggleConfig() {
  const panel = document.getElementById('config-panel');
  const arrow = document.getElementById('config-arrow');
  panel.classList.toggle('open');
  arrow.classList.toggle('open');
}

/* ── Visualize ── */
async function runVisualize() {
  const code = ta.value.trim();
  if (!code) return;

  setStatus('loading', 'parsing…');
  document.getElementById('btn-run').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('btn-label').textContent = 'Processing…';
  document.getElementById('error-bar').classList.remove('show');

  const form = new FormData();
  form.append('lean_code', code);
  form.append('config', document.getElementById('config-json').value);

  try {
    const res = await fetch('/visualize', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) {
      setStatus('error', 'parse error');
      showError(data.error);
    } else {
      document.getElementById('empty-state')?.remove();
      const frame = document.getElementById('graph-frame') || (() => {
        const f = document.createElement('iframe');
        f.id = 'graph-frame';
        f.title = 'Lean expression graph';
        f.style.cssText = 'flex:1;min-height:0;border:none;background:#0D1117;display:block;width:100%;height:100%';
        document.getElementById('graph-pane').appendChild(f);
        return f;
      })();
      frame.src = '/graph?t=' + Date.now();
      setStatus('ready', `rendered · ${data.theorem || ''}`);
      if (data.nodes !== undefined) {
        document.getElementById('status-nodes').style.display = 'flex';
        document.getElementById('status-edges').style.display = 'flex';
        document.getElementById('node-count').textContent = data.nodes;
        document.getElementById('edge-count').textContent = data.edges;
      }
    }
  } catch (e) {
    setStatus('error', 'network error');
    showError(String(e));
  } finally {
    document.getElementById('btn-run').disabled = false;
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('btn-label').textContent = '▶ Parse & Visualize';
  }
}

function setStatus(state, msg) {
  const dot = document.getElementById('status-dot');
  dot.className = 'status-dot ' + state;
  document.getElementById('status-text').textContent = msg;
}

function showError(msg) {
  const bar = document.getElementById('error-bar');
  bar.textContent = '⚠ ' + msg;
  bar.classList.add('show');
}

/* ── Keyboard shortcut ── */
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    runVisualize();
  }
});

/* ── Resizer drag ── */
(function() {
  const resizer = document.getElementById('resizer');
  const sidebar = document.getElementById('sidebar');
  let dragging = false, startX = 0, startW = 0;
  resizer.addEventListener('mousedown', e => {
    dragging = true;
    startX = e.clientX;
    startW = sidebar.offsetWidth;
    resizer.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const w = Math.max(240, Math.min(520, startW + e.clientX - startX));
    sidebar.style.width = w + 'px';
  });
  document.addEventListener('mouseup', () => {
    dragging = false;
    resizer.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
})();
</script>
</body>
</html>"""

# ── Routes ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    default_cfg = {k: v for k, v in DEFAULT_CONFIG.items() if k != "color_map"}
    return render_template_string(
        HTML,
        lean_code="",
        config=json.dumps(default_cfg, indent=2),
        graph_url=None,
        color_map_json=json.dumps(DEFAULT_CONFIG["color_map"]),
    )

@app.route("/visualize", methods=["POST"])
def visualize():
    lean_code = request.form.get("lean_code", "").strip()
    config_raw = request.form.get("config", "")

    # Build effective config
    cfg = {**DEFAULT_CONFIG}
    try:
        if config_raw.strip():
            user_cfg = json.loads(config_raw)
            for k, v in user_cfg.items():
                if k in cfg:
                    cfg[k] = v
    except Exception:
        pass

    if not lean_code:
        return jsonify({"error": "No Lean code provided."})

    try:
        expr_json = get_lean_expr_tree(lean_code)
        if isinstance(expr_json, dict):
            expr_json = [expr_json]

        viz = LeanExprVisualizer(cfg)
        viz.visualize(expr_json, "lean_expr_graph.html")

        # Count nodes/edges
        n_nodes = len(viz.net.nodes)
        n_edges = len(viz.net.edges)
        theorem  = expr_json[0].get("theorem", "") if expr_json else ""
        return jsonify({"ok": True, "nodes": n_nodes, "edges": n_edges, "theorem": theorem})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/graph")
def serve_graph():
    try:
        with open("lean_expr_graph.html", encoding="utf-8") as f:
            html = f.read()
        # Inject dark background override
        html = html.replace("<body>",
            '<body style="background:#0D1117;margin:0;overflow:hidden;">', 1)
        return html
    except FileNotFoundError:
        return "<p style='color:#94A3B8;font-family:monospace;padding:2rem'>No graph generated yet.</p>", 404

if __name__ == "__main__":
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Visualizer: http://localhost:5678      │")
    print("  └─────────────────────────────────────────┘\n")
    app.run(host="0.0.0.0", port=5678, debug=True)