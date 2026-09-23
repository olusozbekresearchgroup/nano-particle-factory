from __future__ import annotations

import argparse
import html
import json
import os
import threading
import webbrowser
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..common import DEFAULT_A, DEFAULT_C
from ..common import cluster_output_dir
from ..input import build_from_spec


ROOT = Path.cwd().resolve()
DEFAULT_INPUT = ROOT / "np_gen.in"
NM_TO_ANGSTROM = 10.0


@dataclass
class GUIState:
    input_file_name: str = "np_gen.in"
    element: str = "Ru"
    shape: str = "sphere"
    crystal_structure: str = "hexagonal"
    size_mode: str = "diameter"
    atoms: int | None = None
    radius: float | None = None
    diameter: float | None = 20.0
    a: float = DEFAULT_A
    b: float | None = None
    c: float = DEFAULT_C
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    surfaces: str = "0001=0.162,10-10=0.181,10-11=0.180"
    facet_mode: str = "top"
    extra_facets: str = "none"
    sphericity: float = 0.0
    radius_tol: float = 0.3
    max_refine: int = 12
    radius_policy: str = "closest"
    radial_trim: float | None = None
    ico_shells: int | None = None
    cube_layers: int | None = None
    octa_length: int | None = None
    octa_cutoff: int = 0
    p: int | None = None
    q: int | None = None
    r: int | None = None
    morph: float = 0.0
    cuboct_layers: int | None = None
    verbose: bool = True
    working_path: str = str(ROOT)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NPF GUI</title>
  <style>
    :root {
      --bg: #11131a;
      --panel: #171b24;
      --panel-2: #1d2330;
      --text: #e7ebf3;
      --muted: #9aa6bb;
      --accent: #6fe3c0;
      --accent-2: #70a6ff;
      --border: #2b3242;
      --warn: #ffcf7a;
      --shadow: 0 20px 60px rgba(0,0,0,.35);
      font-synthesis-weight: none;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, sans-serif;
      background: radial-gradient(circle at top, #1f2634 0%, var(--bg) 60%);
      color: var(--text);
    }
    header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      background: rgba(11, 15, 22, 0.75);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    header h1 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    header p { margin: 6px 0 0; color: var(--muted); }
    main {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 72px);
    }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .card h2 {
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      font-size: 14px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .panel {
      padding: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }
    .field label { font-size: 12px; color: var(--muted); }
    .field input, .field select, .field textarea {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      outline: none;
    }
    .field textarea { min-height: 78px; resize: vertical; }
    .details-section {
      margin: 6px 0 0;
      border-top: 1px solid var(--border);
      padding-top: 10px;
    }
    .details-section > summary {
      list-style: none;
      cursor: pointer;
      user-select: none;
      padding: 10px 0 12px;
      margin: 0;
      color: var(--text);
      font-size: 14px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .details-section > summary::-webkit-details-marker { display: none; }
    .details-section > summary::after {
      content: "Expand";
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0;
      text-transform: none;
      font-weight: 500;
    }
    .details-section[open] > summary::after {
      content: "Collapse";
    }
    .field-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
    }
    .field-toggle input {
      margin: 0;
    }
    .row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    button {
      padding: 11px 14px;
      border-radius: 999px;
      border: 1px solid transparent;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #081018;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: transparent;
      border-color: var(--border);
      color: var(--text);
    }
    .status {
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-size: 13px;
      min-height: 42px;
    }
    .viewer-wrap {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 0;
    }
    #viewer,
    #previewFrame {
      width: 100%;
      height: 100%;
      min-height: 760px;
      display: block;
      background: radial-gradient(circle at center, #0e1320 0%, #080b12 70%);
      cursor: grab;
      border: 0;
    }
    .legend {
      padding: 12px 16px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 13px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #d8e2f1;
      font-size: 12px;
      background: rgba(0,0,0,0.18);
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      max-height: 220px;
      overflow: auto;
    }
    @media (max-width: 1000px) {
      main { grid-template-columns: 1fr; }
      #viewer, #previewFrame { min-height: 520px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>NPF GUI</h1>
    <p>Front end for preparing <code>np_gen.in</code>, generating the cluster, and previewing it in the browser.</p>
  </header>
  <main>
    <section class="card">
      <h2>Settings</h2>
      <form class="panel" id="generatorForm" action="/api/generate-page" method="post" target="previewFrame">
        <div class="field">
          <label for="element">Element</label>
          <input id="element" name="element" value="Ru" />
        </div>
        <div class="field">
          <label for="crystal_structure">Crystal structure</label>
          <select id="crystal_structure" name="crystal_structure" onchange="window.__npfHandleStructureChange && window.__npfHandleStructureChange()">
            <option value="cubic">Cubic</option>
            <option value="hexagonal">Hexagonal</option>
            <option value="tetragonal">Tetragonal</option>
            <option value="orthorhombic">Orthorhombic</option>
            <option value="rhombohedral">Rhombohedral / Trigonal</option>
            <option value="monoclinic">Monoclinic</option>
            <option value="triclinic">Triclinic</option>
          </select>
          <div id="structureNote" style="margin-top:6px;color:var(--muted);font-size:12px;line-height:1.35;"></div>
          <div id="crystalDefinition" style="margin-top:6px;color:var(--muted);font-size:12px;line-height:1.35;"></div>
        </div>
        <div class="field">
          <label for="crystal_variant">Structure subtype</label>
          <select id="crystal_variant" name="crystal_variant">
            <option value="" selected disabled>Select a structure first</option>
          </select>
          <div id="variantNote" style="margin-top:6px;color:var(--muted);font-size:12px;line-height:1.35;"></div>
        </div>
        <div class="grid">
          <div class="field lattice-field lattice-a">
            <label for="a">Lattice a</label>
            <input id="a" name="a" type="number" step="0.001" value="2.706" />
          </div>
          <div class="field lattice-field lattice-b">
            <label for="b">Lattice b</label>
            <input id="b" name="b" type="number" step="0.001" placeholder="Optional" />
          </div>
          <div class="field lattice-field lattice-c">
            <label for="c">Lattice c</label>
            <input id="c" name="c" type="number" step="0.001" value="4.282" />
          </div>
        </div>
        <div class="grid">
          <div class="field lattice-field lattice-alpha">
            <label for="alpha">alpha</label>
            <input id="alpha" name="alpha" type="number" step="0.01" placeholder="Degrees" />
          </div>
          <div class="field lattice-field lattice-beta">
            <label for="beta">beta</label>
            <input id="beta" name="beta" type="number" step="0.01" placeholder="Degrees" />
          </div>
          <div class="field lattice-field lattice-gamma">
            <label for="gamma">gamma</label>
            <input id="gamma" name="gamma" type="number" step="0.01" placeholder="Degrees" />
          </div>
        </div>
        <div class="grid">
          <div class="field">
            <label for="shape">NP shape / geometry</label>
            <select id="shape" name="shape">
              <option value="wulff">Wulff</option>
              <option value="ico">Icosahedron</option>
              <option value="sphere" selected>Sphere</option>
              <option value="cube">Cube</option>
              <option value="octahedron">Octahedron</option>
              <option value="decahedron">Decahedron</option>
              <option value="dodecahedron">Dodecahedron</option>
              <option value="morphed_spherical">Morphed spherical</option>
              <option value="cuboct">Cuboctahedron</option>
              <option value="hexagonal_prism">Hexagonal prism</option>
              <option value="truncated_hexagonal_prism">Truncated hexagonal prism</option>
              <option value="hexagonal_bipyramid">Hexagonal bipyramid</option>
              <option value="truncated_hexagonal_bipyramid">Truncated hexagonal bipyramid</option>
              <option value="nanorod">Nanorod / elongated hexagonal prism</option>
              <option value="hexagonal_platelet">Hexagonal platelet / nanodisc</option>
            </select>
            <div id="shapeNote" style="margin-top:6px;color:var(--muted);font-size:12px;line-height:1.35;"></div>
          </div>
        </div>
        <div class="grid">
          <div class="field">
            <label for="size_mode">Size mode</label>
            <select id="size_mode" name="size_mode">
              <option value="atoms">Number of atoms</option>
              <option value="diameter">Diameter (nm)</option>
            </select>
          </div>
          <div class="field">
            <label id="size_value_label" for="size_value">Diameter (nm)</label>
            <input id="size_value" type="number" value="2" step="0.001" />
            <input id="atoms" name="atoms" type="hidden" value="" />
            <input id="diameter" name="diameter" type="hidden" value="20" />
          </div>
        </div>
        <div class="field">
          <label for="working_path">Working path</label>
          <input id="working_path" value="{{WORKING_PATH}}" readonly />
        </div>
        <div class="field">
          <label for="input_file_name">Input file name</label>
          <input id="input_file_name" name="input_file_name" value="np_gen.in" />
        </div>
        <div class="field">
          <label for="output_file_name">Output file name</label>
          <input id="output_file_name" name="output_file_name" value="" placeholder="Auto-generated if blank" />
        </div>
        <details class="details-section">
          <summary>Geometry Details</summary>
          <div style="padding-top: 6px;">
            <div class="field shape-field wulff-field">
              <label for="surfaces">Surfaces</label>
              <textarea id="surfaces" name="surfaces">0001=0.162,10-10=0.181,10-11=0.180</textarea>
            </div>
            <div class="grid">
              <div class="field shape-field wulff-field">
                <label for="facet_mode">Facet mode</label>
                <select id="facet_mode" name="facet_mode">
                  <option>top</option>
                  <option>minimal</option>
                  <option>full</option>
                </select>
              </div>
              <div class="field shape-field wulff-field">
                <label for="extra_facets">Extra facets</label>
                <select id="extra_facets" name="extra_facets">
                  <option>none</option>
                  <option>auto</option>
                </select>
              </div>
            </div>
            <div class="grid">
              <div class="field shape-field wulff-field">
                <label for="sphericity">Sphericity</label>
                <input id="sphericity" name="sphericity" type="number" step="0.01" value="0.0" />
              </div>
              <div class="field shape-field wulff-field">
                <label for="radius_tol">Radius tol</label>
                <input id="radius_tol" name="radius_tol" type="number" step="0.01" value="0.3" />
              </div>
            </div>
            <div class="grid">
              <div class="field shape-field wulff-field">
                <label for="max_refine">Max refine</label>
                <input id="max_refine" name="max_refine" type="number" value="12" />
              </div>
              <div class="field shape-field wulff-field">
                <label for="radial_trim">Radial trim</label>
                <input id="radial_trim" name="radial_trim" type="number" step="0.01" placeholder="Optional" />
              </div>
            </div>
            <div class="grid">
              <div class="field shape-field wulff-field">
                <label for="radius_policy">Radius policy</label>
                <select id="radius_policy" name="radius_policy">
                  <option>closest</option>
                  <option>at_least</option>
                  <option>at_most</option>
                </select>
              </div>
              <div class="field shape-field wulff-field">
                <label for="verbose">Verbose</label>
                <select id="verbose" name="verbose">
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </div>
            </div>
            <div class="grid">
              <div class="field shape-field ico-field">
                <label for="ico_shells">Icosahedron shells</label>
                <input id="ico_shells" name="ico_shells" type="number" placeholder="Optional" />
              </div>
              <div class="field shape-field cube-field">
                <label for="cube_layers">Cube layers</label>
                <input id="cube_layers" name="cube_layers" type="number" placeholder="Optional" />
              </div>
            </div>
            <div class="grid">
              <div class="field shape-field octahedron-field">
                <label for="octa_length">Octa length</label>
                <input id="octa_length" name="octa_length" type="number" placeholder="Optional" />
              </div>
              <div class="field shape-field octahedron-field">
                <label for="octa_cutoff">Octa cutoff</label>
                <input id="octa_cutoff" name="octa_cutoff" type="number" value="0" />
              </div>
            </div>
            <div class="grid">
              <div class="field shape-field decahedron-field">
                <label for="p">Deca p</label>
                <input id="p" name="p" type="number" placeholder="Optional" />
              </div>
              <div class="field shape-field decahedron-field">
                <label for="q">Deca q</label>
                <input id="q" name="q" type="number" placeholder="Optional" />
              </div>
            </div>
            <div class="grid">
              <div class="field shape-field decahedron-field">
                <label for="r">Deca r</label>
                <input id="r" name="r" type="number" placeholder="Optional" />
              </div>
              <div class="field shape-field cuboct-field">
                <label for="cuboct_layers">Cuboct layers</label>
                <input id="cuboct_layers" name="cuboct_layers" type="number" placeholder="Optional" />
              </div>
            </div>
            <div class="grid">
              <div class="field shape-field morphed_spherical-field">
                <label for="morph">Morph</label>
                <input id="morph" name="morph" type="number" step="0.01" value="0.0" />
              </div>
            </div>
          </div>
        </details>
        <div class="row">
          <button type="submit" name="mode" value="generate">Generate</button>
          <button type="submit" name="mode" value="write_only" class="secondary">Write input only</button>
          <label class="secondary" style="display:inline-flex;align-items:center;gap:8px;padding:11px 14px;border-radius:999px;border:1px solid var(--border);cursor:pointer;">
            Load file
            <input id="fileInput" type="file" accept=".in,.txt,.json" style="display:none;" />
          </label>
        </div>
        <div class="status" id="status">Ready.</div>
        <h2 style="margin:16px -16px 0; border-top:1px solid var(--border);">Exact np_gen.in</h2>
        <div class="panel" style="padding:16px 0 0;">
          <pre id="inputPreview"></pre>
        </div>
      </form>
    </section>
    <section class="card viewer-wrap">
      <h2>Preview</h2>
      <iframe id="previewFrame" name="previewFrame" title="Cluster preview" srcdoc="<html><body style='margin:0;font-family:system-ui,sans-serif;background:#080b12;color:#9aa6bb;display:flex;align-items:center;justify-content:center;min-height:100%;'>Generate a cluster to preview it here.</body></html>"></iframe>
      <div class="legend">
        <div>Server-rendered embedded preview.</div>
        <div id="metaLine">No structure yet.</div>
      </div>
    </section>
  </main>
  <script>
    const shapeFields = {
      wulff: ["wulff-field"],
      ico: ["ico-field"],
      sphere: [],
      cube: ["cube-field"],
      octahedron: ["octahedron-field"],
      decahedron: ["decahedron-field"],
      dodecahedron: [],
      morphed_spherical: ["morphed_spherical-field"],
      cuboct: ["cuboct-field"],
    };
    const hcpCompatibleShapes = new Set([
      "wulff",
      "ico",
      "sphere",
      "hexagonal_prism",
      "truncated_hexagonal_prism",
      "hexagonal_bipyramid",
      "truncated_hexagonal_bipyramid",
      "nanorod",
      "hexagonal_platelet",
    ]);
    const hexOnlyShapes = new Set([
      "hexagonal_prism",
      "truncated_hexagonal_prism",
      "hexagonal_bipyramid",
      "truncated_hexagonal_bipyramid",
      "nanorod",
      "hexagonal_platelet",
    ]);
    const structureDefaults = {
      cubic: {a: 3.0},
      tetragonal: {a: 3.0, c: 5.0},
      orthorhombic: {a: 3.0, b: 4.0, c: 5.0},
      hexagonal: {a: 2.706, c: 4.282},
      rhombohedral: {a: 3.0, alpha: 75.0, beta: 75.0, gamma: 75.0},
      monoclinic: {a: 3.0, b: 4.0, c: 5.0, beta: 110.0},
      triclinic: {a: 3.0, b: 4.0, c: 5.0, alpha: 70.0, beta: 80.0, gamma: 75.0},
    };
    const structureNotes = {
      cubic: "Equal lengths, all angles right angles.",
      tetragonal: "Two equal lengths, all angles right angles.",
      orthorhombic: "Three unequal lengths, all angles right angles.",
      hexagonal: "Two equal basal lengths, gamma is 120°, alpha and beta are 90°.",
      rhombohedral: "All lengths equal, all angles equal but not 90°.",
      monoclinic: "Three unequal lengths, alpha and gamma are 90°, beta is not 90°.",
      triclinic: "Three unequal lengths, angles are all independent.",
    };
    const crystalDefinitionText = "A crystal lattice is defined by three lengths: a, b, c and three angles: alpha, beta, gamma.";
    const structureParameterSummary = {
      cubic: "a = b = c; alpha = beta = gamma = 90°.",
      tetragonal: "a = b ≠ c; alpha = beta = gamma = 90°.",
      orthorhombic: "a ≠ b ≠ c; alpha = beta = gamma = 90°.",
      hexagonal: "a = b ≠ c; alpha = beta = 90°, gamma = 120°.",
      rhombohedral: "a = b = c; alpha = beta = gamma ≠ 90°.",
      monoclinic: "a ≠ b ≠ c; alpha = gamma = 90°, beta ≠ 90°.",
      triclinic: "a ≠ b ≠ c; alpha ≠ beta ≠ gamma, none necessarily 90°.",
    };
    const structureVariants = {
      cubic: [
        ["simple cubic / sc", "simple_cubic"],
        ["bcc", "bcc"],
        ["fcc", "fcc"],
        ["diamond cubic", "diamond_cubic"],
        ["zincblende", "zincblende"],
        ["rocksalt / NaCl", "rocksalt"],
        ["CsCl", "cscl"],
        ["fluorite / CaF₂", "fluorite"],
        ["perovskite / ABO₃", "perovskite"],
        ["spinel", "spinel"],
      ],
      hexagonal: [
        ["hcp", "hcp"],
        ["wurtzite", "wurtzite"],
        ["graphite / graphene-like layered hexagonal", "graphite"],
        ["AlB₂-type", "alb2"],
        ["NiAs-type", "nias"],
      ],
      tetragonal: [
        ["simple tetragonal", "simple_tetragonal"],
        ["body-centered tetragonal / bct", "bct"],
        ["rutile / TiO₂", "rutile"],
        ["anatase / TiO₂", "anatase"],
        ["zircon / ZrSiO₄", "zircon"],
      ],
      orthorhombic: [
        ["simple orthorhombic", "simple_orthorhombic"],
        ["base-centered orthorhombic", "base_centered_orthorhombic"],
        ["body-centered orthorhombic", "body_centered_orthorhombic"],
        ["face-centered orthorhombic", "face_centered_orthorhombic"],
        ["olivine-type", "olivine"],
        ["perovskite-distorted orthorhombic", "perovskite_distorted_orthorhombic"],
      ],
      rhombohedral: [
        ["rhombohedral", "rhombohedral"],
        ["corundum / Al₂O₃", "corundum"],
        ["calcite / CaCO₃", "calcite"],
        ["ilmenite", "ilmenite"],
        ["Bi/Sb-type A7", "a7"],
      ],
      monoclinic: [
        ["simple monoclinic", "simple_monoclinic"],
        ["base-centered monoclinic", "base_centered_monoclinic"],
        ["baddeleyite / ZrO₂", "baddeleyite"],
        ["many molecular/oxide structures", "general_monoclinic"],
      ],
      triclinic: [
        ["simple triclinic", "simple_triclinic"],
        ["low-symmetry molecular crystals", "molecular_triclinic"],
        ["distorted framework materials", "framework_triclinic"],
      ],
    };
    const structureVariantNotes = {
      simple_cubic: "Primitive cubic reference.",
      bcc: "Body-centered cubic variant.",
      fcc: "Face-centered cubic variant.",
      diamond_cubic: "Diamond cubic framework.",
      zincblende: "Binary zincblende framework.",
      rocksalt: "NaCl-type rocksalt.",
      cscl: "CsCl-type cubic binary.",
      fluorite: "CaF2-type fluorite.",
      perovskite: "ABO3 perovskite.",
      spinel: "AB2O4 spinel.",
      hcp: "Hexagonal close-packed.",
      wurtzite: "Binary hexagonal tetrahedral.",
      graphite: "Layered hexagonal graphite/graphene-like.",
      alb2: "AlB2-type hexagonal.",
      nias: "NiAs-type hexagonal.",
      simple_tetragonal: "Primitive tetragonal.",
      bct: "Body-centered tetragonal.",
      rutile: "TiO2 rutile structure.",
      anatase: "TiO2 anatase structure.",
      zircon: "ZrSiO4 zircon structure.",
      simple_orthorhombic: "Primitive orthorhombic.",
      base_centered_orthorhombic: "Base-centered orthorhombic.",
      body_centered_orthorhombic: "Body-centered orthorhombic.",
      face_centered_orthorhombic: "Face-centered orthorhombic.",
      olivine: "Olivine-type orthorhombic.",
      perovskite_distorted_orthorhombic: "Distorted orthorhombic perovskite.",
      rhombohedral: "Rhombohedral/trigonal reference.",
      corundum: "Al2O3 corundum.",
      calcite: "CaCO3 calcite.",
      ilmenite: "Ilmenite-type trigonal.",
      a7: "Bi/Sb A7 structure.",
      simple_monoclinic: "Primitive monoclinic.",
      base_centered_monoclinic: "Base-centered monoclinic.",
      baddeleyite: "ZrO2 baddeleyite.",
      general_monoclinic: "General monoclinic systems.",
      simple_triclinic: "Primitive triclinic.",
      molecular_triclinic: "Low-symmetry molecular triclinic.",
      framework_triclinic: "Distorted framework triclinic.",
    };
    const structureFields = {
      cubic: ["a"],
      tetragonal: ["a", "c"],
      orthorhombic: ["a", "b", "c"],
      hexagonal: ["a", "c"],
      rhombohedral: ["a", "alpha"],
      monoclinic: ["a", "b", "c", "beta"],
      triclinic: ["a", "b", "c", "alpha", "beta", "gamma"],
    };
    const structureAliases = {
      hcp: "hexagonal",
      fcc: "cubic",
      bcc: "cubic",
      trigonal: "rhombohedral",
      custom: "triclinic",
    };

    const el = (id) => document.getElementById(id);
    const status = el("status");
    const inputPreview = el("inputPreview");
    const fileInput = el("fileInput");
    const crystalStructure = el("crystal_structure");
    const crystalVariant = el("crystal_variant");
    const variantNote = el("variantNote");
    const sizeModeSelect = el("size_mode");
    const sizeValueField = el("size_value");
    const sizeValueLabel = el("size_value_label");
    const atomCountField = el("atoms");
    const diameterField = el("diameter");
    const latticeFields = {
      a: el("a"),
      b: el("b"),
      c: el("c"),
      alpha: el("alpha"),
      beta: el("beta"),
      gamma: el("gamma"),
    };
    const structureNote = el("structureNote");
    const crystalDefinition = el("crystalDefinition");
    const shapeNote = el("shapeNote");

    function setStatus(msg) {
      status.textContent = msg;
    }

    function readValue(id) {
      const node = el(id);
      if (!node) return null;
      if (node.disabled) return null;
      if (node.value === "") return null;
      if (node.type === "number") return Number(node.value);
      if (node.type === "checkbox") return node.checked;
      return node.value;
    }

    function normalizeStructure(value) {
      return structureAliases[value] || value || "hexagonal";
    }

    function populateVariantOptions(structure, selected = null) {
      const normalized = normalizeStructure(structure);
      const list = structureVariants[normalized] || [];
      const fallback = list[0] ? list[0][1] : "";
      const current = selected && list.some(([, value]) => value === selected)
        ? selected
        : (crystalVariant.value && list.some(([, value]) => value === crystalVariant.value)
          ? crystalVariant.value
          : fallback);
      crystalVariant.replaceChildren();
      for (const [label, value] of list) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = label;
        crystalVariant.appendChild(opt);
      }
      if (!list.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No subtypes available";
        opt.disabled = true;
        opt.selected = true;
        crystalVariant.appendChild(opt);
      } else {
        crystalVariant.value = current;
        if (!crystalVariant.value && fallback) {
          crystalVariant.value = fallback;
        }
      }
      if (variantNote) {
        variantNote.textContent = structureVariantNotes[current] || "";
      }
    }

    function syncVariantVisibility(structure, selected = null) {
      populateVariantOptions(structure, selected);
    }

    function getSizeMode() {
      return sizeModeSelect.value || "atoms";
    }

    function setSizeMode(mode) {
      sizeModeSelect.value = mode === "diameter" ? "diameter" : "atoms";
      syncSizeVisibility();
    }

    function syncSizeVisibility() {
      const mode = getSizeMode();
      const isDiameter = mode === "diameter";
      sizeValueLabel.textContent = isDiameter ? "Diameter (nm)" : "Number of atoms";
      sizeValueField.step = isDiameter ? "0.001" : "1";
      sizeValueField.placeholder = isDiameter ? "e.g. 2.0" : "e.g. 147";
      if (isDiameter) {
        sizeValueField.value = diameterField.value ? (Number(diameterField.value) / NM_TO_ANGSTROM) : sizeValueField.value;
      } else {
        sizeValueField.value = atomCountField.value || "";
      }
    }

    function syncSizeBackingFields() {
      const mode = getSizeMode();
      if (mode === "diameter") {
        diameterField.value = sizeValueField.value ? (Number(sizeValueField.value) * NM_TO_ANGSTROM) : "";
        atomCountField.value = "";
      } else {
        atomCountField.value = sizeValueField.value;
        diameterField.value = "";
      }
    }

    function syncStructureVisibility() {
      const structure = normalizeStructure(readValue("crystal_structure"));
      const fields = new Set(structureFields[structure] || structureFields.hexagonal);
      structureNote.textContent = structureParameterSummary[structure] || "";
      if (crystalDefinition) {
        crystalDefinition.textContent = crystalDefinitionText;
      }
      syncVariantVisibility(structure, readValue("crystal_variant"));
      syncShapeAvailability(structure, readValue("crystal_variant"));
      Object.entries(latticeFields).forEach(([key, field]) => {
        const visible = fields.has(key);
        field.closest(".field").style.display = visible ? "" : "none";
        field.disabled = !visible;
      });
    }

    function syncShapeAvailability(structure, subtype = null) {
      const normalized = normalizeStructure(structure);
      const normalizedSubtype = normalizeSubtype(subtype || "");
      const shapeSelect = el("shape");
      const restricted = normalized === "hexagonal";
      const fallback = "wulff";
      const allowedShapes = restricted && normalizedSubtype === "hcp"
        ? hcpCompatibleShapes
        : new Set(["wulff", "ico", "sphere"]);
      if (shapeNote) {
        shapeNote.textContent = restricted
          ? (normalizedSubtype === "hcp"
            ? "hcp Ru supports Wulff, Icosahedron, Sphere, and the hexagonal presets here. Other presets are FCC/cubic-only."
            : "This hexagonal subtype uses Wulff, Icosahedron, or Sphere here.")
          : "Sphere is available for every crystal system.";
      }
      Array.from(shapeSelect.options).forEach((opt) => {
        const value = opt.value || opt.textContent.trim();
        const shouldDisable = (restricted && !allowedShapes.has(value)) || (!restricted && hexOnlyShapes.has(value));
        opt.disabled = shouldDisable;
        opt.hidden = shouldDisable;
      });
      if (restricted && !allowedShapes.has(shapeSelect.value)) {
        shapeSelect.value = fallback;
        updateShapeVisibility();
      }
    }

    function inferUiState(spec) {
      const shape = spec.shape || "sphere";
      const structure = normalizeStructure(
        spec.crystal_structure || (shape === "wulff" || shape === "sphere" ? "hexagonal" : "cubic")
      );
      crystalStructure.value = structure;
      syncVariantVisibility(structure, spec.crystal_variant || null);
      if (spec.size_mode === "diameter" || spec.diameter !== null || spec.radius !== null && spec.radius !== undefined && spec.radius !== "") {
        setSizeMode("diameter");
        const diameterAngstrom = spec.diameter !== null && spec.diameter !== undefined && spec.diameter !== ""
          ? Number(spec.diameter)
          : (spec.radius !== null && spec.radius !== undefined && spec.radius !== "" ? Number(spec.radius) * 2 : 20);
        sizeValueField.value = diameterAngstrom / NM_TO_ANGSTROM;
        diameterField.value = diameterAngstrom;
        atomCountField.value = "";
      } else {
        setSizeMode("atoms");
        sizeValueField.value = spec.atoms ?? "";
        atomCountField.value = sizeValueField.value;
        diameterField.value = "";
      }
      for (const [key, field] of Object.entries(latticeFields)) {
        if (spec[key] !== null && spec[key] !== undefined && spec[key] !== "") {
          field.value = spec[key];
        } else if (structureDefaults[structure] && structureDefaults[structure][key] !== undefined) {
          field.value = structureDefaults[structure][key];
        } else {
          field.value = "";
        }
      }
      syncSizeVisibility();
      syncStructureVisibility();
    }

    window.__npfHandleStructureChange = function () {
      syncStructureVisibility();
      updateInputPreview();
    };

    window.__npfPopulateVariantOptions = function (structure) {
      populateVariantOptions(structure, readValue("crystal_variant"));
    };

    window.__npfHandleVariantChange = function () {
      if (variantNote) {
        variantNote.textContent = structureVariantNotes[crystalVariant.value] || "";
      }
      syncShapeAvailability(normalizeStructure(readValue("crystal_structure")), readValue("crystal_variant"));
      updateInputPreview();
    };

    function buildSpec() {
      syncSizeBackingFields();
      const shape = readValue("shape");
      const element = readValue("element");
      const structure = normalizeStructure(readValue("crystal_structure"));
      const sizeModeValue = getSizeMode();
      const atoms = sizeModeValue === "atoms" ? readValue("atoms") : null;
      const diameter = sizeModeValue === "diameter" ? readValue("diameter") : null;
      const diameterNm = sizeModeValue === "diameter" ? readValue("size_value") : null;
      const radius = sizeModeValue === "diameter" && diameter !== null ? diameter / 2 : null;
      const sizeToken = sizeModeValue === "atoms"
        ? (atoms ?? "NA")
        : (diameterNm !== null ? `d${diameterNm}nm` : "diameter");
      const spec = {
        shape,
        input_file_name: readValue("input_file_name"),
        element,
        output_file_name: readValue("output_file_name") || `${element || "Ru"}_${shape}_${sizeToken}.xyz`,
        atoms,
        radius,
        a: readValue("a"),
        b: readValue("b"),
        c: readValue("c"),
        alpha: readValue("alpha"),
        beta: readValue("beta"),
        gamma: readValue("gamma"),
        verbose: readValue("verbose") === "true",
      };
      if (structure) spec.crystal_structure = structure;
      const variant = readValue("crystal_variant");
      if (variant) spec.crystal_variant = variant;
      spec.tag = spec.output_file_name.replace(/\.xyz$/i, "");
      if (shape === "wulff") {
        Object.assign(spec, {
          surfaces: readValue("surfaces"),
          facet_mode: readValue("facet_mode"),
          extra_facets: readValue("extra_facets"),
          sphericity: readValue("sphericity"),
          radius_tol: readValue("radius_tol"),
          max_refine: readValue("max_refine"),
          radius_policy: readValue("radius_policy"),
          radial_trim: readValue("radial_trim"),
        });
      } else if (shape === "ico") {
        Object.assign(spec, { ico_shells: readValue("ico_shells") });
      } else if (shape === "cube") {
        Object.assign(spec, { cube_layers: readValue("cube_layers") });
      } else if (shape === "octahedron") {
        Object.assign(spec, {
          octa_length: readValue("octa_length"),
          octa_cutoff: readValue("octa_cutoff"),
        });
      } else if (shape === "decahedron") {
        Object.assign(spec, {
          p: readValue("p"),
          q: readValue("q"),
          r: readValue("r"),
        });
      } else if (shape === "morphed_spherical") {
        Object.assign(spec, { morph: readValue("morph") });
      } else if (shape === "cuboct") {
        Object.assign(spec, { cuboct_layers: readValue("cuboct_layers") });
      }
      return spec;
    }

    function specToText(spec) {
      const lines = [];
      for (const [key, value] of Object.entries(spec)) {
        if (["crystal_structure", "crystal_variant", "size_mode", "diameter"].includes(key)) continue;
        if (value === null || value === undefined || value === "") continue;
        lines.push(`${key} = ${value}`);
      }
      return lines.join("\\n") + "\\n";
    }

    function updateInputPreview() {
      syncSizeBackingFields();
      inputPreview.textContent = specToText(buildSpec());
    }

    function updateShapeVisibility() {
      const shape = readValue("shape");
      const active = new Set(shapeFields[shape] || []);
      document.querySelectorAll(".shape-field").forEach((node) => {
        const isVisible = active.size > 0 && [...active].some((cls) => node.classList.contains(cls));
        node.style.display = isVisible ? "" : "none";
      });
    }

    async function loadInputText(text, filename="input file") {
      setStatus(`Loading ${filename}...`);
      const resp = await fetch("/api/load", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text, filename}),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setStatus(data.error || "Failed to load input file.");
        return;
      }
      for (const [key, value] of Object.entries(data.spec)) {
        const node = el(key);
        if (node && value !== null && value !== undefined) {
          node.value = value;
        }
      }
      inferUiState(data.spec);
      updateShapeVisibility();
      updateInputPreview();
      setStatus(`Loaded ${filename}.`);
    }

    async function refreshInitial() {
      const resp = await fetch("/api/state");
      const data = await resp.json();
      const spec = data.spec;
      for (const [key, value] of Object.entries(spec)) {
        const node = el(key);
        if (node && value !== null && value !== undefined) {
          node.value = value;
        }
      }
      inferUiState(spec);
      updateShapeVisibility();
      updateInputPreview();
      setStatus(`Ready. Working path: ${data.spec.working_path}`);
    }

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      const text = await file.text();
      await loadInputText(text, file.name);
      fileInput.value = "";
    });

    document.querySelectorAll("input, select, textarea").forEach((node) => {
      node.addEventListener("input", updateInputPreview);
      node.addEventListener("change", updateInputPreview);
    });

    el("shape").addEventListener("change", () => {
      updateShapeVisibility();
      updateInputPreview();
    });
    crystalVariant.addEventListener("change", () => {
      window.__npfHandleVariantChange();
    });
    sizeModeSelect.addEventListener("change", () => {
      syncSizeVisibility();
      syncSizeBackingFields();
      updateInputPreview();
    });
    sizeValueField.addEventListener("input", () => {
      syncSizeBackingFields();
      updateInputPreview();
    });
    sizeValueField.addEventListener("change", () => {
      syncSizeBackingFields();
      updateInputPreview();
    });
    el("crystal_structure").addEventListener("change", () => {
      window.__npfPopulateVariantOptions && window.__npfPopulateVariantOptions(readValue("crystal_structure"));
      window.__npfHandleStructureChange();
    });

    refreshInitial();
  </script>
</body>
</html>
"""


def spec_to_input_text(spec: dict) -> str:
    lines = []
    for key, value in spec.items():
        if key in {"size_mode", "diameter"}:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def _first(form: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    values = form.get(key)
    if not values:
        return default
    return values[0]


def _int_or_none(value: str | None):
    if value is None or value == "":
        return None
    return int(value)


def _float_or_none(value: str | None):
    if value is None or value == "":
        return None
    return float(value)


def _bool_from_text(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def build_spec_from_form(form: dict[str, list[str]]) -> dict:
    shape = _first(form, "shape", "wulff")
    element = _first(form, "element", "Ru")
    crystal_structure = _first(form, "crystal_structure", "hexagonal")
    if crystal_structure == "hcp":
      crystal_structure = "hexagonal"
    elif crystal_structure in {"fcc", "bcc"}:
      crystal_structure = "cubic"
    elif crystal_structure == "trigonal":
      crystal_structure = "rhombohedral"
    elif crystal_structure == "custom":
      crystal_structure = "triclinic"
    size_mode = _first(form, "size_mode", None)
    if size_mode not in {"atoms", "diameter"}:
        size_mode = "diameter"
    atoms = _int_or_none(_first(form, "atoms")) if size_mode == "atoms" else None
    diameter = _float_or_none(_first(form, "diameter")) if size_mode == "diameter" else None
    radius = (diameter / 2.0) if diameter is not None else None
    lattice_fields = {
        "cubic": {"a"},
        "tetragonal": {"a", "c"},
        "orthorhombic": {"a", "b", "c"},
        "hexagonal": {"a", "c"},
        "rhombohedral": {"a", "alpha"},
        "monoclinic": {"a", "b", "c", "beta"},
        "triclinic": {"a", "b", "c", "alpha", "beta", "gamma"},
    }.get(crystal_structure, {"a", "c"})
    lattice_values = {
        "a": _float_or_none(_first(form, "a")),
        "b": _float_or_none(_first(form, "b")),
        "c": _float_or_none(_first(form, "c")),
        "alpha": _float_or_none(_first(form, "alpha")),
        "beta": _float_or_none(_first(form, "beta")),
        "gamma": _float_or_none(_first(form, "gamma")),
    }
    output_file_name = _first(form, "output_file_name")
    if not output_file_name:
        if size_mode == "diameter" and diameter is not None:
            size_part = f"d{diameter:g}"
        else:
            size_part = f"{atoms}" if atoms is not None else "NA"
        output_file_name = f"{element}_{shape}_{size_part}.xyz"
    spec = {
        "shape": shape,
        "input_file_name": _first(form, "input_file_name", "np_gen.in"),
        "output_file_name": output_file_name,
        "tag": Path(output_file_name).stem,
        "element": element,
        "crystal_structure": crystal_structure,
        "crystal_variant": _first(form, "crystal_variant"),
        "size_mode": size_mode,
        "atoms": atoms,
        "radius": radius,
        "verbose": _bool_from_text(_first(form, "verbose"), True),
    }
    for key, value in lattice_values.items():
        if key in lattice_fields:
            if key == "a":
                spec[key] = value if value is not None else DEFAULT_A
            elif key == "c" and crystal_structure in {"hexagonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic"}:
                spec[key] = value if value is not None else DEFAULT_C
            else:
                spec[key] = value
    if shape == "wulff":
        spec.update({
            "surfaces": _first(form, "surfaces", ""),
            "facet_mode": _first(form, "facet_mode", "top"),
            "extra_facets": _first(form, "extra_facets", "none"),
            "sphericity": _float_or_none(_first(form, "sphericity")) or 0.0,
            "radius_tol": _float_or_none(_first(form, "radius_tol")) or 0.3,
            "max_refine": _int_or_none(_first(form, "max_refine")) or 12,
            "radius_policy": _first(form, "radius_policy", "closest"),
            "radial_trim": _float_or_none(_first(form, "radial_trim")),
        })
    elif shape == "ico":
        spec["ico_shells"] = _int_or_none(_first(form, "ico_shells"))
    elif shape == "cube":
        spec["cube_layers"] = _int_or_none(_first(form, "cube_layers"))
    elif shape == "octahedron":
        spec["octa_length"] = _int_or_none(_first(form, "octa_length"))
        spec["octa_cutoff"] = _int_or_none(_first(form, "octa_cutoff")) or 0
    elif shape == "decahedron":
        spec["p"] = _int_or_none(_first(form, "p"))
        spec["q"] = _int_or_none(_first(form, "q"))
        spec["r"] = _int_or_none(_first(form, "r"))
    elif shape == "morphed_spherical":
        spec["morph"] = _float_or_none(_first(form, "morph")) or 0.0
    elif shape == "cuboct":
        spec["cuboct_layers"] = _int_or_none(_first(form, "cuboct_layers"))
    return spec


def _project_points(positions, width=1000, height=760):
    rot_x = -0.45
    rot_y = 0.65
    sx, cxr = __import__("math").sin(rot_x), __import__("math").cos(rot_x)
    sy, cyr = __import__("math").sin(rot_y), __import__("math").cos(rot_y)
    pts = []
    for pos in positions:
        x, y, z = pos
        y1 = y * cxr - z * sx
        z1 = y * sx + z * cxr
        x2 = x * cyr + z1 * sy
        z2 = -x * sy + z1 * cyr
        scale = 180.0 / (1.0 + z2 / 900.0)
        pts.append((width / 2 + x2 * scale / 100, height / 2 + y1 * scale / 100, z2))
    return pts


def _build_bonds(positions):
    if len(positions) < 2:
        return []
    import math
    min_d = float("inf")
    for i in range(len(positions)):
        ax, ay, az = positions[i]
        for j in range(i + 1, len(positions)):
            bx, by, bz = positions[j]
            d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            if 1e-6 < d < min_d:
                min_d = d
    if not math.isfinite(min_d):
        return []
    cutoff = min_d * 1.24
    bonds = []
    for i in range(len(positions)):
        ax, ay, az = positions[i]
        for j in range(i + 1, len(positions)):
            bx, by, bz = positions[j]
            d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            if d <= cutoff:
                bonds.append((i, j))
    return bonds


def render_preview_html(spec: dict, atoms_obj, input_file_path: Path, result=None) -> str:
    positions = atoms_obj.get_positions().tolist()
    symbols = atoms_obj.get_chemical_symbols()
    bonds = _build_bonds(positions)
    exact_input = html.escape(spec_to_input_text(spec))
    cluster_data = json.dumps({
        "positions": positions,
        "symbols": symbols,
        "bonds": bonds,
        "atoms": len(positions),
        "center": atoms_obj.get_positions().mean(axis=0).tolist(),
        "shape": spec.get("shape", "wulff"),
        "output_file_name": spec.get("output_file_name", ""),
        "formula": getattr(result, "formula", None),
    }).replace("</", "<\\/")
    formula = html.escape(getattr(result, "formula", "") or "")
    output_name = html.escape(str(spec.get("output_file_name", "")))
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080b12;
      --panel: #11131a;
      --panel-2: #171b24;
      --text: #d8e2f1;
      --muted: #9aa6bb;
      --border: #2b3242;
      --accent: #6fe3c0;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      height: 100%;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, sans-serif;
    }}
    .wrap {{
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100%;
    }}
    .bar {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01)), var(--panel);
      display: flex;
      gap: 12px;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
    }}
    .bar strong {{ font-weight: 650; }}
    .bar .meta {{ color: var(--muted); font-size: 13px; }}
    .bar button {{
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .control {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255,255,255,.03);
      color: var(--text);
      font-size: 13px;
    }}
    .control select {{
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 999px;
      padding: 6px 10px;
      min-width: 150px;
    }}
    .control input[type="range"] {{
      width: 160px;
      accent-color: var(--accent);
    }}
    .panel {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
      padding: 16px;
      min-height: 0;
      align-items: stretch;
    }}
    .viewer-shell {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-height: 520px;
      height: 100%;
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      background: radial-gradient(circle at top, #101725 0%, #080b12 70%);
      box-shadow: 0 20px 60px rgba(0,0,0,.35);
    }}
    .hint {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      font-size: 13px;
      background: rgba(255,255,255,.02);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      touch-action: none;
      cursor: grab;
    }}
    canvas:active {{ cursor: grabbing; }}
    pre {{
      margin: 0;
      padding: 12px;
      background: rgba(255,255,255,.04);
      border: 1px solid var(--border);
      border-radius: 12px;
      white-space: pre-wrap;
      overflow: auto;
      min-height: 0;
      max-height: calc(100vh - 140px);
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    @media (max-width: 980px) {{
      .panel {{ grid-template-columns: 1fr; }}
      pre {{ max-height: 320px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <div>
        <strong>NPF Preview</strong>
        <div class="meta">Generated: {html.escape(str(input_file_path))} | Working path: {html.escape(str(input_file_path.parent))} | Output: {output_name}</div>
      </div>
      <div class="meta">{formula if formula else html.escape(spec.get("shape", "wulff"))} | {len(positions)} atoms</div>
      <div class="controls">
        <label class="control" for="atomScale">
          Atom radius
          <input id="atomScale" type="range" min="0.4" max="2.5" step="0.05" value="1.0" />
          <span id="atomScaleValue">1.00x</span>
        </label>
        <label class="control" for="projectionMode">
          Projection
          <select id="projectionMode">
            <option value="perspective">Perspective</option>
            <option value="orthographic" selected>Orthographic</option>
            <option value="oblique">Oblique</option>
          </select>
        </label>
        <button id="resetBtn" type="button">Reset view</button>
      </div>
    </div>
    <div class="panel">
      <div class="viewer-shell">
        <div class="hint">
          <span>Drag to rotate. Wheel or trackpad to zoom. Double-click to reset.</span>
          <span id="viewState">ready</span>
        </div>
        <canvas id="viewer"></canvas>
      </div>
      <div>
        <div class="note">Exact input written to disk:</div>
        <pre>{exact_input}</pre>
      </div>
    </div>
  </div>
  <script id="cluster-data" type="application/json">{cluster_data}</script>
  <script>
    const data = JSON.parse(document.getElementById("cluster-data").textContent);
    const canvas = document.getElementById("viewer");
    const ctx = canvas.getContext("2d");
    const viewState = document.getElementById("viewState");
    const resetBtn = document.getElementById("resetBtn");
    const atomScaleInput = document.getElementById("atomScale");
    const atomScaleValue = document.getElementById("atomScaleValue");
    const projectionMode = document.getElementById("projectionMode");
    let resizeRaf = null;

    const state = {{
      rotX: -0.45,
      rotY: 0.65,
      zoom: 1.0,
      atomScale: 1.0,
      projection: "orthographic",
      dragging: false,
      lastX: 0,
      lastY: 0,
    }};

    const palette = {{
      Ru: "#b8c0cc",
      Rh: "#b0bac7",
      Ir: "#aab4c3",
      Pt: "#d4d8e0",
      Pd: "#c7ced8",
      Au: "#ffd66e",
      Ag: "#d5dbe3",
      Cu: "#d79b6d",
      Ni: "#9fb2a8",
      Co: "#a5b6ca",
      Fe: "#c79a8f",
      default: "#b6bec9",
    }};

    function colorFor(symbol) {{
      return palette[symbol] || palette.default;
    }}

    function hexToRgb(hex) {{
      const normalized = hex.replace("#", "");
      const value = normalized.length === 3
        ? normalized.split("").map((c) => c + c).join("")
        : normalized;
      const num = parseInt(value, 16);
      return {{ r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 }};
    }}

    function rgbToCss(rgb, alpha = 1) {{
      return "rgba(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ", " + alpha + ")";
    }}

    function mixColor(hex, target, t) {{
      const a = hexToRgb(hex);
      const b = typeof target === "string" ? hexToRgb(target) : target;
      return {{
        r: Math.round(a.r + (b.r - a.r) * t),
        g: Math.round(a.g + (b.g - a.g) * t),
        b: Math.round(a.b + (b.b - a.b) * t),
      }};
    }}

    function centeredPositions() {{
      const center = data.center || [0, 0, 0];
      return data.positions.map((pos) => [
        pos[0] - center[0],
        pos[1] - center[1],
        pos[2] - center[2],
      ]);
    }}

    function resize() {{
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}

    function rotatePoint(pos) {{
      const x = pos[0], y = pos[1], z = pos[2];
      const sx = Math.sin(state.rotX), cx = Math.cos(state.rotX);
      const sy = Math.sin(state.rotY), cy = Math.cos(state.rotY);
      const y1 = y * cx - z * sx;
      const z1 = y * sx + z * cx;
      const x2 = x * cy + z1 * sy;
      const z2 = -x * sy + z1 * cy;
      return [x2, y1, z2];
    }}

    function bounds() {{
      let maxR = 1;
      for (const pos of centeredPositions()) {{
        const r = Math.hypot(pos[0], pos[1], pos[2]);
        if (r > maxR) maxR = r;
      }}
      return maxR;
    }}

    function project(pos, w, h, scale) {{
      const [x, y, z] = rotatePoint(pos);
      if (state.projection === "orthographic") {{
        const px = w / 2 + x * scale;
        const py = h / 2 + y * scale;
        return [px, py, z, 1.0];
      }}
      if (state.projection === "oblique") {{
        const skew = 0.42;
        const px = w / 2 + (x + skew * z) * scale;
        const py = h / 2 + (y - skew * 0.55 * z) * scale;
        const depth = 1 / (1 + Math.max(-0.85, z / (scale * 8.5)));
        return [px, py, z, depth];
      }}
      const perspective = 1 / (1 + Math.max(-0.85, z / (scale * 8.5)));
      const px = w / 2 + x * scale * perspective;
      const py = h / 2 + y * scale * perspective;
      return [px, py, z, perspective];
    }}

    function drawSphere(x, y, r, fillHex, alpha, shade) {{
      const light = mixColor(fillHex, "#ffffff", 0.48 + 0.12 * shade);
      const mid = mixColor(fillHex, "#b7bcc4", 0.24);
      const dark = mixColor(fillHex, "#11161d", 0.74);
      const g = ctx.createRadialGradient(
        x - r * 0.42,
        y - r * 0.42,
        r * 0.08,
        x,
        y,
        r
      );
      g.addColorStop(0, "rgba(255,255,255,1)");
      g.addColorStop(0.14, rgbToCss(light, 1));
      g.addColorStop(0.42, rgbToCss(mid, 1));
      g.addColorStop(0.72, rgbToCss(hexToRgb(fillHex), 1));
      g.addColorStop(1, rgbToCss(dark, 1));
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();
      ctx.lineWidth = Math.max(1, r * 0.08);
      ctx.strokeStyle = "rgba(255,255,255,0.10)";
      ctx.stroke();
      const spec = ctx.createRadialGradient(
        x - r * 0.22,
        y - r * 0.28,
        0,
        x - r * 0.22,
        y - r * 0.28,
        r * 0.42
      );
      spec.addColorStop(0, "rgba(255,255,255,0.55)");
      spec.addColorStop(0.24, "rgba(255,255,255,0.18)");
      spec.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = spec;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }}

    function draw() {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (!w || !h) return;
      ctx.clearRect(0, 0, w, h);
      const maxR = bounds();
      const scale = Math.min(w, h) * 0.34 / maxR * state.zoom;
      const points = centeredPositions();
      const projected = points.map((pos, i) => {{
        const p = project(pos, w, h, scale);
        return {{
          i,
          x: p[0],
          y: p[1],
          z: p[2],
          depth: p[3],
          symbol: data.symbols[i],
          pos,
        }};
      }});

      ctx.save();
      ctx.strokeStyle = "rgba(185,200,225,0.18)";
      ctx.lineCap = "round";
      ctx.lineWidth = 1.2;
      for (const [i, j] of data.bonds) {{
        const a = projected[i];
        const b = projected[j];
        const depth = Math.max(0.15, Math.min(1, 1 - ((a.z + b.z) / (scale * 16))));
        ctx.globalAlpha = 0.12 + 0.28 * depth;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }}
      ctx.globalAlpha = 1;
      ctx.restore();

      projected.sort((a, b) => a.z - b.z);
      for (const atom of projected) {{
        const depth = Math.max(0.25, Math.min(1, 1 - atom.z / (scale * 10)));
        const radius = Math.max(3.0, 9.5 * depth * state.atomScale);
        const fill = colorFor(atom.symbol);
        drawSphere(atom.x, atom.y, radius, fill, 0.34 + 0.5 * depth, depth);
      }}

      viewState.textContent = "zoom " + state.zoom.toFixed(2) + " | radius " + state.atomScale.toFixed(2) + "x | projection " + state.projection + " | atoms " + data.atoms;
    }}

    function resetView() {{
      state.rotX = -0.45;
      state.rotY = 0.65;
      state.zoom = 1.0;
      state.atomScale = 1.0;
      atomScaleInput.value = "1.0";
      atomScaleValue.textContent = "1.00x";
      draw();
    }}

    atomScaleInput.addEventListener("input", () => {{
      state.atomScale = Number(atomScaleInput.value);
      atomScaleValue.textContent = state.atomScale.toFixed(2) + "x";
      draw();
    }});

    projectionMode.addEventListener("change", () => {{
      state.projection = projectionMode.value;
      draw();
    }});

    canvas.addEventListener("pointerdown", (event) => {{
      state.dragging = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});

      canvas.addEventListener("pointermove", (event) => {{
        if (!state.dragging) return;
        const dx = event.clientX - state.lastX;
        const dy = event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        state.rotY += dx * 0.006;
        state.rotX += dy * 0.006;
        draw();
      }});

    function endDrag(event) {{
      state.dragging = false;
      try {{ canvas.releasePointerCapture(event.pointerId); }} catch (_) {{}}
    }}

    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const delta = Math.sign(event.deltaY);
      state.zoom *= delta > 0 ? 0.92 : 1.08;
      state.zoom = Math.max(0.35, Math.min(3.0, state.zoom));
      draw();
    }}, {{ passive: false }});

    canvas.addEventListener("dblclick", (event) => {{
      event.preventDefault();
      resetView();
    }});

    resetBtn.addEventListener("click", resetView);
    window.addEventListener("resize", () => {{
      if (resizeRaf) cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(resize);
    }});
    if (window.ResizeObserver) {{
      const observer = new ResizeObserver(() => resize());
      observer.observe(canvas);
      observer.observe(canvas.parentElement);
    }}
    window.addEventListener("load", () => requestAnimationFrame(resize));
    requestAnimationFrame(resize);
  </script>
</body>
</html>"""
    return page


class NPFGUIHandler(BaseHTTPRequestHandler):
    server_version = "NPFGui/0.1"

    def _send(self, payload: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            page = INDEX_HTML.replace("{{WORKING_PATH}}", html.escape(str(ROOT)))
            self._send(page.encode("utf-8"))
            return
        if route == "/api/state":
            payload = json.dumps({"spec": asdict(GUIState())}).encode("utf-8")
            self._send(payload, "application/json")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            if route == "/api/load":
                body = json.loads(raw.decode("utf-8"))
                text = body["text"]
                spec = parse_input_text(text)
                payload = json.dumps({
                    "spec": spec,
                    "input_text": spec_to_input_text(spec),
                }).encode("utf-8")
                self._send(payload, "application/json")
                return

            if route not in {"/api/generate-page", "/api/generate"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            if route == "/api/generate":
                body = json.loads(raw.decode("utf-8"))
                spec = body["spec"]
                write_only = bool(body.get("write_only", False))
                input_file_name = spec.get("input_file_name") or "np_gen.in"
            else:
                form = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
                spec = build_spec_from_form(form)
                write_only = _first(form, "mode", "generate") == "write_only"
                input_file_name = spec.get("input_file_name") or "np_gen.in"

            output_dir = cluster_output_dir(ROOT, tag)
            input_path = output_dir / input_file_name
            input_path.write_text(spec_to_input_text(spec), encoding="utf-8")
            if write_only:
                if route == "/api/generate":
                    payload = json.dumps({"input_file": str(input_path), "input_text": spec_to_input_text(spec)}).encode("utf-8")
                    self._send(payload, "application/json")
                else:
                    html_page = f"""<!doctype html><html><body style="font-family:system-ui,sans-serif;background:#080b12;color:#d8e2f1;padding:24px;">
<h2>Input written</h2><p>{html.escape(str(input_path))}</p><p>Working path: {html.escape(str(output_dir))}</p></body></html>"""
                    self._send(html_page.encode("utf-8"))
                return

            result = build_from_spec(spec)
            from ase.io import read
            tag = spec.get("tag") or Path(spec.get("output_file_name") or "Ru_NP").stem
            output_dir = cluster_output_dir(ROOT, tag)
            xyz_path = output_dir / f"{tag}.xyz"
            if not xyz_path.exists():
                raise FileNotFoundError(f"Generated XYZ not found at {xyz_path}")
            atoms_obj = read(str(xyz_path))
            if route == "/api/generate":
                payload = {
                    "input_file": str(input_path),
                    "input_text": spec_to_input_text(spec),
                    "meta": {
                        "formula": result.formula,
                        "n_atoms": result.atoms_count,
                        "actual_radius": result.actual_radius,
                        "shape": spec.get("shape", "wulff"),
                    },
                    "atoms": [{"symbol": s, "position": p} for s, p in zip(atoms_obj.get_chemical_symbols(), atoms_obj.get_positions().tolist())],
                }
                payload["meta"]["center"] = atoms_obj.get_positions().mean(axis=0).tolist()
                self._send(json.dumps(payload).encode("utf-8"), "application/json")
            else:
                page = render_preview_html(spec, atoms_obj, input_path, result)
                self._send(page.encode("utf-8"))
        except Exception as exc:
            if route == "/api/generate":
                self._send(json.dumps({"error": str(exc)}).encode("utf-8"), "application/json", status=500)
            else:
                error_page = f"""<!doctype html><html><body style="font-family:system-ui,sans-serif;background:#080b12;color:#ffcf7a;padding:24px;">
<h2>Generation failed</h2><pre>{html.escape(str(exc))}</pre></body></html>"""
                self._send(error_page.encode("utf-8"), status=500)


def run_gui(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    os.chdir(ROOT)
    server = ThreadingHTTPServer((host, port), NPFGUIHandler)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open_new(url)).start()
    print(f"NPF GUI running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Start the NPF GUI.")
    p.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    p.add_argument("--port", type=int, default=8765, help="Port to bind.")
    p.add_argument("--no-browser", action="store_true", help="Do not automatically open a browser window.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_gui(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
