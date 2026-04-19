#!/usr/bin/env python3.10
"""
ReMEmbR 記憶檢視器
生成 HTML 報告：2D 位置地圖 + 記憶內容表格，方便檢視機器人的空間記憶是否正確。

用法:
    python3.10 memory_viewer.py                          # 使用預設 DB
    python3.10 memory_viewer.py --db ./remembr_memory.db # 指定 DB 路徑
    python3.10 memory_viewer.py --serve                  # 生成後啟動 HTTP 伺服器
"""
import sys
import os
import argparse
import json
import io
import base64
from time import strftime, localtime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

FIXED_SUBTRACT = 1721761000


def load_memories(db_path: str, collection_name: str = "robot_memory"):
    from pymilvus import MilvusClient
    client = MilvusClient(db_path)
    stats = client.get_collection_stats(collection_name)
    total = stats.get("row_count", 0)
    if total == 0:
        client.close()
        return []

    try:
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id", "caption", "position", "time", "theta", "image_path"],
            limit=total + 10,
        )
    except Exception:
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id", "caption", "position", "time", "theta"],
            limit=total + 10,
        )
        for r in results:
            r["image_path"] = ""
    client.close()
    return results


def generate_map_image(memories: list) -> str:
    """Generate a 2D scatter plot of memory positions, return as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

    positions = np.array([m["position"][:2] for m in memories])
    times = []
    for m in memories:
        t_val = m["time"]
        if isinstance(t_val, list) and len(t_val) == 2:
            t_val = t_val[0]
        times.append(float(t_val))
    times = np.array(times)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=times, cmap="viridis", s=60, alpha=0.8, edgecolors="white", linewidths=0.5,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Time (relative)")
    cbar.ax.set_ylabel("Time (earlier → later)", fontsize=9)

    for i, m in enumerate(memories):
        cap = m["caption"][:40].replace("\n", " ")
        x, y = m["position"][0], m["position"][1]
        if i % 3 == 0:
            ax.annotate(
                f"#{i}", (x, y), fontsize=6, alpha=0.6,
                xytext=(4, 4), textcoords="offset points",
            )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"Robot Memory Map ({len(memories)} memories)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    unique_pos = {}
    for m in memories:
        key = (round(m["position"][0], 1), round(m["position"][1], 1))
        unique_pos[key] = unique_pos.get(key, 0) + 1
    for (rx, ry), count in unique_pos.items():
        if count > 2:
            ax.annotate(
                f"×{count}", (rx, ry), fontsize=8, color="red", fontweight="bold",
                xytext=(6, -8), textcoords="offset points",
            )

    ax.plot(positions[0, 0], positions[0, 1], "g^", markersize=12, label="Start", zorder=5)
    ax.plot(positions[-1, 0], positions[-1, 1], "rs", markersize=10, label="End", zorder=5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_html(memories: list, db_path: str, map_b64: str, image_dir: str = "") -> str:
    rows = []
    has_any_image = False
    for i, m in enumerate(memories):
        t_val = m["time"]
        if isinstance(t_val, list) and len(t_val) == 2:
            t_val = t_val[0]
        t_val = float(t_val) + FIXED_SUBTRACT
        t_str = strftime("%Y-%m-%d %H:%M:%S", localtime(t_val))

        pos = [round(float(x), 3) for x in m["position"]]
        theta = round(float(m.get("theta", 0)), 2)
        caption = m["caption"].replace("\n", "<br>")
        caption_short = m["caption"][:120].replace("\n", " ")

        img_path = m.get("image_path", "")
        img_html = '<span class="no-img">無圖片</span>'
        if img_path:
            full_path = os.path.join(image_dir, img_path) if image_dir else img_path
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("ascii")
                img_html = f'<img src="data:image/jpeg;base64,{img_b64}" class="mem-img" onclick="showLarge(this)">'
                has_any_image = True
            else:
                img_html = f'<span class="no-img">{img_path} (missing)</span>'

        rows.append(f"""
        <tr class="mem-row" data-x="{pos[0]}" data-y="{pos[1]}">
            <td class="idx">{i}</td>
            <td class="img-cell">{img_html}</td>
            <td class="time">{t_str}</td>
            <td class="pos">({pos[0]}, {pos[1]})</td>
            <td class="theta">{theta}°</td>
            <td class="caption">
                <div class="caption-short">{caption_short}...</div>
                <div class="caption-full" style="display:none">{caption}</div>
                <button class="toggle-btn" onclick="toggleCaption(this)">展開</button>
            </td>
        </tr>""")

    positions = [m["position"][:2] for m in memories]
    pos_arr = np.array(positions)
    rounded = np.round(pos_arr, 1)
    unique_map = {}
    for i, p in enumerate(rounded):
        key = f"({p[0]}, {p[1]})"
        if key not in unique_map:
            unique_map[key] = {"count": 0, "indices": []}
        unique_map[key]["count"] += 1
        unique_map[key]["indices"].append(i)

    cluster_rows = []
    for pos_key, info in sorted(unique_map.items(), key=lambda x: -x[1]["count"]):
        idx_str = ", ".join(str(x) for x in info["indices"][:8])
        if len(info["indices"]) > 8:
            idx_str += f" ... (+{len(info['indices'])-8})"
        cluster_rows.append(
            f'<tr><td>{pos_key}</td><td>{info["count"]}</td><td class="idx-list">{idx_str}</td></tr>'
        )

    table_rows = "\n".join(rows)
    cluster_table = "\n".join(cluster_rows)

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>ReMEmbR Memory Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1 {{ color: #58a6ff; margin-bottom: 8px; }}
h2 {{ color: #79c0ff; margin: 24px 0 12px; }}
.meta {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 20px; }}
.map-container {{ text-align: center; margin: 20px 0; background: #161b22; border-radius: 12px; padding: 16px; }}
.map-container img {{ max-width: 100%; border-radius: 8px; }}
.stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 24px; text-align: center; }}
.stat-card .val {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
.stat-card .label {{ font-size: 0.85rem; color: #8b949e; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
th {{ background: #161b22; color: #79c0ff; padding: 10px 12px; text-align: left; position: sticky; top: 0; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; vertical-align: top; }}
tr:hover {{ background: #161b2280; }}
.idx {{ color: #8b949e; width: 40px; }}
.time {{ white-space: nowrap; font-size: 0.85rem; color: #58a6ff; }}
.pos {{ white-space: nowrap; font-family: monospace; color: #7ee787; }}
.theta {{ font-family: monospace; color: #d2a8ff; width: 60px; }}
.caption {{ font-size: 0.85rem; line-height: 1.5; max-width: 600px; }}
.caption-short {{ color: #c9d1d9; }}
.caption-full {{ color: #e6edf3; background: #0d1117; padding: 8px; border-radius: 4px; margin-top: 4px; }}
.toggle-btn {{ background: #21262d; color: #58a6ff; border: 1px solid #30363d; border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 0.75rem; margin-top: 4px; }}
.toggle-btn:hover {{ background: #30363d; }}
.idx-list {{ font-family: monospace; font-size: 0.8rem; color: #8b949e; }}
.img-cell {{ width: 120px; text-align: center; }}
.mem-img {{ width: 110px; height: 80px; object-fit: cover; border-radius: 6px; cursor: pointer; transition: transform 0.2s; }}
.mem-img:hover {{ transform: scale(1.05); }}
.no-img {{ color: #484f58; font-size: 0.75rem; }}
#lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 999; justify-content: center; align-items: center; cursor: pointer; }}
#lightbox img {{ max-width: 90%; max-height: 90%; border-radius: 8px; }}
#lightbox.active {{ display: flex; }}
.search-box {{ margin: 12px 0; }}
.search-box input {{ background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; padding: 8px 16px; border-radius: 8px; width: 300px; font-size: 0.95rem; }}
.search-box input:focus {{ outline: none; border-color: #58a6ff; }}
.highlight {{ background: #341a00; }}
</style>
</head>
<body>
<h1>ReMEmbR Memory Viewer</h1>
<div class="meta">DB: {db_path} | Generated: {strftime("%Y-%m-%d %H:%M:%S")} | Total: {len(memories)} memories</div>

<div class="stats">
    <div class="stat-card"><div class="val">{len(memories)}</div><div class="label">Total Memories</div></div>
    <div class="stat-card"><div class="val">{len(unique_map)}</div><div class="label">Unique Positions</div></div>
    <div class="stat-card"><div class="val">{pos_arr[:,0].min():.1f} ~ {pos_arr[:,0].max():.1f}</div><div class="label">X Range (m)</div></div>
    <div class="stat-card"><div class="val">{pos_arr[:,1].min():.1f} ~ {pos_arr[:,1].max():.1f}</div><div class="label">Y Range (m)</div></div>
    <div class="stat-card"><div class="val">{sum(1 for m in memories if m.get('image_path'))}</div><div class="label">With Image</div></div>
</div>

<h2>Position Map</h2>
<div class="map-container">
    <img src="data:image/png;base64,{map_b64}" alt="Memory Map">
</div>

<h2>Position Clusters</h2>
<p style="color:#8b949e;font-size:0.85rem;margin-bottom:8px">相同位置（0.1m 內）的記憶分組，數量過多表示該處停留時間較長</p>
<table>
<tr><th>Position</th><th>Count</th><th>Memory Indices</th></tr>
{cluster_table}
</table>

<h2>All Memories</h2>
<div class="search-box">
    <input type="text" id="search" placeholder="搜尋 caption 關鍵字..." oninput="filterTable()">
</div>
<div id="lightbox" onclick="this.classList.remove('active')"><img></div>
<table id="mem-table">
<tr><th>#</th><th>Image</th><th>Time</th><th>Position</th><th>θ</th><th>Caption</th></tr>
{table_rows}
</table>

<script>
function showLarge(img) {{
    const lb = document.getElementById('lightbox');
    lb.querySelector('img').src = img.src;
    lb.classList.add('active');
}}

function toggleCaption(btn) {{
    const td = btn.parentElement;
    const short = td.querySelector('.caption-short');
    const full = td.querySelector('.caption-full');
    if (full.style.display === 'none') {{
        full.style.display = 'block';
        short.style.display = 'none';
        btn.textContent = '收起';
    }} else {{
        full.style.display = 'none';
        short.style.display = 'block';
        btn.textContent = '展開';
    }}
}}

function filterTable() {{
    const q = document.getElementById('search').value.toLowerCase();
    const rows = document.querySelectorAll('#mem-table .mem-row');
    rows.forEach(row => {{
        const caption = row.querySelector('.caption').textContent.toLowerCase();
        row.style.display = caption.includes(q) ? '' : 'none';
        if (q && caption.includes(q)) {{
            row.classList.add('highlight');
        }} else {{
            row.classList.remove('highlight');
        }}
    }});
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="ReMEmbR Memory Viewer")
    parser.add_argument("--db", default="./remembr_memory.db", help="Milvus Lite DB path")
    parser.add_argument("--collection", default="robot_memory", help="Collection name")
    parser.add_argument("--image-dir", default="./memory_images", help="Directory with saved images")
    parser.add_argument("--output", default="memory_report.html", help="Output HTML file")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server after generating")
    parser.add_argument("--port", type=int, default=8099, help="HTTP server port")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f"[ERROR] DB file not found: {db_path}")
        sys.exit(1)

    print(f"[1/3] Loading memories from {db_path}...")
    memories = load_memories(db_path, args.collection)
    print(f"       Loaded {len(memories)} memories")

    if not memories:
        print("[ERROR] No memories found in DB")
        sys.exit(1)

    image_dir = os.path.abspath(args.image_dir) if args.image_dir else ""
    img_count = sum(1 for m in memories if m.get("image_path"))
    print(f"       {img_count} memories have image paths, image_dir={image_dir}")

    print(f"[2/3] Generating position map...")
    map_b64 = generate_map_image(memories)

    print(f"[3/3] Generating HTML report...")
    html = generate_html(memories, db_path, map_b64, image_dir=image_dir)

    output_path = os.path.join(os.path.dirname(db_path), args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"       Report saved: {output_path}")

    if args.serve:
        import http.server
        import functools
        os.chdir(os.path.dirname(output_path))
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.dirname(output_path))
        print(f"\n  Open in browser: http://localhost:{args.port}/{args.output}")
        print(f"  Press Ctrl+C to stop\n")
        http.server.HTTPServer(("0.0.0.0", args.port), handler).serve_forever()
    else:
        print(f"\n  To view: open {output_path} in browser")
        print(f"  Or run: python3.10 memory_viewer.py --serve")


if __name__ == "__main__":
    main()
