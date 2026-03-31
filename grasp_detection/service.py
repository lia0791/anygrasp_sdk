"""
AnyGrasp HTTP Service
接收 RGB 图、深度图、相机内参（可选分割 mask），返回抓取位姿列表。

启动方式:
    cd grasp_detection
    LD_LIBRARY_PATH=$HOME/lib/openssl11 python service.py \
        --checkpoint_path log/checkpoint_detection.tar \
        --port 8080

请求示例 (Python client):
    import requests, base64, json
    import numpy as np
    from PIL import Image

    def encode(arr: np.ndarray, is_float=False) -> str:
        if is_float:
            return base64.b64encode(arr.astype(np.float32).tobytes()).decode()
        return base64.b64encode(arr.astype(np.uint16).tobytes()).decode()

    color = np.array(Image.open("example_data/color.png"))       # H×W×3 uint8
    depth = np.array(Image.open("example_data/depth.png"))       # H×W   uint16

    resp = requests.post("http://localhost:8080/grasp", json={
        "color":      base64.b64encode(color.tobytes()).decode(),
        "color_dtype": "uint8",
        "color_shape": list(color.shape),
        "depth":      base64.b64encode(depth.tobytes()).decode(),
        "depth_shape": list(depth.shape),
        "intrinsics": {"fx": 927.17, "fy": 927.37, "cx": 651.32, "cy": 349.62},
        "depth_scale": 1000.0,
        "workspace":  [-0.19, 0.12, 0.02, 0.15, 0.0, 1.0],   # xmin,xmax,ymin,ymax,zmin,zmax
        "top_k": 20,
        # "seg_mask": base64...,  # 可选，H×W bool/uint8，只返回 mask 内物体的抓取
    })
    grasps = resp.json()["grasps"]   # list of {translation, rotation, score, width}
"""

import argparse
import base64
import json
import os
import sys
import traceback

import numpy as np
import open3d as o3d
import torch
from http.server import BaseHTTPRequestHandler, HTTPServer

from gsnet import AnyGrasp

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--max_gripper_width", type=float, default=0.1)
parser.add_argument("--gripper_height", type=float, default=0.03)
parser.add_argument("--top_down_grasp", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8080)
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print("Loading AnyGrasp model...")
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()
print("Model loaded.")


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def run_inference(payload: dict) -> dict:
    # --- decode color ---
    color_bytes = base64.b64decode(payload["color"])
    color_shape = tuple(payload["color_shape"])           # (H, W, 3)
    color_dtype = payload.get("color_dtype", "uint8")
    color = np.frombuffer(color_bytes, dtype=color_dtype).reshape(color_shape)
    colors = color.astype(np.float32) / 255.0             # normalise to [0,1]

    # --- decode depth ---
    depth_bytes = base64.b64decode(payload["depth"])
    depth_shape = tuple(payload["depth_shape"])           # (H, W)
    depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(depth_shape)

    # --- intrinsics & scale ---
    intr = payload["intrinsics"]
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    scale = payload.get("depth_scale", 1000.0)

    # --- workspace limits [xmin,xmax,ymin,ymax,zmin,zmax] ---
    lims = payload.get("workspace", None)

    # --- top_k ---
    top_k = int(payload.get("top_k", 50))

    # --- build point cloud ---
    xmap, ymap = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    points_z = depth.astype(np.float32) / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # [DEBUG] depth stats — use this to verify depth_scale is correct
    print(f"[DEBUG] depth raw   min={depth.min()} max={depth.max()} mean={depth.mean():.1f} dtype={depth.dtype}")
    print(f"[DEBUG] depth_scale={scale}")
    print(f"[DEBUG] points_z    min={points_z.min():.4f} max={points_z.max():.4f} mean={points_z[points_z>0].mean():.4f} (unit: m)")
    print(f"[DEBUG] image shape H={depth_shape[0]} W={depth_shape[1]}, total pixels={depth_shape[0]*depth_shape[1]}")

    # depth validity mask — filter sentinel (65535) and z range
    z_max = payload.get("z_max", 2.0)   # default 2m, tune to your scene
    valid = (depth < 65000) & (points_z > 0.1) & (points_z < z_max)
    print(f"[DEBUG] valid pixels after z filter (z_max={z_max}m): {valid.sum()} / {valid.size} ({100*valid.mean():.1f}%)")

    # optional segmentation mask (H×W, uint8 or bool)
    if "seg_mask" in payload:
        mask_bytes = base64.b64decode(payload["seg_mask"])
        mask_shape = tuple(payload.get("seg_mask_shape", depth_shape))
        seg_mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_shape).astype(bool)
        valid = valid & seg_mask
        print(f"[DEBUG] valid pixels after seg_mask: {valid.sum()}")

    points = np.stack([points_x, points_y, points_z], axis=-1)[valid].astype(np.float32)
    colors_filtered = colors[valid].astype(np.float32)

    # downsample to avoid OOM — voxel grid then random cap
    voxel_size = payload.get("voxel_size", 0.005)  # 5mm voxels
    if voxel_size > 0 and len(points) > 0:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
        pcd = pcd.voxel_down_sample(voxel_size)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors_filtered = np.asarray(pcd.colors, dtype=np.float32)

    max_points = payload.get("max_points", 30000)
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors_filtered = colors_filtered[idx]

    print(f"[DEBUG] point cloud size: {len(points)} points")
    if len(points) > 0:
        print(f"[DEBUG] points xyz range: "
              f"x=[{points[:,0].min():.3f}, {points[:,0].max():.3f}] "
              f"y=[{points[:,1].min():.3f}, {points[:,1].max():.3f}] "
              f"z=[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    print(f"[DEBUG] workspace lims: {lims}")

    if len(points) == 0:
        return {"grasps": [], "error": "empty point cloud after filtering"}

    # free fragmented GPU memory before inference
    torch.cuda.empty_cache()

    # --- inference ---
    gg, _ = anygrasp.get_grasp(
        points,
        colors_filtered,
        lims=lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )

    if gg is None or len(gg) == 0:
        return {"grasps": []}

    gg = gg.nms().sort_by_score()
    gg_pick = gg[:top_k]

    results = []
    for g in gg_pick:
        results.append({
            "translation": g.translation.tolist(),   # [x, y, z]  metres
            "rotation":    g.rotation_matrix.tolist(), # 3×3
            "score":       float(g.score),
            "width":       float(g.width),
        })

    # --- save debug image if requested ---
    if payload.get("save_debug", False):
        _save_debug_image(
            color=color,
            results=results,
            fx=fx, fy=fy, cx=cx, cy=cy,
            save_path=payload.get("debug_path", "/tmp/anygrasp_debug.png"),
        )

    return {"grasps": results}


def _save_debug_image(color, results, fx, fy, cx, cy, save_path):
    """Project grasp centers onto RGB image and save as PNG."""
    import cv2
    img = color.copy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for i, g in enumerate(results):
        x, y, z = g["translation"]
        if z <= 0:
            continue
        # project 3D → 2D
        u = int(x / z * fx + cx)
        v = int(y / z * fy + cy)
        if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
            continue

        score = g["score"]
        # color: green=high score, red=low score
        r = int((1 - score) * 255)
        g_c = int(score * 255)
        color_bgr = (0, g_c, r)

        cv2.circle(img, (u, v), 6, color_bgr, -1)
        cv2.circle(img, (u, v), 8, (255, 255, 255), 1)
        label = f"#{i} {score:.2f}"
        cv2.putText(img, label, (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # draw best grasp larger
    if results:
        x, y, z = results[0]["translation"]
        if z > 0:
            u = int(x / z * fx + cx)
            v = int(y / z * fy + cy)
            cv2.circle(img, (u, v), 12, (0, 255, 255), 2)

    cv2.imwrite(save_path, img)
    print(f"[DEBUG] saved debug image → {save_path}")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class GraspHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # suppress default per-request log; print our own
        pass

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/grasp":
            self._respond(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body)
        except Exception as e:
            self._respond(400, {"error": f"invalid JSON: {e}"})
            return

        try:
            result = run_inference(payload)
            print(f"[/grasp] returned {len(result.get('grasps', []))} grasps")
            self._respond(200, result)
        except Exception:
            err = traceback.format_exc()
            print(err, file=sys.stderr)
            self._respond(500, {"error": err})

    def _respond(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    server = HTTPServer((cfgs.host, cfgs.port), GraspHandler)
    print(f"AnyGrasp service listening on {cfgs.host}:{cfgs.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
