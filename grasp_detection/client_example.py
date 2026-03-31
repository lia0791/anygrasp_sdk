"""
AnyGrasp Service 客户端示例
在你的 Agent 里直接 import AnyGraspClient 使用。
"""

import base64
import json

import numpy as np
import requests
from PIL import Image


class AnyGraspClient:
    def __init__(self, host="localhost", port=8080):
        self.url = f"http://{host}:{port}/grasp"

    def detect(
        self,
        color: np.ndarray,          # H×W×3, uint8
        depth: np.ndarray,          # H×W,   uint16 (单位: mm)
        intrinsics: dict,           # {"fx","fy","cx","cy"}
        depth_scale: float = 1000.0,
        workspace=None,             # [xmin,xmax,ymin,ymax,zmin,zmax]，None=不过滤
        seg_mask: np.ndarray = None,# H×W, uint8/bool，None=全场景
        top_k: int = 50,
        save_debug: bool = False,   # 保存抓取点投影图
        debug_path: str = "/tmp/anygrasp_debug.png",
        extra_payload: dict | None = None, # 透传额外参数给服务端，如 voxel_size/max_points
    ) -> list[dict]:
        """
        返回值: list of {
            "translation": [x, y, z],    # 抓取点，相机坐标系，单位 m
            "rotation":    [[...3x3...]],  # 旋转矩阵
            "score":       float,
            "width":       float,          # 夹爪宽度，单位 m
        }
        按 score 降序排列，最多 top_k 个。
        """
        payload = {
            "color":       base64.b64encode(color.tobytes()).decode(),
            "color_dtype": str(color.dtype),
            "color_shape": list(color.shape),
            "depth":       base64.b64encode(depth.astype(np.uint16).tobytes()).decode(),
            "depth_shape": list(depth.shape),
            "intrinsics":  intrinsics,
            "depth_scale": depth_scale,
            "top_k":       top_k,
        }
        if workspace is not None:
            payload["workspace"] = workspace
        if save_debug:
            payload["save_debug"] = True
            payload["debug_path"] = debug_path
        if seg_mask is not None:
            mask = seg_mask.astype(np.uint8)
            payload["seg_mask"]       = base64.b64encode(mask.tobytes()).decode()
            payload["seg_mask_shape"] = list(mask.shape)
        if extra_payload:
            payload.update(extra_payload)

        resp = requests.post(self.url, json=payload, timeout=30,
                             proxies={"http": "", "https": ""})
        resp.raise_for_status()
        return resp.json()["grasps"]

    def health(self) -> bool:
        try:
            r = requests.get(self.url.replace("/grasp", "/health"), timeout=5,
                             proxies={"http": "", "https": ""})
            return r.json().get("status") == "ok"
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 快速测试（直接运行此文件）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = AnyGraspClient()

    if not client.health():
        print("Service not available. Start it first:\n"
              "  LD_LIBRARY_PATH=$HOME/lib/openssl11 python service.py "
              "--checkpoint_path log/checkpoint_detection.tar")
        exit(1)

    color = np.array(Image.open("example_data/color.png"))
    depth = np.array(Image.open("example_data/depth.png"))

    grasps = client.detect(
        color=color,
        depth=depth,
        intrinsics={"fx": 927.17, "fy": 927.37, "cx": 651.32, "cy": 349.62},
        workspace=[-0.19, 0.12, 0.02, 0.15, 0.0, 1.0],
        top_k=5,
    )

    print(f"Got {len(grasps)} grasps:")
    for i, g in enumerate(grasps):
        print(f"  [{i}] score={g['score']:.3f}  "
              f"xyz={[round(v,4) for v in g['translation']]}  "
              f"width={g['width']:.3f}m")
