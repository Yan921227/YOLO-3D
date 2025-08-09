#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO‑3D 停車格偵測（70 % 覆蓋率判定版）
  • 手動標定單應、劃車格
  • 逐幀推論：YOLOv11 + Depth Anything → 3D BOX
  • 主畫面/俯視圖同步顯示、判斷佔用
  • ✅ 進格條件：
        1. 車輛與車格有交集
        2. 交集面積 / 車輛外框面積 ≥ ratio (預設 0.70)
        3. 車輛四個角落 ≥ 3 個在車格內（角落壓線即視為在外）
      → 同時滿足 1+2+3 時，車格顯示綠框 (OK)
  • 執行結果可輸出 MP4
"""
import argparse
import cv2
import numpy as np
from shapely.geometry import Polygon, Point     # 〈〈新增〉〉

# -------- 2D / Depth / 3D 工具 --------
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView

# -------- CLI 參數 --------
parser = argparse.ArgumentParser()
parser.add_argument('--video',      type=str, default='C:\\Users\\User\\Desktop\\car.mp4')
parser.add_argument('--out_video',  type=str, default='result2.mp4',
                    help='留空字串 ("") 則不輸出影片')
parser.add_argument('--yolo_size',  type=str, default='small',
                    choices=['nano', 'small', 'medium', 'large', 'extra'])
parser.add_argument('--depth_size', type=str, default='small',
                    choices=['small', 'base', 'large'])
parser.add_argument('--device',     type=str, default='cuda:0')
parser.add_argument('--conf',       type=float, default=0.4)
# 〈〈新增〉〉—— 以「面積覆蓋率」取代像素距離
parser.add_argument('--ratio',      type=float, default=0.7,
                    help='交集面積 / 車輛外框面積，達此比例才算停好 (0~1)')
parser.add_argument('--debug',      action='store_true', help='顯示除錯文字')
args = parser.parse_args()

# -------- 開啟影片 & 讀取首影格 --------
cap = cv2.VideoCapture(args.video)
ret, frame = cap.read()
if not ret:
    raise IOError(f'無法開啟影片 {args.video}')

H, H_inv          = None, None
parking_polys     = []   # Shapely Polygon (原圖座標，用於幾何判斷)
parking_draw_pts  = []   # np.int32 (N,1,2) (繪圖)

# ========== 1. 手動標定單應 (點地面 4 點) ==========
src_points = []
def click_homography(evt, x, y, *_):
    if evt == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        src_points.append([x, y])
        cv2.circle(tmp, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Select Plane', tmp)

tmp = frame.copy()
cv2.imshow('Select Plane', tmp)
cv2.setMouseCallback('Select Plane', click_homography)
print('請點擊地面矩形 4 角 (順序不限)…')
while len(src_points) < 4:
    cv2.waitKey(1)
cv2.destroyWindow('Select Plane')

w_bev, h_bev = 800, 600
dst_points   = np.float32([[0, 0], [w_bev, 0], [w_bev, h_bev], [0, h_bev]])
H, _  = cv2.findHomography(np.float32(src_points), dst_points)
H_inv = np.linalg.inv(H)

# ========== 2. 手動繪製停車格 (俯視圖) ==========
warp        = cv2.warpPerspective(frame, H, (w_bev, h_bev))
warp_show   = warp.copy()
frame_preview = frame.copy()
current_pts  = []

def click_slot(evt, x, y, *_):
    if evt != cv2.EVENT_LBUTTONDOWN:
        return
    current_pts.append((x, y))
    cv2.circle(warp_show, (x, y), 4, (255, 0, 0), -1)

    if len(current_pts) == 4:
        cv2.polylines(warp_show,
                      [np.array(current_pts, dtype=np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 0), 2)

        # 回到原圖座標
        pts = np.float32(current_pts).reshape(-1, 1, 2)
        inv = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)

        inv[:, 0] = np.clip(inv[:, 0], 0, frame.shape[1] - 1)
        inv[:, 1] = np.clip(inv[:, 1], 0, frame.shape[0] - 1)

        poly  = Polygon(inv)
        parking_polys.append(poly)
        draw  = np.array(inv, dtype=np.int32).reshape(-1, 1, 2)
        parking_draw_pts.append(draw)

        cv2.polylines(frame_preview, [draw], True, (0, 0, 255), 3)
        cv2.imshow('Frame Preview', frame_preview)
        print(f"Saved slot #{len(parking_polys)}")

        current_pts.clear()
    cv2.imshow('Warp', warp_show)

cv2.imshow('Warp', warp_show)
cv2.setMouseCallback('Warp', click_slot)
print('在俯視圖依序點四個角形成一格，ESC 結束…')
while True:
    if cv2.waitKey(1) == 27:
        break
cv2.destroyWindow('Warp')
cv2.destroyWindow('Frame Preview')

if len(parking_polys) == 0:
    print('⚠️ 沒有定義任何車格，程式將只顯示 3D 偵測結果。')

# ========== 3. 影片輸出設定 ==========
writer = None
if args.out_video.strip():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    h_out, w_out = frame.shape[:2]
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w_out, h_out))
    print(f'🔴  將輸出結果至 {args.out_video} (size={w_out}×{h_out}, fps={fps:.1f})')
else:
    print('⚪  不輸出影片 (--out_video="")')

# ========== 4. 載入模型 ==========
print('載入 2D、深度與 3D 模型…')
detector  = ObjectDetector(model_size=args.yolo_size, conf_thres=args.conf, device=args.device)
depthnet  = DepthEstimator(model_size=args.depth_size,    device=args.device)
estimator = BBox3DEstimator()
bev       = BirdEyeView(size=(w_bev, h_bev), scale=50, camera_height=1.2)

# ========== 5. 逐幀推論 ==========
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated, dets = detector.detect(frame, track=True)
    depth_map = depthnet.estimate_depth(frame)
    bev.reset()

    # slot_state：0=空、1=佔用但未停好、2=停好
    slot_state = [0] * len(parking_polys)

    # ---- 處理各偵測 ----
    for bbox2d, score, cls_id, obj_id in dets:
        if cls_id != 2:         # 只看 car
            continue
        x1, y1, x2, y2 = map(int, bbox2d)
        depth_val      = float(depth_map[(y1 + y2) // 2, (x1 + x2) // 2])

        # === 3D BOX + BEV ===
        box3d = estimator.estimate_3d_box(
                    [x1, y1, x2, y2], depth_val,
                    detector.get_class_names()[cls_id], object_id=obj_id)
        annotated = estimator.draw_box_3d(annotated, box3d)
        bev.draw_box(box3d)

        rect_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        rect_area = rect_poly.area
        rect_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # === 與每個車格比較 ===
        for idx, poly in enumerate(parking_polys):
            if not rect_poly.intersects(poly):
                continue

            inter_area = rect_poly.intersection(poly).area
            overlap_ratio = inter_area / rect_area

            inside_cnt = sum(poly.contains(Point(pt)) for pt in rect_corners)

            # 面積覆蓋率 + 角落判定
            if overlap_ratio >= args.ratio and inside_cnt >= 3:
                slot_state[idx] = 2                    # OK
            elif slot_state[idx] < 2:
                slot_state[idx] = 1                    # BAD（已佔用但未達標）

    # ---- 繪製停車格 ----
    for idx, pts in enumerate(parking_draw_pts):
        state = slot_state[idx]
        if state == 2:
            color, label = (0, 255, 0), f'Slot {idx+1} OK'
        elif state == 1:
            color, label = (0, 0, 255), f'Slot {idx+1} BAD'
        else:
            color, label = (0, 0, 255), f'Slot {idx+1} EMPTY'

        cv2.polylines(annotated, [pts], True, color, 4)
        cv2.putText(annotated, label, tuple(pts[0][0] - np.array([0, 5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ---- Debug 文字 (可選) ----
    if args.debug:
        y_dbg = 25
        for idx, s in enumerate(slot_state):
            cv2.putText(annotated, f'[dbg] slot{idx+1}: {["empty","bad","ok"][s]}',
                        (5, y_dbg), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            y_dbg += 18

    # ---- 影片寫檔 ----
    if writer is not None:
        writer.write(annotated)

    # ---- 顯示 ----
    cv2.imshow('Result-3D', annotated)
    cv2.imshow('BirdEye', bev.get_image())
    if cv2.waitKey(1) == 27:      # ESC 結束
        break

# ========== 6. 收尾 ==========
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
print('✅  影片處理完成。')
