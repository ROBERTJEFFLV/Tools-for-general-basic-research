#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV Trajectory Viewer (UI 版) — PyQt5 + QOpenGLWidget

你要的 UI 功能都在这里：
1) 左侧面板可导入/管理多条 CSV 轨迹：一次性多选、显隐开关、帧解释（AUTO/ENU/NED）、色带选择（turbo/jetlike）。
2) 左侧面板提供 Gazebo 风格的障碍物编辑：新增 box/sphere/cylinder，选中后可精确调节 x/y/z/size/yaw。
3) 3D 交互：左键旋转、滚轮缩放、中键或 Shift+左键平移；地面网格与坐标轴；按时间渐变上色。

依赖：
  pip install PyQt5 PyOpenGL numpy pandas

运行：
  python uav_trajectory_plotter.py

CSV 规则：
- 时间列：t、t_s、time、sec、seconds、timestamp、true_time（任一即可；数值或可解析时间戳）
- 坐标列：
  (x, y, z)  视为 ENU（z 向上）
  (n, e, d)  视为 NED（自动转 ENU：x=e, y=n, z=-d）
- 也支持 east/north/up。
"""

import os, sys, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd



from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QCheckBox, QComboBox,
    QDoubleSpinBox, QFileDialog, QGroupBox, QLineEdit, QDialog, QDialogButtonBox, 
    QFormLayout
)
from PyQt5.QtWidgets import QOpenGLWidget  # 改为从 QtWidgets 导入

from OpenGL.GL import *
from OpenGL.GLU import *

# ------------------------- 检测/转换工具 -------------------------
CANDIDATE_T = ["t", "t_s", "time", "sec", "secs", "seconds", "timestamp", "true_time"]
XYZ_SETS = [("x","y","z"),("pos_x","pos_y","pos_z"),("px","py","pz"),("east","north","up")]
NED_SETS = [("n","e","d"),("north","east","down")]


def _downsample_idx(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    step = max(1, n // max_points)
    return np.arange(0, n, step)


def _to_seconds(series: pd.Series, col_name: Optional[str] = None) -> np.ndarray:
    s = series
    if np.issubdtype(s.dtype, np.number):
        v = s.astype(float).values
        name = (col_name or "").lower()
        # 基于列名或数量级判断单位
        if "ns" in name or "nanosec" in name or np.nanmedian(v) > 1e12:
            return v * 1e-9
        if "ms" in name or "millis" in name or (1e3 < np.nanmedian(v) < 1e9):
            return v * 1e-3
        return v  # 视为秒
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        base = dt.iloc[0]
        return (dt - base).dt.total_seconds().values
    except Exception:
        return pd.to_numeric(s, errors="coerce").astype(float).values


def ned_to_enu(n: np.ndarray, e: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ENU: x=east, y=north, z=up
    return e, n, -d


def _detect_time_col(df: pd.DataFrame) -> str:
    low = [c.lower() for c in df.columns]
    for k in CANDIDATE_T:
        if k in low:
            return df.columns[low.index(k)]
    return df.columns[0]


def _detect_xyz_cols(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    low = [c.lower() for c in df.columns]
    for a,b,c in NED_SETS:
        if a in low and b in low and c in low:
            return "NED", df.columns[low.index(a)], df.columns[low.index(b)], df.columns[low.index(c)]
    for a,b,c in XYZ_SETS:
        if a in low and b in low and c in low:
            return "ENU", df.columns[low.index(a)], df.columns[low.index(b)], df.columns[low.index(c)]
    cols = df.columns[:3]
    return "ENU", cols[0], cols[1], cols[2]


def _colormap_time(t01: np.ndarray, scheme: str = "turbo") -> np.ndarray:
    t = np.clip(np.asarray(t01, dtype=np.float32), 0.0, 1.0)
    if scheme == "jetlike":
        r = np.clip(1.5 * t - 0.5, 0, 1)
        g = np.clip(1.5 - np.abs(2 * t - 1.0) * 1.5, 0, 1)
        b = np.clip(1.5 * (1.0 - t) - 0.5, 0, 1)
        return np.stack([r, g, b], axis=1)
    
    if scheme == "turbo":
        r = np.clip(1.5 * t + 0.2 - 0.5 * np.cos(2 * np.pi * t), 0, 1)
        g = np.clip(1.2 - (t - 0.5) ** 2 * 4.0, 0, 1)
        b = np.clip(1.2 * (1.0 - t) + 0.2 * np.cos(2 * np.pi * t), 0, 1)
        return np.stack([r, g, b], axis=1)


# ------------------------- 数据类 -------------------------
@dataclass
class Trajectory:
    path: str
    name: str
    frame_mode: str = "AUTO"  # AUTO / ENU / NED
    visible: bool = True
    color_scheme: str = "jetlike"

    # 新增：显式列映射（可选）
    time_col: Optional[str] = None
    xyz_cols: Optional[Tuple[str, str, str]] = None

    t: Optional[np.ndarray] = None
    xyz: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None

    def load(self):
        df = pd.read_csv(self.path)
        # 去掉无名索引列
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        if df.empty:
            raise RuntimeError("CSV 为空")
        # 1) 时间列
        if self.time_col and self.time_col in df.columns:
            t_col = self.time_col
        else:
            t_col = _detect_time_col(df)
        t_raw = _to_seconds(df[t_col], col_name=t_col)
        # 2) 坐标列
        if self.xyz_cols and all(c in df.columns for c in self.xyz_cols):
            cx, cy, cz = self.xyz_cols
            base_frame = "ENU"  # 明确选择的三元组默认按 ENU 解释
        else:
            base_frame, cx, cy, cz = _detect_xyz_cols(df)        
        
        x_raw = pd.to_numeric(df[cx], errors="coerce").astype(float).values
        y_raw = pd.to_numeric(df[cy], errors="coerce").astype(float).values
        z_raw = pd.to_numeric(df[cz], errors="coerce").astype(float).values
        mask = np.isfinite(t_raw) & np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(z_raw)
        t_raw, x_raw, y_raw, z_raw = t_raw[mask], x_raw[mask], y_raw[mask], z_raw[mask]
        if len(t_raw) < 2:
            raise RuntimeError("有效数据不足 2 个点")
        order = np.argsort(t_raw)
        t_raw = t_raw[order]
        x_raw, y_raw, z_raw = x_raw[order], y_raw[order], z_raw[order]
        # 帧解释
        use_frame = base_frame if self.frame_mode == "AUTO" else self.frame_mode
        if use_frame == "NED":
            if base_frame == "NED":
                x, y, z = ned_to_enu(x_raw, y_raw, z_raw)
            else:
                # 若用户强制按 NED 解释但列非 NED，做安全转换（交换 x/y，z 取反）
                x, y, z = y_raw, x_raw, -z_raw
        else:  # ENU
            if base_frame == "NED":
                x, y, z = ned_to_enu(x_raw, y_raw, z_raw)
            else:
                x, y, z = x_raw, y_raw, z_raw
        idx = _downsample_idx(len(t_raw), 120_000)
        self.t = t_raw[idx]
        self.xyz = np.stack([x[idx], y[idx], z[idx]], axis=1)
        t01 = (self.t - self.t.min()) / max(1e-12, (self.t.max() - self.t.min()))
        self.colors = _colormap_time(t01, scheme=self.color_scheme)

    def bbox(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.xyz is None or len(self.xyz) == 0:
            return None
        return self.xyz.min(0), self.xyz.max(0)


@dataclass
class Shape:
    kind: str  # 'box' | 'sphere' | 'cylinder'
    pos: np.ndarray
    size: float = 0.5
    yaw_deg: float = 0.0
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)


# ------------------------- OpenGL 视图 -------------------------
class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.trajectories: List[Trajectory] = []
        self.shapes: List[Shape] = []

        # 相机（轨道）
        self.target = np.array([0.0, 0.0, 0.0], dtype=float)
        self.distance = 8.0
        self.yaw = 45.0
        self.pitch = 25.0

        self._mouse_last = None
        self._left_down = False
        self._mid_down = False

    # ---- OpenGL 生命周期 ----
    def initializeGL(self):
        glClearColor(1, 1, 1, 1)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glLineWidth(2.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, max(1, w), max(1, h))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, max(1.0, w) / max(1.0, h), 0.05, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # 相机矩阵
        pr = math.radians(self.pitch)
        yr = math.radians(self.yaw)
        eye = self.target + np.array([
            self.distance * math.cos(pr) * math.cos(yr),
            self.distance * math.cos(pr) * math.sin(yr),
            self.distance * math.sin(pr)
        ])
        gluLookAt(eye[0], eye[1], eye[2], self.target[0], self.target[1], self.target[2], 0, 0, 1)

        # 地面网格与坐标轴
        self._draw_grid()
        self._draw_axes()

        # 轨迹
        for traj in self.trajectories:
            if not traj.visible or traj.xyz is None or traj.colors is None:
                continue
            glBegin(GL_LINE_STRIP)
            for (x, y, z), c in zip(traj.xyz, traj.colors):
                glColor3f(c[0], c[1], c[2])
                glVertex3f(x, y, z)
            glEnd()

        # 障碍物
        for shp in self.shapes:
            self._draw_shape(shp)

    # ---- 绘制帮助 ----
    def _draw_grid(self, extent=40, step=1.0):
        glColor3f(0.92, 0.92, 0.92)
        glBegin(GL_LINES)
        for i in range(-extent, extent + 1):
            x = i * step
            glVertex3f(x, -extent * step, 0)
            glVertex3f(x,  extent * step, 0)
            glVertex3f(-extent * step, x, 0)
            glVertex3f( extent * step, x, 0)
        glEnd()

    def _draw_axes(self, L=1.5):
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(L, 0, 0)  # X 红
        glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, L, 0)  # Y 绿
        glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, L)  # Z 蓝
        glEnd()



    def _draw_shape(self, shp: Shape):
        glPushMatrix()
        glTranslatef(*shp.pos)
        glRotatef(shp.yaw_deg, 0, 0, 1)
        glColor3f(*shp.color)
        if shp.kind == 'box':
            s = shp.size * 0.5
            # 实体面
            glBegin(GL_QUADS)
            # +Z
            glVertex3f(-s, -s,  s); glVertex3f( s, -s,  s); glVertex3f( s,  s,  s); glVertex3f(-s,  s,  s)
            # -Z
            glVertex3f(-s, -s, -s); glVertex3f(-s,  s, -s); glVertex3f( s,  s, -s); glVertex3f( s, -s, -s)
            # +X
            glVertex3f( s, -s, -s); glVertex3f( s,  s, -s); glVertex3f( s,  s,  s); glVertex3f( s, -s,  s)
            # -X
            glVertex3f(-s, -s, -s); glVertex3f(-s, -s,  s); glVertex3f(-s,  s,  s); glVertex3f(-s,  s, -s)
            # +Y
            glVertex3f(-s,  s, -s); glVertex3f(-s,  s,  s); glVertex3f( s,  s,  s); glVertex3f( s,  s, -s)
            # -Y
            glVertex3f(-s, -s, -s); glVertex3f( s, -s, -s); glVertex3f( s, -s,  s); glVertex3f(-s, -s,  s)
            glEnd()
            # 黑色线框描边
            glColor3f(0,0,0)
            glBegin(GL_LINES)
            for x in (-s,s):
                for y in (-s,s):
                    for z in (-s,s):
                        if x==s: glVertex3f(x,y,z); glVertex3f(-x,y,z)
                        if y==s: glVertex3f(x,y,z); glVertex3f(x,-y,z)
                        if z==s: glVertex3f(x,y,z); glVertex3f(x,y,-z)
            glEnd()
        elif shp.kind == 'sphere':
            q = gluNewQuadric(); gluSphere(q, shp.size * 0.5, 24, 18); gluDeleteQuadric(q)
            # 黑色线框
            glColor3f(0,0,0)
            glut_wire = gluNewQuadric(); gluQuadricDrawStyle(glut_wire, GLU_LINE)
            gluSphere(glut_wire, shp.size * 0.5, 12, 8)
            gluDeleteQuadric(glut_wire)
        elif shp.kind == 'cylinder':
            q = gluNewQuadric()
            r = shp.size * 0.3
            h = shp.size
            glRotatef(90, 1, 0, 0)
            gluCylinder(q, r, r, h, 24, 1)
            gluDisk(q, 0.0, r, 24, 1); glTranslatef(0, 0, h); gluDisk(q, 0.0, r, 24, 1)
            gluDeleteQuadric(q)
            # 黑色线框圆柱
            glColor3f(0,0,0)
            q2 = gluNewQuadric(); gluQuadricDrawStyle(q2, GLU_LINE)
            glRotatef(-90,1,0,0)
            gluCylinder(q2, r, r, h, 12, 1)
            gluDeleteQuadric(q2)
        glPopMatrix()



    # ---- 交互 ----
    def mousePressEvent(self, e):
        self._mouse_last = e.pos()
        if e.button() == QtCore.Qt.LeftButton:
            self._left_down = True
        elif e.button() == QtCore.Qt.MiddleButton:
            self._mid_down = True

    def mouseReleaseEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._left_down = False
        elif e.button() == QtCore.Qt.MiddleButton:
            self._mid_down = False

    def mouseMoveEvent(self, e):
        if self._mouse_last is None:
            self._mouse_last = e.pos(); return
        dx = e.x() - self._mouse_last.x()
        dy = e.y() - self._mouse_last.y()
        self._mouse_last = e.pos()
        if self._left_down and not (e.modifiers() & QtCore.Qt.ShiftModifier):
            # 旋转
            self.yaw   -= dx * 0.2
            self.pitch += dy * 0.2
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self.update()
        if self._mid_down or (self._left_down and (e.modifiers() & QtCore.Qt.ShiftModifier)):
            # 平移
            self._pan(-dx, dy)
            self.update()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        scale = 0.9 if delta > 0 else 1.1
        self.distance = float(np.clip(self.distance * scale, 0.3, 1e6))
        self.update()

    def _pan(self, dx_pix: float, dy_pix: float):
        w = max(1.0, self.width()); h = max(1.0, self.height())
        f = self.distance * math.tan(math.radians(45.0 / 2))
        sx = (dx_pix / w) * f * 2
        sy = (dy_pix / h) * f * 2
        yawr = math.radians(self.yaw)
        right = np.array([ math.sin(yawr), -math.cos(yawr), 0.0 ])
        up = np.array([0.0, 0.0, 1.0])
        self.target -= right * sx
        self.target += up * sy

    # 公共：根据所有元素自适应视图
    def fit_to_scene(self):
        mins, maxs = [], []
        for t in self.trajectories:
            bb = t.bbox();
            if bb is not None:
                a, b = bb; mins.append(a); maxs.append(b)
        for s in self.shapes:
            p = s.pos; r = max(0.5, s.size)
            mins.append(p - r); maxs.append(p + r)
        if not mins: return
        mins = np.min(np.stack(mins), axis=0)
        maxs = np.max(np.stack(maxs), axis=0)
        center = (mins + maxs) * 0.5
        size = np.linalg.norm(maxs - mins)
        if size < 1e-6: size = 1.0
        self.target = center
        self.distance = max(3.0, size * 0.8)
        self.update()

class ColumnMapDialog(QDialog):
    """
    读取 CSV 表头后，允许选择：
      - 时间列
      - 多个“轨迹映射”（Name, X, Y, Z）
    支持 Auto Detect：按 *.x/*.y/*.z 前缀分组。
    """
    def __init__(self, csv_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map Columns from CSV")
        self.resize(600, 360)
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, nrows=2)  # 看表头足够
        self.df = self.df.loc[:, ~self.df.columns.astype(str).str.startswith("Unnamed")]
        cols = [str(c) for c in self.df.columns]

        # 顶部：时间列
        top = QFormLayout()
        self.cmb_time = QComboBox()
        self.cmb_time.addItems(cols)
        # 预选：尽量挑 t_ns / t_s / time
        pref = [c for c in cols if c.lower() in ("t_ns","t_s","time","timestamp","seconds","sec","true_time")]
        if pref:
            self.cmb_time.setCurrentText(pref[0])
        top.addRow("Time column:", self.cmb_time)

        # 轨迹区（可增加多条）
        self.tracks_area = QVBoxLayout()
        # 控件缓存：[(name_edit, cmb_x, cmb_y, cmb_z), ...]
        self.track_widgets: List[Tuple[QLineEdit, QComboBox, QComboBox, QComboBox]] = []

        btns_row = QHBoxLayout()
        self.btn_add_track = QPushButton("Add Track")
        self.btn_auto = QPushButton("Auto Detect")
        btns_row.addWidget(self.btn_add_track)
        btns_row.addWidget(self.btn_auto)
        self.btn_add_track.clicked.connect(self._add_track_row)
        self.btn_auto.clicked.connect(self._auto_detect)

        # 先放两条（UAV1 / UAV2）
        self._add_track_row("UAV1")
        self._add_track_row("UAV2")

        # 底部按钮
        self.box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.box.accepted.connect(self.accept)
        self.box.rejected.connect(self.reject)

        # 布局
        root = QVBoxLayout(self)
        gb_top = QGroupBox("Time")
        gb_top.setLayout(top)
        gb_trk = QGroupBox("Tracks (each needs X/Y/Z)")
        lay_trk = QVBoxLayout(); lay_trk.addLayout(self.tracks_area); lay_trk.addLayout(btns_row)
        gb_trk.setLayout(lay_trk)
        root.addWidget(gb_top)
        root.addWidget(gb_trk)
        root.addWidget(self.box)

        # 预跑一次自动识别
        self._auto_detect()

    def _add_track_row(self, name: Optional[str] = None):
        cols = [str(c) for c in self.df.columns]
        line = QHBoxLayout()
        name_edit = QLineEdit(name or f"Track{len(self.track_widgets)+1}")
        cmb_x = QComboBox(); cmb_y = QComboBox(); cmb_z = QComboBox()
        for cmb in (cmb_x, cmb_y, cmb_z):
            cmb.setEditable(True)
            cmb.addItems(cols)
        line.addWidget(QLabel("Name:")); line.addWidget(name_edit, 1)
        line.addWidget(QLabel("X:")); line.addWidget(cmb_x, 1)
        line.addWidget(QLabel("Y:")); line.addWidget(cmb_y, 1)
        line.addWidget(QLabel("Z:")); line.addWidget(cmb_z, 1)
        self.tracks_area.addLayout(line)
        self.track_widgets.append((name_edit, cmb_x, cmb_y, cmb_z))

    def _auto_detect(self):
        """
        规则：把以 .x/.y/.z 或 /x,/y,/z 结尾的列，按去掉后缀的前缀进行分组。
        取前两组填入前两条轨迹。
        """
        cols = [str(c) for c in self.df.columns]
        groups = {}  # key -> {'x':col, 'y':col, 'z':col, 'name':suggest}
        def key_of(c: str):
            cl = c.lower()
            for suf in (".x",".y",".z","/x","/y","/z"):
                if cl.endswith(suf):
                    return c[: -len(suf)], suf[-1]
            return None, None

        for c in cols:
            k, axis = key_of(c)
            if k and axis in "xyz":
                g = groups.setdefault(k, {})
                g[axis] = c
                # 取名字建议：key 最后一个 path 片段
                parts = k.split("/")
                g.setdefault("name", parts[-1] if parts and parts[-1] else k)

        # 把完整 xyz 的组挑出来
        complete = [(k, g) for k, g in groups.items() if all(a in g for a in "xyz")]
        # 排序：稳定即可
        complete.sort(key=lambda kv: kv[0])

        for i, (_, g) in enumerate(complete[: len(self.track_widgets)]):
            name_edit, cx, cy, cz = self.track_widgets[i]
            name_edit.setText(str(g.get("name","Track")))
            cx.setCurrentText(g['x']); cy.setCurrentText(g['y']); cz.setCurrentText(g['z'])

        # 时间列也顺带优选 t_ns/t_s
        for pref in ("t_ns","t_s","time","timestamp"):
            for c in cols:
                if c.lower() == pref:
                    self.cmb_time.setCurrentText(c); return

    def get_result(self):
        time_col = self.cmb_time.currentText().strip()
        tracks = []
        for name_edit, cx, cy, cz in self.track_widgets:
            name = name_edit.text().strip()
            xs, ys, zs = cx.currentText().strip(), cy.currentText().strip(), cz.currentText().strip()
            if xs and ys and zs and all(c in self.df.columns for c in (xs,ys,zs)):
                tracks.append( (name or "Track", (xs,ys,zs)) )
        return time_col, tracks



# ------------------------- 主窗口/面板 -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Trajectory Viewer (UI)")
        self.resize(1280, 820)

        self.gl = GLWidget()

        # —— 左侧：轨迹 & 障碍物 控制面板 ——
        left = QWidget(); left_layout = QVBoxLayout(left)

        # 轨迹组
        traj_group = QGroupBox("Trajectories")
        traj_layout = QVBoxLayout(traj_group)
        self.traj_list = QListWidget()
        btns = QHBoxLayout()
        self.btn_add_csv = QPushButton("Add CSV(s)…")
        self.btn_add_mapped = QPushButton("Add + Map…")   # 新增
        self.btn_remove_csv = QPushButton("Remove")
        self.btn_fit = QPushButton("Fit View")
        btns.addWidget(self.btn_add_csv); btns.addWidget(self.btn_remove_csv); btns.addWidget(self.btn_fit); btns.addWidget(self.btn_add_mapped)                # 新增

        # 轨迹属性区
        prop = QGridLayout()
        self.chk_visible = QCheckBox("Visible")
        self.cmb_frame = QComboBox(); self.cmb_frame.addItems(["AUTO","ENU","NED"])
        self.cmb_cmap = QComboBox(); self.cmb_cmap.addItems(["turbo","jetlike"])
        prop.addWidget(QLabel("Visibility:"), 0,0); prop.addWidget(self.chk_visible, 0,1)
        prop.addWidget(QLabel("Frame:"),     1,0); prop.addWidget(self.cmb_frame, 1,1)
        prop.addWidget(QLabel("Colormap:"),  2,0); prop.addWidget(self.cmb_cmap, 2,1)

        traj_layout.addWidget(self.traj_list)
        traj_layout.addLayout(btns)
        traj_layout.addLayout(prop)

        # 障碍物组
        obs_group = QGroupBox("Obstacles")
        obs_layout = QVBoxLayout(obs_group)
        self.obs_list = QListWidget()
        row1 = QHBoxLayout()
        self.cmb_obs_kind = QComboBox(); self.cmb_obs_kind.addItems(["box","sphere","cylinder"])
        self.btn_add_obs = QPushButton("Add")
        self.btn_del_obs = QPushButton("Delete")
        row1.addWidget(QLabel("Type:")); row1.addWidget(self.cmb_obs_kind,1); row1.addWidget(self.btn_add_obs); row1.addWidget(self.btn_del_obs)
        grid = QGridLayout()
        self.sp_x = QDoubleSpinBox(); self.sp_y = QDoubleSpinBox(); self.sp_z = QDoubleSpinBox()
        for sp in (self.sp_x, self.sp_y, self.sp_z):
            sp.setRange(-1e6, 1e6); sp.setDecimals(3); sp.setSingleStep(0.1)
        self.sp_size = QDoubleSpinBox(); self.sp_size.setRange(0.01, 1e6); self.sp_size.setValue(0.5); self.sp_size.setDecimals(3)
        self.sp_yaw  = QDoubleSpinBox(); self.sp_yaw.setRange(-360, 360); self.sp_yaw.setDecimals(1)
        grid.addWidget(QLabel("x"),0,0); grid.addWidget(self.sp_x,0,1)
        grid.addWidget(QLabel("y"),1,0); grid.addWidget(self.sp_y,1,1)
        grid.addWidget(QLabel("z"),2,0); grid.addWidget(self.sp_z,2,1)
        grid.addWidget(QLabel("size"),3,0); grid.addWidget(self.sp_size,3,1)
        grid.addWidget(QLabel("yaw"),4,0); grid.addWidget(self.sp_yaw,4,1)
        obs_layout.addWidget(self.obs_list)
        obs_layout.addLayout(row1)
        obs_layout.addLayout(grid)

        left_layout.addWidget(traj_group)
        left_layout.addWidget(obs_group)
        left_layout.addStretch(1)

        # 主布局：左侧控制 + 右侧 GL 视图
        central = QWidget(); root = QHBoxLayout(central)
        root.addWidget(left, 0)
        root.addWidget(self.gl, 1)
        self.setCentralWidget(central)

        # —— 连接信号 ——
        self.btn_add_csv.clicked.connect(self.on_add_csv)
        self.btn_remove_csv.clicked.connect(self.on_remove_csv)
        self.btn_fit.clicked.connect(self.gl.fit_to_scene)
        self.traj_list.currentRowChanged.connect(self.on_select_traj)
        self.chk_visible.toggled.connect(self.on_traj_prop_changed)
        self.cmb_frame.currentTextChanged.connect(self.on_traj_prop_changed)
        self.cmb_cmap.currentTextChanged.connect(self.on_traj_prop_changed)

        self.btn_add_mapped.clicked.connect(self.on_add_mapped)


        self.btn_add_obs.clicked.connect(self.on_add_obs)
        self.btn_del_obs.clicked.connect(self.on_del_obs)
        self.obs_list.currentRowChanged.connect(self.on_select_obs)
        self.sp_x.valueChanged.connect(self.on_obs_changed)
        self.sp_y.valueChanged.connect(self.on_obs_changed)
        self.sp_z.valueChanged.connect(self.on_obs_changed)
        self.sp_size.valueChanged.connect(self.on_obs_changed)
        self.sp_yaw.valueChanged.connect(self.on_obs_changed)

        # 首次提示
        self.statusBar().showMessage("Add CSV(s) 导入轨迹；左键旋转，滚轮缩放，中键/Shift+左键平移。")

    # ---------------- 轨迹操作 ----------------
    def on_add_csv(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open CSV(s)", "", "CSV Files (*.csv)")
        if not paths:
            return
        for p in paths:
            try:
                traj = Trajectory(path=p, name=os.path.basename(p))
                traj.load()
                self.gl.trajectories.append(traj)
                item = QListWidgetItem(traj.name)
                item.setCheckState(QtCore.Qt.Checked)
                self.traj_list.addItem(item)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load error", f"{p}\n{e}")
        self.gl.fit_to_scene()
        self.gl.update()

    def on_remove_csv(self):
        row = self.traj_list.currentRow()
        if row < 0: return
        self.traj_list.takeItem(row)
        self.gl.trajectories.pop(row)
        self.gl.update()

    def on_add_mapped(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            dlg = ColumnMapDialog(path, self)
            if dlg.exec_() != QDialog.Accepted:
                return
            time_col, tracks = dlg.get_result()
            if not tracks:
                QtWidgets.QMessageBox.information(self, "Empty", "未选择任何三元组。")
                return
            created = 0
            for name, (cx,cy,cz) in tracks:
                traj = Trajectory(
                    path=path,
                    name=f"{os.path.basename(path)} [{name}]",
                    time_col=time_col,
                    xyz_cols=(cx,cy,cz)
                )
                traj.load()
                self.gl.trajectories.append(traj)
                item = QListWidgetItem(traj.name)
                item.setCheckState(QtCore.Qt.Checked)
                self.traj_list.addItem(item)
                created += 1
            if created:
                self.gl.fit_to_scene()
                self.gl.update()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load error", f"{path}\n{e}")


    def on_select_traj(self, row: int):
        if row < 0 or row >= len(self.gl.trajectories):
            self.chk_visible.setChecked(False)
            return
        t = self.gl.trajectories[row]
        self.chk_visible.setChecked(t.visible)
        self.cmb_frame.setCurrentText(t.frame_mode)
        self.cmb_cmap.setCurrentText(t.color_scheme)

    def on_traj_prop_changed(self):
        row = self.traj_list.currentRow()
        if row < 0: return
        t = self.gl.trajectories[row]
        t.visible = self.chk_visible.isChecked()
        frame_new = self.cmb_frame.currentText()
        cmap_new  = self.cmb_cmap.currentText()
        reload_needed = (frame_new != t.frame_mode) or (cmap_new != t.color_scheme)
        t.frame_mode = frame_new
        t.color_scheme = cmap_new
        if reload_needed:
            try:
                t.load()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Reload error", str(e))
        self.gl.update()

    # ---------------- 障碍物操作 ----------------
    def on_add_obs(self):
        kind = self.cmb_obs_kind.currentText()
        shp = Shape(kind=kind, pos=np.array([0.0, 0.0, 0.0], dtype=float), size=0.5, yaw_deg=0.0)
        self.gl.shapes.append(shp)
        self.obs_list.addItem(QListWidgetItem(f"{kind}#{len(self.gl.shapes)}"))
        self.obs_list.setCurrentRow(self.obs_list.count()-1)
        self.gl.update()

    def on_del_obs(self):
        row = self.obs_list.currentRow()
        if row < 0: return
        self.obs_list.takeItem(row)
        self.gl.shapes.pop(row)
        if self.obs_list.count():
            self.obs_list.setCurrentRow(min(row, self.obs_list.count()-1))
        self.gl.update()

    def on_select_obs(self, row: int):
        if row < 0 or row >= len(self.gl.shapes):
            for sp in (self.sp_x, self.sp_y, self.sp_z, self.sp_size, self.sp_yaw):
                sp.blockSignals(True); sp.setValue(0.0); sp.blockSignals(False)
            return
        s = self.gl.shapes[row]
        self.sp_x.blockSignals(True); self.sp_y.blockSignals(True); self.sp_z.blockSignals(True)
        self.sp_size.blockSignals(True); self.sp_yaw.blockSignals(True)
        self.sp_x.setValue(float(s.pos[0])); self.sp_y.setValue(float(s.pos[1])); self.sp_z.setValue(float(s.pos[2]))
        self.sp_size.setValue(float(s.size)); self.sp_yaw.setValue(float(s.yaw_deg))
        self.sp_x.blockSignals(False); self.sp_y.blockSignals(False); self.sp_z.blockSignals(False)
        self.sp_size.blockSignals(False); self.sp_yaw.blockSignals(False)

    def on_obs_changed(self):
        row = self.obs_list.currentRow()
        if row < 0: return
        s = self.gl.shapes[row]
        s.pos[0] = self.sp_x.value(); s.pos[1] = self.sp_y.value(); s.pos[2] = self.sp_z.value()
        s.size   = self.sp_size.value()
        s.yaw_deg= self.sp_yaw.value()
        self.gl.update()


# ------------------------- 入口 -------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
