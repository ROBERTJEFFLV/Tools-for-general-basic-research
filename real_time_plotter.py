#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Realtime Plotter (with curve dragging + robust legend name updates)
Fix: replace PlotDataItem.setName(...) (not available in some pyqtgraph versions)
with a portable helper that updates the legend label correctly.
"""
import sys, os, json
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

CANDIDATE_X = ["t_s", "time", "true_time", "timestamp", "sec", "seconds"]
CANDIDATE_Y = ["mic_distance", "distance", "mic_distance_test", "value", "y"]


def detect_x_col(df: pd.DataFrame):
    for k in CANDIDATE_X:
        if k in df.columns:
            return k
    return df.columns[0]


def detect_y_col(df: pd.DataFrame):
    for k in CANDIDATE_Y:
        if k in df.columns:
            return k
    return df.columns[1] if len(df.columns) >= 2 else df.columns[0]


def to_seconds(series: pd.Series):
    if np.issubdtype(series.dtype, np.number):
        return series.astype(float).values
    try:
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        base = dt.iloc[0]
        sec = (dt - base).dt.total_seconds().values
        return sec
    except Exception:
        return pd.to_numeric(series, errors="coerce").astype(float).values


def downsample_xy(x, y, max_points=200_000):
    n = len(x)
    if n <= max_points:
        return x, y
    step = max(1, n // max_points)
    return x[::step], y[::step]


class SeriesItem(QtWidgets.QWidget):
    data_changed = QtCore.pyqtSignal()

    def __init__(self, path, parent=None, line_color=None):
        super().__init__(parent)
        self.path = path
        self.last_mtime = None
        self.df = None
        self.line_color = line_color or pg.intColor(np.random.randint(0, 256))
        self.dx = 0.0
        self.dy = 0.0

        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = QtWidgets.QComboBox()
        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.name_edit = QtWidgets.QLineEdit(os.path.basename(path))
        self.name_edit.setPlaceholderText("Legend name")
        self.lock_chk = QtWidgets.QCheckBox("Lock")
        # NEW: Δx / Δy 数值框
        self.dx_spin = QtWidgets.QDoubleSpinBox()
        self.dx_spin.setRange(-1e12, 1e12)
        self.dx_spin.setDecimals(6)
        self.dx_spin.setSingleStep(0.001)
        self.dx_spin.setSuffix(" s")
        self.dx_spin.setValue(0.0)

        self.dy_spin = QtWidgets.QDoubleSpinBox()
        self.dy_spin.setRange(-1e12, 1e12)
        self.dy_spin.setDecimals(6)
        self.dy_spin.setSingleStep(0.001)
        self.dy_spin.setValue(0.0)

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QtWidgets.QLabel("File:"))
        h.addWidget(self.name_edit, 1)
        h.addWidget(QtWidgets.QLabel("X:"))
        h.addWidget(self.x_combo)
        h.addWidget(QtWidgets.QLabel("Y:"))
        h.addWidget(self.y_combo)

        # NEW: 加入 Δx/Δy 控件
        h.addWidget(QtWidgets.QLabel("Δx:"))
        h.addWidget(self.dx_spin)
        h.addWidget(QtWidgets.QLabel("Δy:"))
        h.addWidget(self.dy_spin)


        h.addWidget(self.lock_chk)
        h.addWidget(self.reload_btn)
        h.addWidget(self.remove_btn)

        self.reload_btn.clicked.connect(self.load_file)
        self.x_combo.currentIndexChanged.connect(self.data_changed.emit)
        self.y_combo.currentIndexChanged.connect(self.data_changed.emit)
        self.name_edit.textChanged.connect(self.data_changed.emit)

        
        # NEW: 监听 Δx/Δy 输入改变
        self.dx_spin.valueChanged.connect(self._on_spin_changed)
        self.dy_spin.valueChanged.connect(self._on_spin_changed)
        self.lock_chk.toggled.connect(self._on_lock_toggled)


        self.load_file()





    def load_file(self):
        try:
            df = pd.read_csv(self.path)
            df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
            if df.empty:
                raise RuntimeError("CSV is empty")
            self.df = df
            self.last_mtime = os.path.getmtime(self.path)

            self.x_combo.blockSignals(True); self.y_combo.blockSignals(True)
            self.x_combo.clear(); self.y_combo.clear()
            cols = list(map(str, df.columns))
            self.x_combo.addItems(cols)
            self.y_combo.addItems(cols)
            self.x_combo.setCurrentText(detect_x_col(df))
            self.y_combo.setCurrentText(detect_y_col(df))
            self.x_combo.blockSignals(False); self.y_combo.blockSignals(False)

            self.data_changed.emit()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load error", f"{self.path}\n{e}")

    def maybe_reload(self):
        try:
            mtime = os.path.getmtime(self.path)
            if self.last_mtime is None or mtime - self.last_mtime > 1e-6:
                self.load_file()
                return True
        except Exception:
            pass
        return False

    def legend_name(self):
        base = self.name_edit.text().strip() or os.path.basename(self.path)
        ycol = self.y_combo.currentText() if self.y_combo.count() else ""
        return f"{base} [Y: {ycol}]"

    def set_offsets(self, dx: float, dy: float):
        self.dx = float(dx)
        self.dy = float(dy)
        # 同步到 spinbox（阻断信号避免递归）
        self.dx_spin.blockSignals(True)
        self.dy_spin.blockSignals(True)
        try:
            self.dx_spin.setValue(self.dx)
            self.dy_spin.setValue(self.dy)
        finally:
            self.dx_spin.blockSignals(False)
            self.dy_spin.blockSignals(False)
        self.data_changed.emit()


    def get_offsets(self):
        return self.dx, self.dy

    def get_xy(self):
        if self.df is None:
            return None, None
        xcol = self.x_combo.currentText()
        ycol = self.y_combo.currentText()
        if xcol not in self.df.columns or ycol not in self.df.columns:
            return None, None
        x = to_seconds(self.df[xcol])
        y = pd.to_numeric(self.df[ycol], errors="coerce").astype(float).values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        order = np.argsort(x)
        x, y = x[order], y[order]
        x, y = downsample_xy(x, y, max_points=200_000)
        return x, y

    def get_xy_shifted(self):
        x, y = self.get_xy()
        if x is None or y is None:
            return None, None
        return x + self.dx, y + self.dy
    
    def _on_spin_changed(self, *_):
        # 手动输入驱动位移
        self.set_offsets(self.dx_spin.value(), self.dy_spin.value())

    def _on_lock_toggled(self, locked: bool):
        # 仅锁定“拖动”；数值框仍允许手动改（你若想锁住输入，可在此禁用 spin）
        # self.dx_spin.setEnabled(not locked)
        # self.dy_spin.setEnabled(not locked)
        self.data_changed.emit()  # 触发重绘（不强制）


class DragViewBox(pg.ViewBox):
    def __init__(self, window_ref=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_ref = window_ref

    def mouseDragEvent(self, ev, axis=None):
        win = self.window_ref
        if win and win.drag_mode_chk.isChecked() and (ev.button() == QtCore.Qt.LeftButton):
            pos_v = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                win._drag_begin(pos_v)
                ev.accept(); return
            elif ev.isFinish():
                win._drag_update(pos_v)
                win._drag_end()
                ev.accept(); return
            else:
                win._drag_update(pos_v)
                ev.accept(); return
        super().mouseDragEvent(ev, axis=axis)


class CSVPlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Realtime Plotter (pan/zoom/drag)")
        self.resize(1200, 760)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central); v.setContentsMargins(8, 8, 8, 8)
        self.setCentralWidget(central)

        ctrl = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add CSV…")
        self.autoscale_btn = QtWidgets.QPushButton("Autoscale")
        self.clear_btn = QtWidgets.QPushButton("Clear All")
        self.refresh_spin = QtWidgets.QDoubleSpinBox(); self.refresh_spin.setRange(0.2, 10.0)
        self.refresh_spin.setSingleStep(0.2); self.refresh_spin.setValue(1.0)
        self.refresh_spin.setSuffix(" s refresh")
        self.legend_chk = QtWidgets.QCheckBox("Show Legend"); self.legend_chk.setChecked(True)

        self.drag_mode_chk = QtWidgets.QCheckBox("Drag mode")
        self.drag_x_chk = QtWidgets.QCheckBox("X"); self.drag_x_chk.setChecked(True)
        self.drag_y_chk = QtWidgets.QCheckBox("Y"); self.drag_y_chk.setChecked(True)
        self.reset_sel_btn = QtWidgets.QPushButton("Reset Selected Offset")
        self.copy_offsets_btn = QtWidgets.QPushButton("Copy Offsets")
        self.selected_lbl = QtWidgets.QLabel("Selected: <none>")

        ctrl.addWidget(self.add_btn); ctrl.addWidget(self.autoscale_btn); ctrl.addWidget(self.clear_btn)
        ctrl.addStretch(1)
        ctrl.addWidget(QtWidgets.QLabel("Auto-reload:")); ctrl.addWidget(self.refresh_spin)
        ctrl.addSpacing(10); ctrl.addWidget(self.legend_chk)
        ctrl.addSpacing(15); ctrl.addWidget(self.drag_mode_chk)
        ctrl.addWidget(self.drag_x_chk); ctrl.addWidget(self.drag_y_chk)
        ctrl.addWidget(self.reset_sel_btn); ctrl.addWidget(self.copy_offsets_btn)
        ctrl.addWidget(self.selected_lbl)
        v.addLayout(ctrl)

        self.series_panel = QtWidgets.QVBoxLayout(); v.addLayout(self.series_panel)

        pg.setConfigOptions(antialias=True)
        self.vb = DragViewBox(window_ref=self)
        self.plot = pg.PlotWidget(background='w', viewBox=self.vb)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Distance', units='')
        v.addWidget(self.plot, 1)

        # 关闭自动自适应，后续完全手动控制视图
        self.vb.disableAutoRange()                        # ← 新增：禁止自动缩放
        self._doing_autoscale = False                     # ← 新增：正在执行手动自适应的标记
        self._has_user_view = False                       # ← 新增：是否已经有稳定视图（首条曲线后会设 True）

        self.legend = self.plot.addLegend() if self.legend_chk.isChecked() else None
        self.legend_chk.stateChanged.connect(self.toggle_legend)

        self.add_btn.clicked.connect(self.add_csv)
        self.autoscale_btn.clicked.connect(self.auto_range)
        self.clear_btn.clicked.connect(self.clear_all)
        self.reset_sel_btn.clicked.connect(self.reset_selected_offset)
        self.copy_offsets_btn.clicked.connect(self.copy_offsets)

        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.on_timer)
        self.timer.start(int(self.refresh_spin.value() * 1000))
        self.refresh_spin.valueChanged.connect(lambda v: self.timer.start(int(v * 1000)))

        self.series_items = []
        self.curves = []

        self._drag_series_index = None
        self._drag_start_pos = None
        self._drag_start_offsets = (0.0, 0.0)

        self.statusBar().showMessage("拖拽模式：左键按住曲线平移；滚轮缩放；双击自适应。")
        self.plot.scene().sigMouseClicked.connect(self.on_scene_click)

    # ---------- Legend handling ----------
    def _apply_curve_name(self, idx: int, name: str):
        """Portable way to apply legend label across pyqtgraph versions.
        1) setOpts(name=...) if available;
        2) if legend exists, remove+add to refresh label.
        """
        curve = self.curves[idx]
        # Option A: use setOpts if available
        if hasattr(curve, 'setOpts'):
            try:
                curve.setOpts(name=name)
            except Exception:
                # Fallback to direct opts dict
                try:
                    curve.opts['name'] = name
                except Exception:
                    pass
        else:
            # Very old versions: try opts dict
            try:
                curve.opts['name'] = name
            except Exception:
                pass
        # Ensure legend reflects the change
        if self.legend is not None:
            try:
                self.legend.removeItem(curve)
            except Exception:
                pass
            try:
                self.legend.addItem(curve, name)
            except Exception:
                pass

    def toggle_legend(self, st):
        if st and self.legend is None:
            self.legend = self.plot.addLegend()
        elif not st and self.legend is not None:
            try:
                self.legend.setParentItem(None)
            except Exception:
                pass
            self.legend = None
        self._refresh_legend()   # ← 新增
        self.redraw()

    def add_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        item = SeriesItem(path, line_color=pg.intColor(len(self.series_items) * 31))
        item.data_changed.connect(self.redraw)

        # 直接把 SeriesItem 小部件加到 panel
        self.series_panel.addWidget(item)
        self.series_items.append(item)

        curve = self.plot.plot(pen=pg.mkPen(item.line_color, width=2), name=item.legend_name())
        self.curves.append(curve)

        # 绑定移除
        item.remove_btn.clicked.connect(lambda: self.remove_item(item))

        self._refresh_legend()
        self.redraw()

    def remove_item(self, item: SeriesItem):
        if item not in self.series_items:
            return
        idx = self.series_items.index(item)

        # 先从图里移除曲线
        try:
            self.plot.removeItem(self.curves[idx])
        except Exception:
            pass
        self.curves.pop(idx)

        # 再从 panel 移除对应的小部件
        self.series_panel.removeWidget(item)
        item.setParent(None)       # 或者 item.deleteLater()

        # 维护列表
        self.series_items.pop(idx)

        # 修正当前拖拽选中索引
        if self._drag_series_index == idx:
            self._drag_series_index = None
            self.selected_lbl.setText("Selected: <none>")
        elif self._drag_series_index is not None and self._drag_series_index > idx:
            self._drag_series_index -= 1

        self._refresh_legend()
        self.redraw()



    # ---------- View ops ----------
    def auto_range(self):
        # 只手动自适应一次，之后仍保持禁用自动自适应
        self._doing_autoscale = True
        try:
            self.vb.autoRange()
        finally:
            self._doing_autoscale = False
            self._has_user_view = True


    def clear_all(self):
        # 移除所有曲线
        for c in self.curves:
            try:
                self.plot.removeItem(c)
            except Exception:
                pass
        self.curves.clear()

        # 移除所有 SeriesItem 小部件
        while self.series_items:
            it = self.series_items.pop()
            self.series_panel.removeWidget(it)
            it.setParent(None)     # 或 it.deleteLater()

        self._drag_series_index = None
        self.selected_lbl.setText("Selected: <none>")
        self._refresh_legend()
        self.redraw()



    def on_scene_click(self, ev):
        if ev.double():
            self.auto_range()

    # ---------- Timer ----------
    def on_timer(self):
        changed = False
        for it in self.series_items:
            if it.maybe_reload():
                changed = True
        if changed:
            self.redraw()

    # ---------- Dragging ----------
    def _pick_series_near(self, px: float, py: float):
        best_idx, best_dist = None, float('inf')
        x_range = self.vb.viewRange()[0]
        y_range = self.vb.viewRange()[1]
        tol_x = (x_range[1] - x_range[0]) * 0.01
        tol_y = (y_range[1] - y_range[0]) * 0.02
        for i, it in enumerate(self.series_items):
            if it.lock_chk.isChecked():
                continue
            x, y = it.get_xy_shifted()
            if x is None or y is None or len(x) == 0:
                continue
            j = np.searchsorted(x, px)
            for k in (max(0, j - 1), min(len(x) - 1, j)):
                dx = abs(x[k] - px) / max(tol_x, 1e-12)
                dy = abs(y[k] - py) / max(tol_y, 1e-12)
                d = dx + dy
                if d < best_dist:
                    best_dist, best_idx = d, i
        return best_idx if best_dist < 2.0 else None

    def _drag_begin(self, pos_v):
        if not (self.drag_x_chk.isChecked() or self.drag_y_chk.isChecked()):
            return
        px, py = float(pos_v.x()), float(pos_v.y())
        idx = self._pick_series_near(px, py)
        if idx is None:
            return
        self._drag_series_index = idx
        self._drag_start_pos = (px, py)
        self._drag_start_offsets = self.series_items[idx].get_offsets()
        self.selected_lbl.setText(f"Selected: {self.series_items[idx].legend_name()}")
        self.statusBar().showMessage("Dragging…")

    def _drag_update(self, pos_v):
        if self._drag_series_index is None or self._drag_start_pos is None:
            return
        it = self.series_items[self._drag_series_index]
        # NEW: 若拖动过程中被锁住，立即停止响应
        if it.lock_chk.isChecked():
            return

        px, py = float(pos_v.x()), float(pos_v.y())
        sx, sy = self._drag_start_pos
        dx_cur, dy_cur = px - sx, py - sy
        dx0, dy0 = self._drag_start_offsets
        use_x, use_y = self.drag_x_chk.isChecked(), self.drag_y_chk.isChecked()
        dx_new = dx0 + (dx_cur if use_x else 0.0)
        dy_new = dy0 + (dy_cur if use_y else 0.0)
        it.set_offsets(dx_new, dy_new)


    def _drag_end(self):
        if self._drag_series_index is None:
            return
        it = self.series_items[self._drag_series_index]
        dx, dy = it.get_offsets()
        self.statusBar().showMessage(f"Pinned offsets for '{it.legend_name()}': Δx={dx:.6f}s, Δy={dy:.6f}")

    def reset_selected_offset(self):
        if self._drag_series_index is None:
            return
        it = self.series_items[self._drag_series_index]
        it.set_offsets(0.0, 0.0)
        self.statusBar().showMessage(f"Offsets reset for '{it.legend_name()}'")

    def copy_offsets(self):
        payload = []
        for it in self.series_items:
            dx, dy = it.get_offsets()
            payload.append({
                "file": it.path,
                "legend": it.legend_name(),
                "dx_seconds": dx,
                "dy_units": dy,
                "locked": bool(it.lock_chk.isChecked()),
                "x_col": it.x_combo.currentText(),
                "y_col": it.y_combo.currentText(),
            })
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        QtWidgets.QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Offsets copied to clipboard as JSON")

    # ---------- Redraw ----------
    def redraw(self):
        # 记录当前视图范围
        x_before, y_before = self.vb.viewRange()

        # —— 原有的逐曲线 setData —— 
        for i, it in enumerate(self.series_items):
            x, y = it.get_xy_shifted()
            name = it.legend_name()
            if x is None or y is None or len(x) == 0:
                self.curves[i].setData([], [])
                # 这里不要再做 legend 的 add/remove，集中到 _refresh_legend()
                continue
            self.curves[i].setData(x, y, connect='finite')
        # —— 统一刷新 legend（见下一节）——
        self._refresh_legend()

        # 如果用户刚点了 Autoscale，就不恢复旧视图
        if self._doing_autoscale:
            self._has_user_view = True
            return

        # 如果还没有稳定视图（比如首次加载有数据），做一次自动适配
        if not self._has_user_view:
            try:
                self.vb.autoRange()
            finally:
                self._has_user_view = True
            return

        # 否则，恢复先前的视图范围（防止“回到初始状态”）
        self.vb.setRange(xRange=x_before, yRange=y_before, padding=0.0)

    def _refresh_legend(self):
        if not self.legend:
            return
        # 清空 legend（不同版本：优先用 clear，兜底逐条移除）
        try:
            self.legend.clear()
        except Exception:
            # 旧版本没有 clear()，就暴力重建
            try:
                self.legend.setParentItem(None)
            except Exception:
                pass
            self.legend = self.plot.addLegend()

        # 重新按当前曲线顺序添加
        for i, it in enumerate(self.series_items):
            name = it.legend_name()
            try:
                self.legend.addItem(self.curves[i], name)
            except Exception:
                pass




def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CSVPlotter()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
