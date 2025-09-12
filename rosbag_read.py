#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Bag → CSV 可视化导出器（PyQt5）

功能概述
- 选择 ROS2 bag 目录（需包含 metadata.yaml 与 *.db3）
- 自动列出 topic 与 message type
- 按 message type 递归解析字段，支持内嵌消息与定长数组（变量长序列默认不展开）
- 为每个 topic 勾选需要导出的字段（列名格式：<topic>/<field_path>）
- 统一时间轴（bag 接收时间 或 header.stamp），最近邻对齐（可设置容差，毫秒）
- 导出所有所选字段到同一 CSV

依赖
- PyQt5, pandas
- ROS2 Humble 环境，且可导入：rosbag2_py, rclpy, rosidl_runtime_py

注意
- header.stamp 需要消息内含 std_msgs/Header；若无则 fallback 到 bag 时间
- 变量长度序列（如 list<float>）默认不展开（避免列数不定），可按需要在 _flatten_message() 中扩展策略
"""

import os
import sys
import math
import sqlite3
from collections import defaultdict
from bisect import bisect_left
from typing import Dict, Any, List, Tuple, Optional

from PyQt5 import QtWidgets, QtCore
import pandas as pd

# ROS2 imports (确保已 source 对应 ROS2 环境)
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from PyQt5.QtGui import QFont, QFontDatabase



# ---- 类型→字段模板（可按需扩充） ----
TYPE_TEMPLATES = {
    "geometry_msgs/msg/PointStamped": ["point.x", "point.y", "point.z"],
    "geometry_msgs/msg/Vector3Stamped": ["vector.x", "vector.y", "vector.z"],
    "geometry_msgs/msg/PoseStamped": ["pose.position.x", "pose.position.y", "pose.position.z"],  # 默认不勾选四元数
    "geometry_msgs/msg/TwistStamped": [
        "twist.linear.x", "twist.linear.y", "twist.linear.z",
        "twist.angular.x", "twist.angular.y", "twist.angular.z"
    ],
    "nav_msgs/msg/Odometry": [
        "pose.pose.position.x", "pose.pose.position.y", "pose.pose.position.z",
        "twist.twist.linear.x", "twist.twist.linear.y", "twist.twist.linear.z"
    ],
    "sensor_msgs/msg/Imu": [
        "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"
    ],
    "sensor_msgs/msg/NavSatFix": ["latitude", "longitude", "altitude"],
    "geometry_msgs/msg/TransformStamped": [
        "transform.translation.x", "transform.translation.y", "transform.translation.z"
    ],
    "std_msgs/msg/Float64": ["data"],
    "std_msgs/msg/Float32": ["data"],
    "std_msgs/msg/Int32":  ["data"],
    "std_msgs/msg/Bool":   ["data"],
}



# ----------------------------- 工具函数（消息反射与拍平） -----------------------------

_PRIMITIVE_PY_TYPES = (bool, int, float, str, bytes)

def _is_primitive_value(v) -> bool:
    return isinstance(v, _PRIMITIVE_PY_TYPES) or v is None

def _get_fields_and_types(msg) -> Dict[str, str]:
    # ROS2 Python 消息对象具有 get_fields_and_field_types()
    return msg.get_fields_and_field_types()

def _is_fixed_size_array_of_primitives(ros_type: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    判断是否为定长数组且元素为基础类型。
    例如: 'float32[3]' → (True, 3, 'float32')
    """
    if '[' in ros_type and ']' in ros_type:
        base = ros_type.split('[')[0]
        size_str = ros_type.split('[')[1].split(']')[0]
        if size_str.isdigit():
            # 仅认为基础类型：布尔、整型、浮点、字符串（字符串通常非定长，默认不展开）
            if base in ('boolean','byte','char',
                        'float','float32','float64',
                        'double','int8','uint8','int16','uint16','int32','uint32','int64','uint64'):
                return True, int(size_str), base
    return False, None, None

def _get_attr(obj, name):
    # 支持 . 与 数组下标访问（例如 pose.position.x 或 data[0]）
    if '[' in name and name.endswith(']'):
        base, idx = name.split('[', 1)
        idx = int(idx[:-1])
        seq = getattr(obj, base)
        return seq[idx]
    else:
        return getattr(obj, name)

def _flatten_message(msg, prefix: str = "", out: Dict[str, Any] = None, schema: Dict[str, str] = None):
    """
    递归将 ROS 消息拍平成 {列名: 值} 的字典。
    - prefix: 列名前缀（例如 '/tf/pose'）
    - schema: 记录列名→类型（可选）
    """
    if out is None:
        out = {}
    if schema is None:
        schema = {}

    fields = _get_fields_and_types(msg)
    for name, ros_type in fields.items():
        col_base = f"{prefix}.{name}" if prefix else name

        # 处理定长数组的基础类型（如 float32[3]）
        is_fixed, size, base = _is_fixed_size_array_of_primitives(ros_type)
        if is_fixed:
            seq = getattr(msg, name)
            # 防御：长度不足则填 NaN
            for i in range(size):
                v = seq[i] if (i < len(seq)) else float('nan')
                col = f"{col_base}[{i}]"
                out[col] = v
                schema[col] = f"{base}[]"
            continue

        # 处理 builtin_interfaces/Time
        if ros_type.endswith('builtin_interfaces/msg/Time'):
            t = getattr(msg, name)
            # 转 ns，便于对齐
            ns = int(getattr(t, 'sec')) * 10**9 + int(getattr(t, 'nanosec'))
            out[col_base] = ns
            schema[col_base] = 'time_ns'
            continue

        # 处理基础类型与 bytes/string
        v = getattr(msg, name)
        if _is_primitive_value(v):
            out[col_base] = v
            schema[col_base] = ros_type
            continue

        # 处理变量长度序列（默认不展开，避免列数不定；可按需要改成限制长度展开）
        if isinstance(v, (list, tuple)):
            # 可选策略：记录长度
            out[col_base + ".__len__"] = len(v)
            schema[col_base + ".__len__"] = 'sequence_len'
            # 如需展开前 N 项，可在此加入：
            # MAX_N = 8
            # for i in range(min(len(v), MAX_N)):
            #     elem = v[i]
            #     if _is_primitive_value(elem):
            #         out[f"{col_base}[{i}]"] = elem
            #         schema[f"{col_base}[{i}]"] = f"{ros_type}[]"
            continue

        # 处理子消息（递归）
        try:
            _flatten_message(v, prefix=col_base, out=out, schema=schema)
        except Exception:
            # 保险：无法递归时，保留为字符串表示
            out[col_base] = str(v)
            schema[col_base] = ros_type

    return out, schema


def _get_header_stamp_ns_if_available(msg) -> Optional[int]:
    """若消息包含 header.stamp，返回其纳秒时间；否则 None。"""
    try:
        header = getattr(msg, 'header', None)
        if header is None:
            return None
        stamp = getattr(header, 'stamp', None)
        if stamp is None:
            return None
        return int(stamp.sec) * 10**9 + int(stamp.nanosec)
    except Exception:
        return None
    
def _set_cjk_font(app):
    # 按优先级挑一款系统已安装的中文等宽/通用字体
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK",   # Ubuntu 装了 fonts-noto-cjk 就有
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Sarasa UI SC", "Sarasa Mono SC",
        "Microsoft YaHei UI", "Microsoft YaHei",  # 若在 Windows 原生运行
        "PingFang SC", "SimHei", "SimSun"
    ]
    fams = set(QFontDatabase().families())
    for name in candidates:
        if name in fams:
            app.setFont(QFont(name, 10))
            break


# ----------------------------- 读取 bag 与反序列化 -----------------------------

class BagIndex:
    """读取 bag 的 topic→type 映射并支持按 topic 遍历消息。"""
    def __init__(self, bag_dir: str):
        self.bag_dir = bag_dir
        self.reader = SequentialReader()
        storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        self.reader.open(storage_options, converter_options)

        self.topics_and_types = self.reader.get_all_topics_and_types()
        self.topic_type_map = {t.name: t.type for t in self.topics_and_types}

        # 为 read_next 走一次游标创建镜像（独立 reader 不带 seek，简单策略：重开 reader）
        self._opened = True

    def reopen(self):
        if self._opened:
            # 简单重开
            self.reader = SequentialReader()
            storage_options = StorageOptions(uri=self.bag_dir, storage_id='sqlite3')
            converter_options = ConverterOptions('', '')
            self.reader.open(storage_options, converter_options)

    def get_topics(self) -> List[str]:
        return sorted(self.topic_type_map.keys())

    def get_type(self, topic: str) -> str:
        return self.topic_type_map[topic]

    def iter_messages(self, wanted_topics: List[str]):
        """
        以时间顺序遍历 bag；仅返回 wanted_topics 的消息。
        产出： (topic, serialized, t_ns[bag])
        """
        self.reopen()
        has_next = self.reader.has_next()
        while has_next:
            topic, data, t = self.reader.read_next()
            if topic in wanted_topics:
                yield topic, data, int(t)
            has_next = self.reader.has_next()


# ----------------------------- 同步对齐（最近邻） -----------------------------

def _nearest_index(a: List[int], x: int) -> int:
    """返回 a 中与 x 最近的索引（a 升序）。"""
    i = bisect_left(a, x)
    if i == 0:
        return 0
    if i == len(a):
        return len(a) - 1
    before = a[i - 1]
    after = a[i]
    if abs(after - x) < abs(x - before):
        return i
    return i - 1

def align_and_merge(
    per_topic_series: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    selected_columns: Dict[str, List[str]],
    time_source: str = "bag",           # "bag" or "header"
    tol_ns: int = 5_000_000,            # 5 ms
    round_to_ms: Optional[int] = None   # 对齐后时间列可选取整（毫秒），例如 round_to_ms=1/5/10
) -> pd.DataFrame:
    """
    将各 topic 已解析的 (t_ns, {col:val}) 序列按统一时间轴合并。
    - 时间轴采用所有时间点的并集（你也可以改成交集策略）
    - 对于每个时间点，在其他 topic 中找最近时间，若差值 <= tol_ns，则取该行值，否则缺失
    - selected_columns: topic → 该 topic 需落盘的列（列名已是 <topic>/<field_path> 全名）

    返回 pandas.DataFrame，列包括：'t_ns'（统一时间）与各选列。
    """
    # 构建统一时间集合
    all_times = set()
    topic_times: Dict[str, List[int]] = {}
    topic_data: Dict[str, List[Dict[str, Any]]] = {}

    for topic, seq in per_topic_series.items():
        ts = [t for t, _ in seq]
        topic_times[topic] = ts
        topic_data[topic] = [d for _, d in seq]
        all_times.update(ts)

    axis = sorted(all_times)

    # 逐时间填充
    rows = []
    for t in axis:
        row = {"t_ns": t}
        for topic, ts in topic_times.items():
            cols = selected_columns.get(topic, [])
            if not cols:
                continue
            if not ts:
                # 没有任何数据
                for c in cols:
                    row[c] = math.nan
                continue
            j = _nearest_index(ts, t)
            if abs(ts[j] - t) <= tol_ns:
                vals = topic_data[topic][j]
                for c in cols:
                    row[c] = vals.get(c, math.nan)
            else:
                for c in cols:
                    row[c] = math.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    # 可选：对时间列取整到毫秒粒度（便于外部软件处理）
    if round_to_ms is not None and round_to_ms > 0:
        q = int(round_to_ms) * 1_000_000
        df["t_ns"] = ( (df["t_ns"] + q//2) // q ) * q
    return df


# ----------------------------- UI -----------------------------

class FieldSelectorDialog(QtWidgets.QDialog):
    """字段勾选对话框：显示某 topic 的字段树（拍平成路径），供多选。"""
    def __init__(self, topic: str, type_name: str, sample_msg: Any, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"选择字段 - {topic} ({type_name})")
        self.resize(700, 500)

        self.topic = topic
        self.type_name = type_name
        self.sample_msg = sample_msg

        self.selected_paths: List[str] = []

        # 展平示例消息，获得可选列（路径）及类型说明
        flat, schema = _flatten_message(sample_msg)
        paths = sorted(flat.keys())

        layout = QtWidgets.QVBoxLayout(self)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setPlaceholderText("输入关键词过滤，如 pose.position 或 x ...")
        layout.addWidget(self.search_edit)

        self.listw = QtWidgets.QListWidget(self)
        self.listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for p in paths:
            it = QtWidgets.QListWidgetItem(f"{p}   [{schema.get(p,'')}]")
            it.setData(QtCore.Qt.UserRole, p)
            self.listw.addItem(it)

        # === 根据模板自动预选 ===
        tmpl = TYPE_TEMPLATES.get(self.type_name, [])
        if tmpl:
            want = set(tmpl)
            for i in range(self.listw.count()):
                it = self.listw.item(i)
                path = it.data(QtCore.Qt.UserRole)
                if path in want:
                    it.setSelected(True)

        layout.addWidget(self.listw)

        # 常用快捷过滤（位置/姿态等）
        quick = QtWidgets.QHBoxLayout()
        for key in ["x", "y", "z", "position", "orientation", "linear", "angular", "stamp", "pose", "twist"]:
            btn = QtWidgets.QPushButton(key, self)
            btn.clicked.connect(lambda _, k=key: self._filter_contains(k))
            quick.addWidget(btn)
        layout.addLayout(quick)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.search_edit.textChanged.connect(self._apply_filter)

    def _filter_contains(self, word: str):
        self.search_edit.setText(word)

    def _apply_filter(self, text: str):
        text = text.strip().lower()
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            vis = (text in it.text().lower())
            it.setHidden(not vis)

    def accept(self):
        self.selected_paths = []
        for it in self.listw.selectedItems():
            self.selected_paths.append(it.data(QtCore.Qt.UserRole))
        super().accept()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROS2 Bag → CSV 导出器")
        self.resize(1000, 650)

        # 状态
        self.bag: Optional[BagIndex] = None
        self.topic_to_type: Dict[str, str] = {}
        self.type_cache: Dict[str, Any] = {}  # type_name → Python class
        self.sample_cache: Dict[str, Any] = {}  # type_name → 空消息样本
        self.selected_fields: Dict[str, List[str]] = defaultdict(list)  # topic → field_path 列

        # UI
        root = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 2)

        btn_bag = QtWidgets.QPushButton("选择bag目录", self)
        btn_bag.clicked.connect(self.choose_bag_dir)
        left.addWidget(btn_bag)

        self.bag_label = QtWidgets.QLabel("未选择", self)
        left.addWidget(self.bag_label)

        self.topic_list = QtWidgets.QListWidget(self)
        self.topic_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        left.addWidget(self.topic_list, 1)

        btn_field = QtWidgets.QPushButton("选择字段（针对当前选中topic）", self)
        btn_field.clicked.connect(self.choose_fields_for_topic)
        left.addWidget(btn_field)

        # 右侧：已选字段表 + 导出参数
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 3)

        self.sel_table = QtWidgets.QTableWidget(self)
        self.sel_table.setColumnCount(2)
        self.sel_table.setHorizontalHeaderLabels(["Topic", "Field Path"])
        self.sel_table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.sel_table, 1)

        # 参数区
        form = QtWidgets.QGroupBox("对齐与导出参数", self)
        grid = QtWidgets.QGridLayout(form)

        self.time_src_combo = QtWidgets.QComboBox(self)
        self.time_src_combo.addItems(["bag 接收时间", "header.stamp（若无则回退 bag 时间）"])
        grid.addWidget(QtWidgets.QLabel("时间源："), 0, 0)
        grid.addWidget(self.time_src_combo, 0, 1)

        self.tol_spin = QtWidgets.QSpinBox(self)
        self.tol_spin.setRange(0, 10_000)  # ms
        self.tol_spin.setValue(5)
        grid.addWidget(QtWidgets.QLabel("最近邻对齐容差 (ms)："), 1, 0)
        grid.addWidget(self.tol_spin, 1, 1)

        self.round_ms_spin = QtWidgets.QSpinBox(self)
        self.round_ms_spin.setRange(0, 1000)
        self.round_ms_spin.setValue(0)
        grid.addWidget(QtWidgets.QLabel("时间取整 (ms，0=不取整)："), 2, 0)
        grid.addWidget(self.round_ms_spin, 2, 1)

        right.addWidget(form)

        btn_export = QtWidgets.QPushButton("导出为 CSV", self)
        btn_export.clicked.connect(self.export_csv)
        right.addWidget(btn_export)

        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        right.addWidget(self.log, 1)

    # ---------------- UI slots ----------------

    def _log(self, s: str):
        self.log.appendPlainText(s)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def choose_bag_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 bag 目录")
        if not d:
            return
        # 基本检查
        if not os.path.exists(os.path.join(d, "metadata.yaml")):
            QtWidgets.QMessageBox.warning(self, "错误", "所选目录缺少 metadata.yaml")
            return
        has_db3 = any(fn.endswith(".db3") for fn in os.listdir(d))
        if not has_db3:
            QtWidgets.QMessageBox.warning(self, "错误", "所选目录下未发现 .db3 文件")
            return

        try:
            self.bag = BagIndex(d)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "打开失败", str(e))
            return

        self.bag_label.setText(d)
        self.topic_to_type = {t: self.bag.get_type(t) for t in self.bag.get_topics()}
        self.topic_list.clear()
        for t in sorted(self.topic_to_type.keys()):
            self.topic_list.addItem(f"{t}  [{self.topic_to_type[t]}]")
        self.selected_fields.clear()
        self._refresh_selected_table()
        self._log(f"已载入 bag：{d}")
        self._log(f"发现 {len(self.topic_to_type)} 个 topic")

    def _load_msg_class(self, type_name: str):
        if type_name in self.type_cache:
            return self.type_cache[type_name]
        cls = get_message(type_name)
        self.type_cache[type_name] = cls
        return cls

    def _deserialize_one(self, type_name: str, data: bytes):
        cls = self._load_msg_class(type_name)
        return deserialize_message(data, cls)

    def _get_sample_msg(self, type_name: str):
        if type_name in self.sample_cache:
            return self.sample_cache[type_name]
        # 尝试从 bag 中读一条该类型消息作为样本；若找不到，构造空消息实例
        try:
            # 简便策略：扫描一会儿
            want_topic = None
            for t, ty in self.topic_to_type.items():
                if ty == type_name:
                    want_topic = t
                    break
            if want_topic is not None and self.bag is not None:
                for topic, data, _ in self.bag.iter_messages([want_topic]):
                    msg = self._deserialize_one(type_name, data)
                    self.sample_cache[type_name] = msg
                    return msg
        except Exception:
            pass
        # 构造一个默认实例（字段为零值）
        cls = self._load_msg_class(type_name)
        msg = cls()
        self.sample_cache[type_name] = msg
        return msg

    def choose_fields_for_topic(self):
        it = self.topic_list.currentItem()
        if not it:
            QtWidgets.QMessageBox.information(self, "提示", "请先在左侧选择一个 topic")
            return
        # 解析 topic 与类型
        txt = it.text()
        topic = txt.split("  [")[0].strip()
        type_name = self.topic_to_type.get(topic)
        if not type_name:
            QtWidgets.QMessageBox.warning(self, "错误", "未找到该 topic 的类型信息")
            return

        sample = self._get_sample_msg(type_name)
        dlg = FieldSelectorDialog(topic, type_name, sample, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # 将选中的路径记录为列（列名采用 <topic>/<path>）
            paths = dlg.selected_paths
            if paths:
                cols = [f"{topic}/{p}" for p in paths]
                # 合并（去重）
                existed = set(self.selected_fields.get(topic, []))
                for c in cols:
                    if c not in existed:
                        existed.add(c)
                self.selected_fields[topic] = sorted(existed)
                self._refresh_selected_table()

    def _refresh_selected_table(self):
        rows = []
        for topic, cols in sorted(self.selected_fields.items()):
            for p in cols:
                # p 是完整列名 <topic>/<path>，表格里只显示路径
                rows.append((topic, p.split('/', 1)[1] if '/' in p else p))

        self.sel_table.setRowCount(len(rows))
        for i, (topic, path) in enumerate(rows):
            self.sel_table.setItem(i, 0, QtWidgets.QTableWidgetItem(topic))
            self.sel_table.setItem(i, 1, QtWidgets.QTableWidgetItem(path))

    def export_csv(self):
        if self.bag is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择 bag 目录")
            return
        total_cols = sum(len(v) for v in self.selected_fields.values())
        if total_cols == 0:
            QtWidgets.QMessageBox.information(self, "提示", "请先为至少一个 topic 选择字段")
            return

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存 CSV", filter="CSV Files (*.csv)")
        if not out_path:
            return

        time_source = "bag" if self.time_src_combo.currentIndex() == 0 else "header"
        tol_ms = int(self.tol_spin.value())
        round_ms = int(self.round_ms_spin.value())
        self._log(f"开始导出：time_source={time_source}, tol_ms={tol_ms}, round_ms={round_ms}")
        try:
            df = self._do_export(time_source=time_source, tol_ms=tol_ms, round_ms=round_ms)
            df.to_csv(out_path, index=False)
            self._log(f"✅ 导出完成：{out_path}")
            QtWidgets.QMessageBox.information(self, "完成", f"已导出：{out_path}")
        except Exception as e:
            self._log(f"❌ 导出失败：{e}")
            QtWidgets.QMessageBox.critical(self, "导出失败", str(e))

    def _do_export(self, time_source: str, tol_ms: int, round_ms: int) -> pd.DataFrame:
        # 反向映射：完整列名 → (topic, path)
        need_topics = [t for t, cols in self.selected_fields.items() if cols]
        type_map = {t: self.topic_to_type[t] for t in need_topics}

        # 每 topic 收集：[(t_ns, {<topic/path>: value, ...}), ...]
        per_topic_series: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {t: [] for t in need_topics}

        # 重开 reader
        self.bag.reopen()

        # 为解析做缓存
        msg_cls_cache = {ty: self._load_msg_class(ty) for ty in set(type_map.values())}

        # 建一个快速表：topic → 所需字段路径（不含 <topic>/ 前缀）
        topic_path_need: Dict[str, List[str]] = {}
        for topic, cols in self.selected_fields.items():
            paths = []
            for full in cols:
                # full 是 <topic>/<path>
                if full.startswith(topic + "/"):
                    paths.append(full[len(topic)+1:])
            topic_path_need[topic] = paths

        # 预先扫描一遍，逐条解析
        cnt = 0
        for topic, data, t_bag in self.bag.iter_messages(need_topics):
            ty = type_map[topic]
            msg = deserialize_message(data, msg_cls_cache[ty])

            # 选时间源
            t_ns = t_bag
            if time_source == "header":
                t_header = _get_header_stamp_ns_if_available(msg)
                if t_header is not None:
                    t_ns = t_header

            # 拍平消息
            flat, _ = _flatten_message(msg)

            # 取所需列 → 构造 {<topic>/<path>: value}
            record = {}
            for p in topic_path_need[topic]:
                key_flat = p
                v = flat.get(key_flat, math.nan)
                record[f"{topic}/{p}"] = v

            per_topic_series[topic].append((t_ns, record))
            cnt += 1
            if cnt % 2000 == 0:
                self._log(f"已解析 {cnt} 条消息...")

        self._log(f"解析完成，共 {cnt} 条消息。开始对齐合并...")

        # 对齐合并
        df = align_and_merge(
            per_topic_series=per_topic_series,
            selected_columns=self.selected_fields,
            time_source=time_source,
            tol_ns=tol_ms * 1_000_000,
            round_to_ms=(round_ms if round_ms > 0 else None)
        )

        # 附带一个更友好的时间列（秒）
        df.insert(0, "t_s", df["t_ns"] / 1e9)

        self._log(f"合并完成：{df.shape[0]} 行，{df.shape[1]} 列。")
        return df



def main():
    app = QtWidgets.QApplication(sys.argv)
    _set_cjk_font(app)   # ✅ 关键：设置全局字体
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
