#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS bag -> CSV Export Tool (with simple Tkinter GUI)

Features:
1. Supports both .bag and .bag.active files.
   - For .active files, automatically calls `rosbag reindex` to rebuild the index.
2. Loads all topics contained in the bag and allows the user to select one topic.
3. Inspects the first message of the selected topic, flattens it into column names
   (using dot-separated hierarchical names, e.g., pose.position.x).
4. Lets the user select which flattened fields to export.
5. Exports the selected topic to a CSV file:
   - Column 'time' is aligned such that the bag start time is 0.0 seconds.
   - By default, uses header.stamp if available; otherwise uses the bag's message time.
6. Simple progress bar and status messages during export.
"""

import os
import csv
import subprocess
import traceback

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import rosbag
import rospy  # imported to ensure ROS types/genpy are available


# ==================== Message flattening ====================

_PRIMITIVE_TYPES = (bool, int, float, str)


def _is_primitive(x):
    return isinstance(x, _PRIMITIVE_TYPES)


def flatten_message(msg, prefix="", out=None, max_depth=6):
    """
    Flatten a ROS message (or arbitrary Python object) into a flat dict:
        { "field.name": value, ... }

    Rules:
    - Nested fields are joined by '.', e.g. pose.position.x
    - Lists/tuples:
        * If all elements are "simple" (primitive or time-like), they are joined into
          a single string with ';' as separator.
        * Otherwise, each element is expanded separately with index-based prefixes,
          e.g. arr.0.x, arr.1.x, ...
    - Time-like objects (having a `to_sec()` method) are converted to float seconds.
    - If max_depth is exceeded, the value is converted to string without further expansion.
    """
    if out is None:
        out = {}

    def key_name(p):
        return p[:-1] if p.endswith(".") else p

    if max_depth <= 0:
        out[key_name(prefix)] = str(msg)
        return out

    # Time-like objects with to_sec()
    if hasattr(msg, "to_sec") and callable(getattr(msg, "to_sec")):
        out[key_name(prefix)] = msg.to_sec()
        return out

    # Primitive types
    if _is_primitive(msg):
        out[key_name(prefix)] = msg
        return out

    # List or tuple
    if isinstance(msg, (list, tuple)):
        if not msg:
            out[key_name(prefix)] = ""
            return out

        # Check if all elements are simple (primitive or time-like)
        simple = True
        for v in msg:
            if hasattr(v, "to_sec") and callable(getattr(v, "to_sec")):
                continue
            if not _is_primitive(v):
                simple = False
                break

        if simple:
            vals = []
            for v in msg:
                if hasattr(v, "to_sec") and callable(getattr(v, "to_sec")):
                    vals.append(str(v.to_sec()))
                else:
                    vals.append(str(v))
            out[key_name(prefix)] = ";".join(vals)
            return out
        else:
            # Expand each element with an index
            for i, v in enumerate(msg):
                flatten_message(v, prefix=f"{prefix}{i}.", out=out, max_depth=max_depth - 1)
            return out

    # ROS message: typically has __slots__
    if hasattr(msg, "__slots__"):
        for slot in msg.__slots__:
            if slot.startswith("_"):
                continue
            try:
                value = getattr(msg, slot)
            except Exception:
                continue
            flatten_message(value, prefix=f"{prefix}{slot}.", out=out, max_depth=max_depth - 1)
        return out

    # Fallback: unknown complex type, just stringify
    out[key_name(prefix)] = str(msg)
    return out


# ==================== .active handling (reindex) ====================

def ensure_bag_indexed(path):
    """
    If the path ends with '.active', call `rosbag reindex <path>` to rebuild
    the index. Raises RuntimeError if `rosbag` is not found or reindex fails.
    """
    if not path.endswith(".active"):
        return

    # Ensure `rosbag` command exists
    try:
        subprocess.run(
            ["rosbag", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        raise RuntimeError("Unable to find 'rosbag' command. Please ensure you run this inside a ROS1 environment.")

    # Run `rosbag reindex`
    proc = subprocess.run(
        ["rosbag", "reindex", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("rosbag reindex failed:\n" + proc.stderr)


# ==================== GUI application ====================

class RosbagCSVExporterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ROS bag -> CSV Export Tool")
        self.geometry("900x600")

        # Bag state
        self.bag_path = None
        self.bag = None
        self.bag_start_time = None

        # Topic information
        self.topics_info = {}       # topic -> TopicInfo
        self.topic_fields = {}      # topic -> [field_name, ...]
        self.topic_field_vars = {}  # topic -> {field_name: tk.BooleanVar}

        # Currently selected topic
        self.current_topic = None

        # Option: prefer header.stamp or use bag time
        self.use_header_stamp_var = tk.BooleanVar(value=True)

        # Status and progress
        self.status_var = tk.StringVar(value="Please load a bag file.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.bag_path_var = tk.StringVar(value="No file selected")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI building ----------

    def _build_ui(self):
        # Top: "Load bag" button and bag path label
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        btn_load = ttk.Button(top_frame, text="Load bag", command=self.on_load_bag)
        btn_load.pack(side=tk.LEFT)

        lbl_path = ttk.Label(top_frame, textvariable=self.bag_path_var)
        lbl_path.pack(side=tk.LEFT, padx=10)

        # Middle: left side (topics), right side (fields)
        middle_frame = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: topic list
        left_frame = ttk.Frame(middle_frame)
        middle_frame.add(left_frame, weight=1)

        lbl_topics = ttk.Label(left_frame, text="Topic list")
        lbl_topics.pack(anchor=tk.W)

        self.topics_listbox = tk.Listbox(left_frame, height=20)
        self.topics_listbox.pack(fill=tk.BOTH, expand=True)
        self.topics_listbox.bind("<<ListboxSelect>>", self.on_topic_selected)

        # Right: fields and options
        right_frame = ttk.Frame(middle_frame)
        middle_frame.add(right_frame, weight=2)

        top_right_frame = ttk.Frame(right_frame)
        top_right_frame.pack(fill=tk.X)

        lbl_fields = ttk.Label(top_right_frame, text="Fields (checked columns will be exported)")
        lbl_fields.pack(side=tk.LEFT)

        btn_select_all = ttk.Button(top_right_frame, text="Select all", command=self.on_select_all_fields)
        btn_select_all.pack(side=tk.RIGHT, padx=2)

        btn_clear_all = ttk.Button(top_right_frame, text="Clear all", command=self.on_clear_all_fields)
        btn_clear_all.pack(side=tk.RIGHT, padx=2)

        # Option: prefer header.stamp
        chk_header = ttk.Checkbutton(
            right_frame,
            text="Prefer header.stamp as timestamp (fallback to bag time)",
            variable=self.use_header_stamp_var,
        )
        chk_header.pack(anchor=tk.W, pady=4)

        # Scrollable field list
        self.fields_canvas = tk.Canvas(right_frame, borderwidth=0)
        self.fields_frame = ttk.Frame(self.fields_canvas)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.fields_canvas.yview)
        self.fields_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.fields_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fields_canvas.create_window((0, 0), window=self.fields_frame, anchor="nw")
        self.fields_frame.bind(
            "<Configure>",
            lambda e: self.fields_canvas.configure(scrollregion=self.fields_canvas.bbox("all")),
        )

        # Bottom: export button, progress bar, status label
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        btn_export = ttk.Button(bottom_frame, text="Export current topic to CSV", command=self.on_export_csv)
        btn_export.pack(side=tk.LEFT)

        progress = ttk.Progressbar(bottom_frame, variable=self.progress_var, maximum=100)
        progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        lbl_status = ttk.Label(bottom_frame, textvariable=self.status_var)
        lbl_status.pack(side=tk.RIGHT)

    # ---------- Event handlers ----------

    def on_load_bag(self):
        """Handle 'Load bag' button."""
        path = filedialog.askopenfilename(
            title="Select ROS bag file",
            filetypes=[("ROS bag files", "*.bag *.bag.active"), ("All files", "*.*")],
        )
        if not path:
            return

        self.status_var.set("Processing bag file...")
        self.update_idletasks()

        try:
            # Handle .active case
            ensure_bag_indexed(path)

            # Close previous bag if any
            if self.bag is not None:
                self.bag.close()

            self.bag = rosbag.Bag(path, "r")
            self.bag_path = path
            self.bag_start_time = self.bag.get_start_time()

            info = self.bag.get_type_and_topic_info()
            self.topics_info = info.topics  # dict: topic -> TopicInfo

            # Populate topic listbox
            self.topics_listbox.delete(0, tk.END)
            topics_sorted = sorted(self.topics_info.keys())
            for t in topics_sorted:
                self.topics_listbox.insert(tk.END, t)

            # Clear fields
            self.topic_fields.clear()
            self.topic_field_vars.clear()
            self.current_topic = None
            self._clear_fields_frame()

            self.bag_path_var.set(path)
            self.status_var.set("Bag loaded. Please select a topic.")

        except Exception as e:
            if self.bag is not None:
                self.bag.close()
            self.bag = None
            self.bag_path = None
            self.bag_start_time = None
            self.topics_info = {}
            self.topic_fields.clear()
            self.topic_field_vars.clear()
            self._clear_fields_frame()
            self.bag_path_var.set("No file selected")

            messagebox.showerror("Error", f"Failed to load bag:\n{e}")
            traceback.print_exc()
            self.status_var.set("Load failed.")

    def on_topic_selected(self, event):
        """When the user selects a topic in the listbox."""
        if self.bag is None:
            return

        sel = self.topics_listbox.curselection()
        if not sel:
            return

        idx = sel[0]
        topic = self.topics_listbox.get(idx)
        self.current_topic = topic

        self.status_var.set(f"Inspecting topic: {topic}")
        self.update_idletasks()

        # Inspect fields if not done yet
        if topic not in self.topic_fields:
            fields = self._inspect_topic_fields(topic)
            self.topic_fields[topic] = fields
            self.topic_field_vars[topic] = {name: tk.BooleanVar(value=False) for name in fields}

        # Fill the fields frame
        self._populate_fields_frame(topic)
        self.status_var.set(f"Topic loaded: {topic} ({len(self.topic_fields[topic])} fields)")

    def on_select_all_fields(self):
        """Select all fields of the current topic."""
        topic = self.current_topic
        if not topic or topic not in self.topic_field_vars:
            return
        for var in self.topic_field_vars[topic].values():
            var.set(True)

    def on_clear_all_fields(self):
        """Clear all field selections of the current topic."""
        topic = self.current_topic
        if not topic or topic not in self.topic_field_vars:
            return
        for var in self.topic_field_vars[topic].values():
            var.set(False)

    def on_export_csv(self):
        """Export the selected topic and selected fields to a CSV file."""
        if self.bag is None or self.bag_path is None:
            messagebox.showwarning("Warning", "Please load a bag file first.")
            return

        topic = self.current_topic
        if not topic:
            messagebox.showwarning("Warning", "Please select a topic on the left.")
            return

        if topic not in self.topic_field_vars:
            messagebox.showwarning("Warning", "This topic has no inspected fields. Please re-select it.")
            return

        # Collect selected fields
        field_vars = self.topic_field_vars[topic]
        selected_fields = [name for name, var in field_vars.items() if var.get()]

        if not selected_fields:
            messagebox.showwarning("Warning", "Please select at least one field to export.")
            return

        # Suggest a default output file name
        base = os.path.splitext(os.path.basename(self.bag_path))[0]
        topic_safe = topic.strip("/").replace("/", "_")
        default_name = f"{base}_{topic_safe}.csv"

        out_path = filedialog.asksaveasfilename(
            title="Select output CSV file",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not out_path:
            return

        self.status_var.set("Exporting CSV...")
        self.progress_var.set(0)
        self.update_idletasks()

        try:
            self._export_topic_to_csv(topic, selected_fields, out_path)
            self.status_var.set(f"Export finished: {out_path}")
            messagebox.showinfo("Done", f"Export completed:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV:\n{e}")
            traceback.print_exc()
            self.status_var.set("Export failed.")
        finally:
            self.progress_var.set(0)

    def on_close(self):
        """Handle window close."""
        if self.bag is not None:
            self.bag.close()
        self.destroy()

    # ---------- Internal helpers ----------

    def _clear_fields_frame(self):
        for w in self.fields_frame.winfo_children():
            w.destroy()

    def _populate_fields_frame(self, topic):
        """Populate the right-hand field checkbox list for the given topic."""
        self._clear_fields_frame()
        fields = self.topic_fields.get(topic, [])
        field_vars = self.topic_field_vars.get(topic, {})

        for name in fields:
            var = field_vars[name]
            cb = ttk.Checkbutton(self.fields_frame, text=name, variable=var)
            cb.pack(anchor=tk.W, padx=4, pady=1)

    def _inspect_topic_fields(self, topic):
        """
        Read the first message of the topic, flatten it, and return a sorted
        list of field names. If there is no message, return an empty list.
        """
        fields = []
        for _, msg, _ in self.bag.read_messages(topics=[topic]):
            flat = flatten_message(msg)
            fields = sorted(flat.keys())
            break
        return fields

    def _export_topic_to_csv(self, topic, fields, out_path):
        """
        Core export logic:

        - 'time' column:
            * Use header.stamp.to_sec() if the option is enabled and it exists.
            * Otherwise, use the bag time 't.to_sec()'.
          Relative time is computed as (timestamp - bag_start_time).

        - Other columns:
            * Values are taken from flatten_message(msg)[field_name].
            * If a field is absent in a particular message, an empty string is written.
        """
        if self.bag_start_time is None:
            raise RuntimeError("Bag start time is unknown.")

        use_header = self.use_header_stamp_var.get()

        # Number of messages for progress reporting
        msg_count = 0
        if topic in self.topics_info:
            msg_count = self.topics_info[topic].message_count

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["time"] + list(fields)
            writer.writerow(header)

            written = 0
            for _, msg, t in self.bag.read_messages(topics=[topic]):
                # Time selection: header.stamp or bag time
                if use_header and hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                    try:
                        t_sec = msg.header.stamp.to_sec()
                    except Exception:
                        t_sec = t.to_sec()
                else:
                    t_sec = t.to_sec()

                rel_t = t_sec - self.bag_start_time

                flat = flatten_message(msg)

                row = [f"{rel_t:.9f}"]
                for name in fields:
                    val = flat.get(name, "")
                    row.append(val)
                writer.writerow(row)

                written += 1
                if msg_count > 0 and written % 100 == 0:
                    progress = min(100.0, written * 100.0 / msg_count)
                    self.progress_var.set(progress)
                    self.status_var.set(f"Exporting: {written}/{msg_count}")
                    self.update_idletasks()

        self.progress_var.set(100.0)


def main():
    app = RosbagCSVExporterApp()
    app.mainloop()


if __name__ == "__main__":
    main()

