#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclei Counter — Klinik GUI (PySide6) — Mirror output yapısı
Gereklilikler: PySide6 numpy pandas matplotlib scikit-image tifffile aicsimageio csbdeep stardist
"""

import os
import json
from dataclasses import asdict
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

# Yerel CLI modülü (sende zaten var)
from nuclei_cli_stardist import (
    Params, configure_tf, collect_paths, process_one, get_stardist_model
)

APP_TITLE = "Nuclei Counter — Klinik GUI"


# ---------- Worker ----------

class Worker(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    log = QtCore.Signal(str)
    row_ready = QtCore.Signal(dict)     # {idx,image,folder,count,csv,overlay}
    preview = QtCore.Signal(str)
    finished_ok = QtCore.Signal()
    finished_err = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, params: Params, parent=None):
        super().__init__(parent)
        self.params = params
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        try:
            # TF ve model hazırla
            configure_tf(cpu_only=self.params.cpu_only)
            _ = get_stardist_model(self.params.stardist_model)

            # Girdi yollarını topla
            paths = collect_paths(self.params.input_path, self.params.file_glob)
            total = len(paths)
            if total == 0:
                self.log.emit("[INFO] İşlenecek dosya bulunamadı.")
                self.finished_ok.emit()
                return

            # Output kökü: Summary output alanı varsa orası, yoksa input_root/output
            input_root = os.path.abspath(self.params.input_path)
            preferred_out = (self.params.output_dir or "").strip()
            output_root = os.path.abspath(preferred_out) if preferred_out else os.path.join(input_root, "output")
            os.makedirs(output_root, exist_ok=True)

            self.log.emit(f"[INFO] {total} dosya bulundu. Çıkış kökü: {output_root}")
            self.log.emit("[INFO] Alt klasörler input yapısı ile MIRROR biçimde oluşturulacak.")

            cancelled = False
            for i, pth in enumerate(paths, start=1):
                if self._stop:
                    cancelled = True
                    self.log.emit("[INFO] Kullanıcı tarafından iptal edildi.")
                    break

                # ------- MIRROR: görüntünün bulunduğu klasörün input_root'a göre görece yolu
                img_dir = os.path.dirname(os.path.abspath(pth))
                try:
                    # img_dir input_root'un içinde mi?
                    common = os.path.commonpath([input_root, img_dir])
                    if common != input_root:
                        # Güvenli tarafta kal: doğrudan output_root'a yaz
                        rel = ""
                    else:
                        rel = os.path.relpath(img_dir, input_root)
                except Exception:
                    rel = ""

                out_dir = os.path.join(output_root, rel)
                os.makedirs(out_dir, exist_ok=True)
                # -------

                try:
                    count, stem, csv_path, overlay_path = process_one(
                        pth, self.params, out_dir=out_dir
                    )
                    self.row_ready.emit({
                        "idx": i,
                        "image": stem,
                        "folder": out_dir,         # tabloya gerçek çıkış klasörünü göster
                        "count": count,
                        "csv": csv_path,
                        "overlay": overlay_path or ""
                    })
                    if overlay_path and os.path.exists(overlay_path):
                        self.preview.emit(overlay_path)

                    self.log.emit(f"[{i:04d}/{total}] OK {pth}  →  {out_dir}  nuclei={count}")
                except Exception as e:
                    self.row_ready.emit({
                        "idx": i, "image": os.path.basename(pth), "folder": out_dir,
                        "count": -1, "csv": "ERROR", "overlay": ""
                    })
                    self.log.emit(f"[{i:04d}/{total}] ERROR {pth}: {e}")

                self.progress.emit(i, total)

            if cancelled: self.cancelled.emit()
            else: self.finished_ok.emit()

        except Exception as e:
            self.finished_err.emit(str(e))


# ---------- GUI ----------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1180, 740)

        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        h = QtWidgets.QHBoxLayout(cw)

        # Sol panel
        left = QtWidgets.QVBoxLayout(); h.addLayout(left, 0)

        # IO
        gb_io = QtWidgets.QGroupBox("Girdi / Çıktı"); left.addWidget(gb_io)
        f = QtWidgets.QFormLayout(gb_io)

        self.edit_input = QtWidgets.QLineEdit()
        btn_in = QtWidgets.QPushButton("Klasör Seç"); btn_in.clicked.connect(self.pick_input)
        inrow = QtWidgets.QHBoxLayout(); inrow.addWidget(self.edit_input); inrow.addWidget(btn_in)
        f.addRow("Input root:", wrap(inrow))

        self.edit_output = QtWidgets.QLineEdit()
        btn_out = QtWidgets.QPushButton("Özet Çıktı Klasörü"); btn_out.clicked.connect(self.pick_output)
        outrow = QtWidgets.QHBoxLayout(); outrow.addWidget(self.edit_output); outrow.addWidget(btn_out)
        f.addRow("Summary output:", wrap(outrow))

        self.edit_glob = QtWidgets.QLineEdit("*.tif")
        f.addRow("Dosya deseni:", self.edit_glob)

        # StarDist
        gb_sd = QtWidgets.QGroupBox("StarDist"); left.addWidget(gb_sd)
        g = QtWidgets.QGridLayout(gb_sd)

        self.combo_model = QtWidgets.QComboBox(); self.combo_model.setEditable(True)
        self.combo_model.addItems(["2D_versatile_fluo", "2D_versatile_he", "2D_demo"])

        self.spin_prob = dblspin(0.30, 0.0, 1.0, step=0.01, decimals=2)
        self.spin_nms  = dblspin(0.30, 0.0, 1.0, step=0.01, decimals=2)

        self.spin_tile_r = intspin(0, 0, 32)
        self.spin_tile_c = intspin(0, 0, 32)
        self.spin_target_tile = intspin(768, 64, 4096)

        self.chk_csbdeep_norm = QtWidgets.QCheckBox("csbdeep.normalize(1,99.8)")
        self.chk_exclude_border = QtWidgets.QCheckBox("Sınırdaki objeleri hariç tut")
        self.chk_cpu_only = QtWidgets.QCheckBox("CPU-only (GPU/MPS kapat)")

        r = 0
        g.addWidget(QtWidgets.QLabel("Model:"), r,0); g.addWidget(self.combo_model, r,1,1,3); r+=1
        g.addWidget(QtWidgets.QLabel("prob thresh:"), r,0); g.addWidget(self.spin_prob, r,1)
        g.addWidget(QtWidgets.QLabel("nms thresh:"),  r,2); g.addWidget(self.spin_nms,  r,3); r+=1
        g.addWidget(QtWidgets.QLabel("Tiles (R,C 0=auto):"), r,0); g.addWidget(self.spin_tile_r, r,1); g.addWidget(self.spin_tile_c, r,2)
        g.addWidget(QtWidgets.QLabel("Target tile ~px:"), r,3); g.addWidget(self.spin_target_tile, r,4); r+=1
        g.addWidget(self.chk_csbdeep_norm, r,0,1,3); g.addWidget(self.chk_cpu_only, r,3,1,2); r+=1
        g.addWidget(self.chk_exclude_border, r,0,1,5); r+=1

        # Ön-işleme
        gb_pre = QtWidgets.QGroupBox("Ön-işleme"); left.addWidget(gb_pre)
        fpre = QtWidgets.QFormLayout(gb_pre)

        self.chk_pre_enable = QtWidgets.QCheckBox("Ön-işlemeyi etkinleştir (gamma + tophat + Gaussian + CLAHE)")
        self.spin_rb_um     = dblspin(10.0, 0.0, 50.0, step=0.5, decimals=1)
        self.spin_gauss_px  = dblspin(1.0,  0.0, 5.0,  step=0.1, decimals=1)
        self.spin_clahe     = dblspin(0.012,0.001,0.05, step=0.001, decimals=3)
        self.spin_min_area  = dblspin(30.0, 1.0, 500.0, step=1.0, decimals=1)
        self.spin_gamma     = dblspin(0.90, 0.50, 2.00, step=0.05, decimals=2)

        fpre.addRow(self.chk_pre_enable)
        fpre.addRow("Rolling-ball (µm):", self.spin_rb_um)
        fpre.addRow("Gaussian σ (px):",   self.spin_gauss_px)
        fpre.addRow("CLAHE clip_limit:",  self.spin_clahe)
        fpre.addRow("Min alan (µm²):",    self.spin_min_area)
        fpre.addRow("Gamma:",             self.spin_gamma)

        # Post-split
        gb_pp = QtWidgets.QGroupBox("Birleşik çekirdeği ayır (opsiyonel)"); left.addWidget(gb_pp)
        f2 = QtWidgets.QFormLayout(gb_pp)
        self.chk_pp_split = QtWidgets.QCheckBox("Watershed ile böl")
        f2.addRow(self.chk_pp_split)
        self.spin_pp_min_peak = intspin(8, 1, 32)
        self.spin_pp_area_um2 = dblspin(120.0, 1.0, 5000.0, step=1.0, decimals=1)
        self.spin_pp_ecc = dblspin(0.88, 0.0, 0.999, step=0.01, decimals=2)
        self.spin_pp_max_new = intspin(3, 1, 12)
        f2.addRow("Min peak dist (px):", self.spin_pp_min_peak)
        f2.addRow("Alan eşiği üstü (µm²):", self.spin_pp_area_um2)
        f2.addRow("Eksen.trş:", self.spin_pp_ecc)
        f2.addRow("Maks yeni parça/obje:", self.spin_pp_max_new)

        # Alt butonlar
        btns = QtWidgets.QHBoxLayout(); left.addLayout(btns)
        self.btn_start = QtWidgets.QPushButton("▶ Başlat")
        self.btn_cancel = QtWidgets.QPushButton("■ İptal")
        self.btn_save = QtWidgets.QPushButton("Ayarları Kaydet")
        self.btn_load = QtWidgets.QPushButton("Ayar Yükle")
        btns.addWidget(self.btn_start); btns.addWidget(self.btn_cancel)
        btns.addStretch(1); btns.addWidget(self.btn_save); btns.addWidget(self.btn_load)

        # Orta (log)
        mid = QtWidgets.QVBoxLayout(); h.addLayout(mid, 1)
        self.text_log = QtWidgets.QPlainTextEdit(); self.text_log.setReadOnly(True)
        self.prog = QtWidgets.QProgressBar()
        mid.addWidget(self.text_log, 1); mid.addWidget(self.prog)

        # Sağ (tablo + önizleme)
        right = QtWidgets.QVBoxLayout(); h.addLayout(right, 1)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "image", "folder", "nuclei_count", "per_image_csv"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        right.addWidget(self.table, 1)

        gb_prev = QtWidgets.QGroupBox("Son görüntü (overlay)"); right.addWidget(gb_prev)
        vprev = QtWidgets.QVBoxLayout(gb_prev)
        self.lbl_prev = QtWidgets.QLabel(); self.lbl_prev.setMinimumSize(320, 240)
        self.lbl_prev.setAlignment(QtCore.Qt.AlignCenter); self.lbl_prev.setFrameShape(QtWidgets.QFrame.Box)
        vprev.addWidget(self.lbl_prev)

        # Varsayılanları yükle
        self.apply_presets()

        # Sinyaller
        self.btn_start.clicked.connect(self.start_clicked)
        self.btn_cancel.clicked.connect(self.cancel_clicked)
        self.btn_save.clicked.connect(self.save_settings)
        self.btn_load.clicked.connect(self.load_settings)

        self.worker: Optional[Worker] = None
        self._settings = QtCore.QSettings("OrtachLab", "NucleiCounterGUI")
        geom = self._settings.value("geom", None)
        if geom is not None: self.restoreGeometry(geom)

    # ---- UI yardımcıları ----
    def apply_presets(self):
        self.edit_input.setText("/Users/ortach/Desktop/sayim/export/Alaz_Il6")
        self.edit_output.setText("/Users/ortach/Desktop/sayim/output")
        self.edit_glob.setText("*.tif")

        self.combo_model.setCurrentText("2D_versatile_fluo")
        self.spin_prob.setValue(0.30)
        self.spin_nms.setValue(0.30)
        self.spin_tile_r.setValue(0); self.spin_tile_c.setValue(0)
        self.spin_target_tile.setValue(768)
        self.chk_csbdeep_norm.setChecked(True)
        self.chk_exclude_border.setChecked(False)
        self.chk_cpu_only.setChecked(True)

        self.chk_pre_enable.setChecked(True)
        self.spin_rb_um.setValue(10.0)
        self.spin_gauss_px.setValue(1.0)
        self.spin_clahe.setValue(0.012)
        self.spin_min_area.setValue(30.0)
        self.spin_gamma.setValue(0.90)

        self.chk_pp_split.setChecked(False)
        self.spin_pp_min_peak.setValue(8)
        self.spin_pp_area_um2.setValue(120.0)
        self.spin_pp_ecc.setValue(0.88)
        self.spin_pp_max_new.setValue(3)

    def pick_input(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Input klasörü seç")
        if d: self.edit_input.setText(d)

    def pick_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Özet çıktı klasörü seç")
        if d: self.edit_output.setText(d)

    def params_from_ui(self) -> Params:
        p = Params()
        p.input_path = self.edit_input.text().strip()
        p.output_dir = self.edit_output.text().strip()
        p.file_glob = self.edit_glob.text().strip()

        p.stardist_model = self.combo_model.currentText().strip()
        p.sd_prob = float(self.spin_prob.value())
        p.sd_nms = float(self.spin_nms.value())

        r, c = int(self.spin_tile_r.value()), int(self.spin_tile_c.value())
        p.sd_tiles = None if (r == 0 and c == 0) else [r, c]
        p.sd_target_tile = int(self.spin_target_tile.value())
        p.sd_auto_tiles = True
        p.sd_use_csbdeep_norm = self.chk_csbdeep_norm.isChecked()
        p.exclude_touching_border = self.chk_exclude_border.isChecked()
        p.cpu_only = self.chk_cpu_only.isChecked()

        # Ön-işleme
        p.pre_enable = self.chk_pre_enable.isChecked()
        p.rolling_ball_um = float(self.spin_rb_um.value())
        p.gaussian_sigma_px = float(self.spin_gauss_px.value())
        p.use_clahe = True
        p.clahe_clip_limit = float(self.spin_clahe.value())
        p.min_area_um2 = float(self.spin_min_area.value())
        p.gamma = float(self.spin_gamma.value())

        # Post-split
        p.pp_split = self.chk_pp_split.isChecked()
        p.pp_min_peak_dist_px = int(self.spin_pp_min_peak.value())
        p.pp_area_um2 = float(self.spin_pp_area_um2.value())
        p.pp_ecc = float(self.spin_pp_ecc.value())
        p.pp_max_new_per_obj = int(self.spin_pp_max_new.value())
        p.save_debug_figs = True
        p.max_area_um2 = 520.0
        return p

    def start_clicked(self):
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "Zaten çalışıyor."); return
        p = self.params_from_ui()
        if not p.input_path or not os.path.isdir(p.input_path):
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "Geçerli bir input klasörü seçiniz."); return
        if p.output_dir and not os.path.isdir(p.output_dir):
            try: os.makedirs(p.output_dir, exist_ok=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, APP_TITLE, f"Output klasörü oluşturulamadı:\n{e}")
                return

        self.text_log.clear(); self.table.setRowCount(0); self.preview_show(None); self.prog.setValue(0)

        self.worker = Worker(p, self)
        self.worker.log.connect(self.log_append)
        self.worker.progress.connect(self.on_progress)
        self.worker.row_ready.connect(self.on_row_ready)
        self.worker.preview.connect(self.preview_show)
        self.worker.finished_ok.connect(lambda: self.log_append("[DONE] Bitti."))
        self.worker.cancelled.connect(lambda: self.log_append("[CANCELLED] İşlem iptal edildi."))
        self.worker.finished_err.connect(lambda msg: self.log_append(f"[ERROR] {msg}"))
        self.worker.start()

    def cancel_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop(); self.log_append("[INFO] Durduruluyor…")

    def log_append(self, s: str): self.text_log.appendPlainText(s)
    def on_progress(self, i: int, total: int):
        self.prog.setMaximum(total); self.prog.setValue(i)

    def on_row_ready(self, info: dict):
        row = self.table.rowCount(); self.table.insertRow(row)
        vals = [str(info.get("idx","")), info.get("image",""), info.get("folder",""),
                str(info.get("count","")), info.get("csv","")]
        for c, v in enumerate(vals):
            item = QtWidgets.QTableWidgetItem(v)
            if c in (0,3): item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, c, item)
        self.table.scrollToBottom()

    def preview_show(self, path: Optional[str]):
        if not path or not os.path.exists(path):
            self.lbl_prev.setPixmap(QtGui.QPixmap()); self.lbl_prev.setText("Önizleme yok"); return
        pm = QtGui.QPixmap(path)
        self.lbl_prev.setPixmap(pm.scaled(self.lbl_prev.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def save_settings(self):
        p = self.params_from_ui()
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Ayar kaydet", "stardist_gui_config.json", "JSON (*.json)")
        if not fn: return
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(asdict(p), f, indent=2, ensure_ascii=False)
        self.log_append(f"[CONFIG] Kaydedildi: {fn}")

    def load_settings(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Ayar yükle", "", "JSON (*.json)")
        if not fn: return
        with open(fn, "r", encoding="utf-8") as f: d = json.load(f)

        self.edit_input.setText(d.get("input_path", self.edit_input.text()))
        self.edit_output.setText(d.get("output_dir", self.edit_output.text()))
        self.edit_glob.setText(d.get("file_glob", self.edit_glob.text()))

        self.combo_model.setCurrentText(d.get("stardist_model", self.combo_model.currentText()))
        self.spin_prob.setValue(float(d.get("sd_prob", self.spin_prob.value())))
        self.spin_nms.setValue(float(d.get("sd_nms", self.spin_nms.value())))
        tiles = d.get("sd_tiles", None)
        if tiles and len(tiles) == 2:
            self.spin_tile_r.setValue(int(tiles[0])); self.spin_tile_c.setValue(int(tiles[1]))
        else:
            self.spin_tile_r.setValue(0); self.spin_tile_c.setValue(0)
        self.spin_target_tile.setValue(int(d.get("sd_target_tile", self.spin_target_tile.value())))
        self.chk_csbdeep_norm.setChecked(bool(d.get("sd_use_csbdeep_norm", self.chk_csbdeep_norm.isChecked())))
        self.chk_exclude_border.setChecked(bool(d.get("exclude_touching_border", self.chk_exclude_border.isChecked())))
        self.chk_cpu_only.setChecked(bool(d.get("cpu_only", self.chk_cpu_only.isChecked())))

        self.chk_pre_enable.setChecked(bool(d.get("pre_enable", self.chk_pre_enable.isChecked())))
        self.spin_rb_um.setValue(float(d.get("rolling_ball_um", self.spin_rb_um.value())))
        self.spin_gauss_px.setValue(float(d.get("gaussian_sigma_px", self.spin_gauss_px.value())))
        self.spin_clahe.setValue(float(d.get("clahe_clip_limit", self.spin_clahe.value())))
        self.spin_min_area.setValue(float(d.get("min_area_um2", self.spin_min_area.value())))
        self.spin_gamma.setValue(float(d.get("gamma", self.spin_gamma.value())))

        self.chk_pp_split.setChecked(bool(d.get("pp_split", self.chk_pp_split.isChecked())))
        self.spin_pp_min_peak.setValue(int(d.get("pp_min_peak_dist_px", self.spin_pp_min_peak.value())))
        self.spin_pp_area_um2.setValue(float(d.get("pp_area_um2", self.spin_pp_area_um2.value())))
        self.spin_pp_ecc.setValue(float(d.get("pp_ecc", self.spin_pp_ecc.value())))
        self.spin_pp_max_new.setValue(int(d.get("pp_max_new_per_obj", self.spin_pp_max_new.value())))
        self.log_append(f"[CONFIG] Yüklendi: {fn}")

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.stop(); self.worker.wait(5000)
        finally:
            self._settings.setValue("geom", self.saveGeometry()); self._settings.sync()
            super().closeEvent(e)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        pm = self.lbl_prev.pixmap()
        if pm:
            self.lbl_prev.setPixmap(pm.scaled(self.lbl_prev.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


# ---------- küçük yardımcılar ----------

def wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget(); w.setLayout(layout); return w

def dblspin(val, lo, hi, step=0.1, decimals=2):
    s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(decimals); s.setValue(val); return s

def intspin(val, lo, hi, step=1):
    s = QtWidgets.QSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setValue(val); return s


# ---------- main ----------

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
