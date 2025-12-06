#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclei Counter — Klinik CLI (StarDist)
GUI ile uyumlu çekirdek:

- Params alanları GUI ile birebir (pre_enable, gamma, sd_use_csbdeep_norm, sd_target_tile, sd_auto_tiles, …)
- Ön-işleme: (opsiyonel) gamma → rolling-ball/tophat → Gaussian → CLAHE → normalize → StarDist
- StarDist eşikleri GUI'den geldiği gibi predict_instances'a verilir (prob_thresh, nms_thresh)
- Opsiyonel watershed split (pp_*), border temizliği, alan filtreleri
- Çıktılar: per-image CSV ve overlay: *_ws_overlay.png
"""

from __future__ import annotations
import os, sys, json, csv, re, argparse, traceback, inspect, math
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# görüntü / bilim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import io, exposure, morphology, measure, segmentation, color, filters, util
from skimage.morphology import disk
from scipy import ndimage as ndi
import tifffile as tiff

# TF / StarDist
import tensorflow as tf
from stardist.models import StarDist2D
from csbdeep.utils import normalize as csb_norm

# Opsiyonel (daha iyi metadata okuma)
try:
    from aicsimageio import AICSImage
    _HAS_AICS = True
except Exception:
    _HAS_AICS = False


# ===================== PARAMETRELER =====================

@dataclass
class Params:
    # IO
    input_path: str = "./input"
    output_dir: str = "./output"              # sadece summary için kullanılır
    file_glob: str = "*.tif"

    # Kanal ayarları (çok-kanallı veride DAPI/Signal seçimi)
    dapi_channel: int = 0
    signal_channel: int = 1                   # yoksa dapi_channel kullanılır

    # Piksel & ön-işleme
    pixel_size_um: Optional[float] = None     # None: metadata
    default_pixel_size_um: float = 0.2707
    pre_enable: bool = True                   # GUI ile uyumlu
    gamma: float = 0.90                       # GUI ile uyumlu
    rolling_ball_um: float = 12.0
    gaussian_sigma_px: float = 0.5
    use_clahe: bool = True
    clahe_clip_limit: float = 0.02

    # Alan & morfoloji (StarDist sonrası süzgeç)
    min_area_um2: float = 30.0
    max_area_um2: float = 520.0
    exclude_touching_border: bool = False

    save_debug_figs: bool = True
    csv_precision: int = 3

    # --- StarDist ---
    stardist_model: str = "2D_versatile_fluo"   # veya local model klasörü
    sd_tiles: Optional[List[int]] = None        # [rows, cols]; None/0 -> otomatik
    sd_prob: float = 0.30                       # GUI varsayılanı
    sd_nms: float = 0.55                        # GUI varsayılanı
    sd_overlap: Optional[int] = None            # destekleyen sürümlerde
    sd_use_csbdeep_norm: bool = True           # GUI varsayılanı
    sd_target_tile: int = 1024                 # GUI varsayılanı
    sd_auto_tiles: bool = True                 # GUI varsayılanı

    # Güvenlik / performans
    cpu_only: bool = True                      # GUI varsayılanı

    # --- Post-process split (opsiyonel) ---
    pp_split: bool = False
    pp_min_peak_dist_px: int = 6
    pp_area_um2: float = 90.0
    pp_ecc: float = 0.70
    pp_max_new_per_obj: int = 4


# ===================== TF / MODEL =====================

def configure_tf(cpu_only: bool = True):
    """GPU görünürlüğünü ayarla ve mümkünse memory growth etkinleştir (GUI ile aynı davranış)."""
    try:
        if cpu_only:
            tf.config.set_visible_devices([], "GPU")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            print("[TF] CPU-only mod.", flush=True)
        else:
            gpus = tf.config.list_physical_devices("GPU")
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception:
                    pass
            print(f"[TF] GPU'lar: {gpus if gpus else 'yok'}", flush=True)
    except Exception as e:
        print(f"[TF] yapılandırma atlandı: {e}", flush=True)


_MODEL_CACHE: Dict[str, StarDist2D] = {}

def get_stardist_model(name_or_path: str) -> StarDist2D:
    """Önceden yüklenen modeli döndür; yoksa yükle.
    - '2D_versatile_fluo' gibi pretrained adları için from_pretrained
    - Bir klasör yolu verilirse (…/basedir/name) oradan yükler
    """
    key = os.path.abspath(name_or_path) if os.path.isdir(name_or_path) else name_or_path
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if os.path.isdir(name_or_path):
        basedir = os.path.dirname(os.path.abspath(name_or_path))
        name = os.path.basename(os.path.abspath(name_or_path))
        model = StarDist2D(None, name=name, basedir=basedir)
        print(f"[StarDist] local model: basedir={basedir} name={name}", flush=True)
    else:
        model = StarDist2D.from_pretrained(name_or_path)
        print(f"[StarDist] pretrained: {name_or_path}", flush=True)

    _MODEL_CACHE[key] = model
    return model


# ===================== YARDIMCILAR =====================

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def _rescale_01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    vmin = np.nanmin(img); vmax = np.nanmax(img)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    x = (img - vmin) / (vmax - vmin + 1e-9)
    return np.clip(x, 0, 1).astype(np.float32, copy=False)

def _read_image_any(path: str) -> Tuple[np.ndarray, Dict]:
    """C, Z, Y, X sırasıyla 4D döndür (gerekirse genişlet)."""
    meta = {"source": path, "pixel_size_um": None, "reader": None}
    if _HAS_AICS:
        try:
            img = AICSImage(path)
            data = img.get_image_data("CZYX")
            meta["reader"] = "aicsimageio"
            try:
                ps = img.get_physical_pixel_size()
                if ps is not None and ps.Y is not None:
                    meta["pixel_size_um"] = float(ps.Y)
            except Exception:
                pass
            return np.asarray(data), meta
        except Exception:
            pass

    arr = tiff.imread(path)
    meta["reader"] = "tifffile"
    a = np.asarray(arr)
    if a.ndim == 2:            # YX
        a = a[None, None, ...]
    elif a.ndim == 3:          # CYX veya ZYX
        if a.shape[0] in (1,2,3,4,5,6):
            a = a[:, None, ...]  # C,1,Y,X
        else:
            a = a[None, ...]     # 1,Z,Y,X
    return a, meta

def _pixel_size_um(path: str, params: Params, meta: Dict) -> float:
    if params.pixel_size_um and params.pixel_size_um > 0:
        return float(params.pixel_size_um)
    if meta.get("pixel_size_um"):
        return float(meta["pixel_size_um"])
    return float(params.default_pixel_size_um)

def collect_paths(root: str, pattern: str) -> List[str]:
    root_p = Path(root)
    if root_p.is_file():
        return [str(root_p)] if root_p.match(pattern) else []

    hits: List[Path] = []
    for p in root_p.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".tif",".tiff",".ome.tif",".ome.tiff") and p.match(pattern):
            hits.append(p)
    # doğal sıralama: klasör numarası → T/C/Z → isim
    _NUMDIR = re.compile(r"^(\d+)[-_]{2}")
    _T = re.compile(r"[Tt](\d+)")
    _C = re.compile(r"[Cc](\d+)")
    _Z = re.compile(r"[Zz](\d+)")
    def _try(pat,s,d=10**9): 
        m=pat.search(s); return int(m.group(1)) if m else d
    def _seq(path: Path, d=10**9):
        for part in path.parts[::-1]:
            m=_NUMDIR.match(part)
            if m: return int(m.group(1))
        return d
    def _key(p: Path):
        return (_seq(p), _try(_T,p.name), _try(_C,p.name), _try(_Z,p.name), p.name.lower())
    hits.sort(key=_key)
    return [str(p) for p in hits]


# ===================== ÖN-İŞLEME & STARDIST =====================

def preprocess_channel(ch: np.ndarray, p: Params, px_um: float) -> np.ndarray:
    """GUI ile uyumlu ön-işleme (p.pre_enable kapalıysa yalnız normalize)."""
    x = ch.astype(np.float32, copy=False)
    x = _rescale_01(x)

    if not p.pre_enable:
        return x

    # gamma
    x = np.power(np.clip(x, 0, 1), p.gamma).astype(np.float32, copy=False)

    # rolling-ball (mümkünse) / white tophat
    try:
        from skimage.restoration import rolling_ball
        rad_px = max(1, int(round(p.rolling_ball_um / max(px_um, 1e-6))))
        bg = rolling_ball(x, radius=rad_px)
        x = x - bg; x[x < 0] = 0
    except Exception:
        se = disk(max(1, int(round(p.rolling_ball_um / max(px_um, 1e-6)))))
        x = morphology.white_tophat(x, se)

    # gaussian
    if p.gaussian_sigma_px > 0:
        x = filters.gaussian(x, sigma=p.gaussian_sigma_px, preserve_range=True)

    # CLAHE
    if p.use_clahe and p.clahe_clip_limit > 0:
        x = exposure.equalize_adapthist((np.clip(x,0,1)*255).astype(np.uint8),
                                        clip_limit=p.clahe_clip_limit)
        x = util.img_as_float32(x)

    return np.clip(x, 0, 1).astype(np.float32, copy=False)

def _auto_tiles(shape: Tuple[int,int], target_px: int) -> Optional[Tuple[int,int]]:
    H, W = shape
    r = max(1, int(math.ceil(H / max(1, target_px))))
    c = max(1, int(math.ceil(W / max(1, target_px))))
    return None if (r==1 and c==1) else (r, c)

def _predict_instances_compat(model: StarDist2D, img: np.ndarray,
                              *, n_tiles: Optional[Tuple[int,int]],
                              prob_thresh: Optional[float],
                              nms_thresh: Optional[float],
                              overlap_px: Optional[int]):
    kw = {}
    if n_tiles is not None: kw["n_tiles"] = n_tiles
    if prob_thresh is not None: kw["prob_thresh"] = float(prob_thresh)
    if nms_thresh is not None: kw["nms_thresh"] = float(nms_thresh)
    sig = inspect.signature(model.predict_instances)
    if overlap_px is not None:
        if "overlap" in sig.parameters: kw["overlap"] = overlap_px
        elif "overlap_label" in sig.parameters: kw["overlap_label"] = overlap_px
    return model.predict_instances(img, **kw)

def segment_nuclei_stardist(dapi_2d: np.ndarray, p: Params) -> np.ndarray:
    """2D DAPI görüntüsünden StarDist instance labels döndürür."""
    model = get_stardist_model(p.stardist_model)

    # normalize (StarDist girişi)
    x = dapi_2d.astype(np.float32, copy=False)
    if p.sd_use_csbdeep_norm:
        x = csb_norm(x, 1, 99.8, axis=None)
    else:
        x = _rescale_01(x)

    # tiles
    if p.sd_tiles and len(p.sd_tiles) == 2 and max(p.sd_tiles) > 0:
        n_tiles = (int(p.sd_tiles[0]), int(p.sd_tiles[1]))
    elif p.sd_auto_tiles and p.sd_target_tile > 0:
        n_tiles = _auto_tiles(x.shape[:2], p.sd_target_tile)
    else:
        n_tiles = None

    labels, _details = _predict_instances_compat(
        model, x,
        n_tiles=n_tiles,
        prob_thresh=p.sd_prob,
        nms_thresh=p.sd_nms,
        overlap_px=p.sd_overlap
    )

    if p.exclude_touching_border:
        labels = segmentation.clear_border(labels)
        labels = measure.label(labels > 0)

    return labels


# ===================== POST-SPLIT + ÖLÇÜM =====================

def split_touching(labels: np.ndarray,
                   *, min_peak_dist: int, area_thr_um2: float, ecc_thr: float,
                   min_area_um2: float, px_area_um2: float, max_new_per_obj: int) -> np.ndarray:
    """Büyük/uzamış objeleri distance-based watershed ile böl."""
    if labels.size == 0 or labels.max() == 0:
        return labels

    lab = labels.copy()
    area_thr_px = int(np.ceil(area_thr_um2 / px_area_um2))
    min_area_px = int(np.ceil(min_area_um2 / px_area_um2))
    props = measure.regionprops(lab)
    cur = lab.max() + 1

    for r in props:
        if r.area < area_thr_px and (r.eccentricity is None or r.eccentricity < ecc_thr):
            continue
        rr, cc = r.slice
        mask = (lab[rr, cc] == r.label)
        if mask.sum() < 5:
            continue

        dist = ndi.distance_transform_edt(mask)
        peaks = morphology.local_maxima(dist) & mask
        if peaks.sum() < 2:
            continue

        # en kuvvetli tepe sayısını sınırla
        ys, xs = np.nonzero(peaks); vals = dist[ys, xs]
        if len(vals) > max_new_per_obj:
            keep_idx = np.argsort(vals)[-max_new_per_obj:]
            sel = np.zeros_like(peaks); sel[ys[keep_idx], xs[keep_idx]] = True
            markers = measure.label(sel)
        else:
            markers = measure.label(peaks)

        ws = segmentation.watershed(-dist, markers, mask=mask)
        if ws.max() < 2:
            continue

        # çok küçük parçaları ele
        good = np.zeros(ws.max()+1, bool)
        for k in range(1, ws.max()+1):
            if (ws == k).sum() >= min_area_px:
                good[k] = True
        if good.sum() < 2:
            continue

        lab[rr, cc][mask] = 0
        for k in range(1, ws.max()+1):
            if not good[k]: continue
            lab[rr, cc][ws == k] = cur; cur += 1

    lab, _, _ = segmentation.relabel_sequential(lab)
    return lab

def quantify_signal(labels: np.ndarray, signal_stack: np.ndarray) -> pd.DataFrame:
    sig_sum = signal_stack.sum(axis=0).astype(np.float64)
    rows=[]
    for p in measure.regionprops(labels, intensity_image=sig_sum):
        rows.append({
            "nucleus_id": int(p.label),
            "area_px": int(p.area),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
            "mean_signal": float(p.mean_intensity),
            "integrated_signal": float(p.mean_intensity * p.area),
        })
    return pd.DataFrame(rows)


# ===================== ÇIKIŞLAR =====================

def quicklook_overlay(dapi_2d: np.ndarray, labels: np.ndarray, out_png: str):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    ax.imshow(dapi_2d, cmap="gray")
    ax.imshow(color.label2rgb(labels, bg_label=0, alpha=0.25))
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ===================== ANA İŞLEV =====================

def process_one(path: str, params: Params, out_dir: Optional[str]=None):
    """
    Tek bir görüntüyü işler ve (count, stem, per_image_csv, overlay_png) döndürür.
    Çıktılar <out_dir> içine yazılır. (GUI Worker bunu çağırır.)
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(path), "output")
    ensure_dir(out_dir)

    arr, meta = _read_image_any(path)
    C,Z,Y,X = arr.shape
    px_um = _pixel_size_um(path, params, meta)

    dapi_idx = params.dapi_channel if params.dapi_channel < C else 0
    sig_idx  = params.signal_channel if params.signal_channel < C else dapi_idx
    dapi_3d, signal_3d = arr[dapi_idx], arr[sig_idx]

    # 2D giriş (MIP)
    dapi_mip  = np.max(dapi_3d, axis=0).astype(np.float32, copy=False)

    # Ön-işleme
    dapi_prep = preprocess_channel(dapi_mip, params, px_um)

    # StarDist
    labels = segment_nuclei_stardist(dapi_prep, params)

    # Post-split (opsiyonel)
    if params.pp_split:
        labels = split_touching(
            labels,
            min_peak_dist=params.pp_min_peak_dist_px,
            area_thr_um2=params.pp_area_um2,
            ecc_thr=params.pp_ecc,
            min_area_um2=params.min_area_um2,
            px_area_um2=(px_um**2),
            max_new_per_obj=params.pp_max_new_per_obj
        )

    # Alan filtreleri (µm²)
    min_px = int(round(params.min_area_um2 / (px_um**2)))
    max_px = int(round(params.max_area_um2 / (px_um**2))) if params.max_area_um2 > 0 else 0
    if min_px > 1 or max_px > 0:
        lab = labels.copy()
        keep = np.zeros(lab.max()+1, bool)
        for r in measure.regionprops(lab):
            ok = (r.area >= max(1, min_px)) and (max_px == 0 or r.area <= max_px)
            keep[r.label] = ok
        labels = lab * keep[lab]
        labels = measure.label(labels > 0)

    # Ölçüm
    df = quantify_signal(labels, signal_3d)
    df["area_um2"] = df["area_px"] * (px_um ** 2)
    df = df[["nucleus_id","area_px","area_um2","centroid_y","centroid_x","mean_signal","integrated_signal"]]

    stem = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(out_dir, f"{stem}_per_image.csv")
    df.round(params.csv_precision).to_csv(csv_path, index=False)

    overlay_path = os.path.join(out_dir, f"{stem}_ws_overlay.png")
    if params.save_debug_figs:
        quicklook_overlay(dapi_prep, labels, overlay_path)

    # meta/debug
    meta_out = {"reader": meta.get("reader"), "pixel_size_um": px_um,
                "shape_CZYX": [int(C), int(Z), int(Y), int(X)],
                "params": asdict(params), "input": path,
                "outputs": {"csv": csv_path, "overlay": overlay_path if params.save_debug_figs else None}}
    with open(os.path.join(out_dir, f"{stem}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    return int(labels.max()), stem, csv_path, (overlay_path if os.path.exists(overlay_path) else None)


# ===================== PIPELINE & CLI =====================

def run_pipeline(params: Params, dry_run: bool=False) -> int:
    ensure_dir(params.output_dir)
    paths = collect_paths(params.input_path, params.file_glob)
    if not paths:
        print("[INFO] İşlenecek dosya bulunamadı.")
        return 0

    print(f"[INFO] Toplam {len(paths)} dosya bulundu.")
    if dry_run:
        for i, pth in enumerate(paths, start=1):
            print(f"[ORDER] {i:04d} -> {pth}")
        print("[DRY] Sadece sıralama gösterildi.")
        return 0

    # TF & model
    configure_tf(cpu_only=params.cpu_only)
    _ = get_stardist_model(params.stardist_model)

    summary_rows = []
    total = len(paths)
    for i, pth in enumerate(paths, start=1):
        try:
            out_dir = os.path.join(os.path.dirname(pth), "output")
            count, stem, per_csv, overlay = process_one(pth, params, out_dir=out_dir)
            summary_rows.append({
                "processing_index": i,
                "image": stem,
                "folder": os.path.dirname(pth),
                "nuclei_count": int(count),
                "per_image_csv": per_csv
            })
            print(f"[{i:04d}/{total}] OK {pth} → {per_csv}  nuclei={count}")
        except Exception as e:
            print(f"[{i:04d}/{total}] ERROR {pth}: {e}")
            traceback.print_exc()
            summary_rows.append({
                "processing_index": i,
                "image": os.path.basename(pth),
                "folder": os.path.dirname(pth),
                "nuclei_count": None,
                "per_image_csv": "ERROR"
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows, columns=["processing_index","image","folder","nuclei_count","per_image_csv"])
        summary_path = os.path.join(params.output_dir, "_summary_counts.csv")
        ensure_dir(params.output_dir)
        df.to_csv(summary_path, index=False)
        print(f"[SUMMARY] {len(df)} görüntü işlendi.")
        print(f"[SUMMARY CSV] {summary_path}")
    return len(summary_rows)


# ===================== ARGÜMANLAR =====================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Nuclei Counter — Klinik CLI (StarDist)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("-i","--input", dest="input_path", default=Params.input_path)
    ap.add_argument("-o","--output", dest="output_dir", default=Params.output_dir)
    ap.add_argument("--glob", dest="file_glob", default=Params.file_glob)

    ap.add_argument("--dapi-channel", type=int, default=Params.dapi_channel)
    ap.add_argument("--signal-channel", type=int, default=Params.signal_channel)

    ap.add_argument("--pixel-size-um", type=float, default=-1.0, help="Piksel (µm). -1: metadata/varsayılan")
    ap.add_argument("--default-pixel-um", type=float, default=Params.default_pixel_size_um)

    ap.add_argument("--pre-enable", dest="pre_enable", action="store_true", default=Params.pre_enable)
    ap.add_argument("--no-pre", dest="pre_enable", action="store_false")
    ap.add_argument("--gamma", type=float, default=Params.gamma)
    ap.add_argument("--rolling-ball-um", type=float, default=Params.rolling_ball_um)
    ap.add_argument("--gaussian-sigma-px", type=float, default=Params.gaussian_sigma_px)
    ap.add_argument("--clahe", dest="use_clahe", action="store_true", default=Params.use_clahe)
    ap.add_argument("--no-clahe", dest="use_clahe", action="store_false")
    ap.add_argument("--clahe-clip", type=float, default=Params.clahe_clip_limit)

    ap.add_argument("--min-area-um2", type=float, default=Params.min_area_um2)
    ap.add_argument("--max-area-um2", type=float, default=Params.max_area_um2)
    ap.add_argument("--exclude-border", dest="exclude_touching_border", action="store_true", default=Params.exclude_touching_border)
    ap.add_argument("--include-border", dest="exclude_touching_border", action="store_false")

    ap.add_argument("--debug-figs", dest="save_debug_figs", action="store_true", default=Params.save_debug_figs)
    ap.add_argument("--no-debug-figs", dest="save_debug_figs", action="store_false")
    ap.add_argument("--csv-precision", type=int, default=Params.csv_precision)

    # StarDist
    ap.add_argument("--stardist-model", type=str, default=Params.stardist_model)
    ap.add_argument("--sd-tiles", nargs="+", type=int, default=None)
    ap.add_argument("--sd-prob", type=float, default=Params.sd_prob)
    ap.add_argument("--sd-nms", type=float, default=Params.sd_nms)
    ap.add_argument("--sd-overlap", type=int, default=None)
    ap.add_argument("--sd-csbdeep-norm", dest="sd_use_csbdeep_norm", action="store_true", default=Params.sd_use_csbdeep_norm)
    ap.add_argument("--no-sd-csbdeep-norm", dest="sd_use_csbdeep_norm", action="store_false")
    ap.add_argument("--sd-target-tile", type=int, default=Params.sd_target_tile)
    ap.add_argument("--no-auto-tiles", dest="sd_auto_tiles", action="store_false", default=Params.sd_auto_tiles)

    # Performans
    ap.add_argument("--cpu-only", dest="cpu_only", action="store_true", default=Params.cpu_only)

    # Post-split
    pp = ap.add_argument_group("Postprocess splitting (opsiyonel)")
    pp.add_argument("--pp-split", action="store_true", default=Params.pp_split)
    pp.add_argument("--pp-min-peak-dist-px", type=int, default=Params.pp_min_peak_dist_px)
    pp.add_argument("--pp-area-um2", type=float, default=Params.pp_area_um2)
    pp.add_argument("--pp-ecc", type=float, default=Params.pp_ecc)
    pp.add_argument("--pp-max-new-per-obj", type=int, default=Params.pp_max_new_per_obj)

    # Konfig
    ap.add_argument("-c","--config", dest="config_json", default=None, help="JSON konfig dosyası (Params anahtarları)")
    ap.add_argument("--save-config", dest="save_config", default=None, help="Geçerli parametreleri JSON’a kaydet ve çık")
    ap.add_argument("-n","--dry-run", dest="dry_run", action="store_true", help="Sadece sıralamayı yaz, işlem yapma")
    return ap


def params_from_args(args: argparse.Namespace) -> Params:
    base = {}
    if args.config_json:
        with open(args.config_json, "r") as f:
            cfg = json.load(f)
        known = {f.name for f in fields(Params)}
        base = {k:v for k,v in cfg.items() if k in known}

    p = Params(**base)
    p.input_path = args.input_path
    p.output_dir = args.output_dir
    p.file_glob = args.file_glob

    p.dapi_channel = args.dapi_channel
    p.signal_channel = args.signal_channel

    p.pixel_size_um = None if args.pixel_size_um is None or args.pixel_size_um < 0 else float(args.pixel_size_um)
    p.default_pixel_size_um = args.default_pixel_um

    p.pre_enable = args.pre_enable
    p.gamma = args.gamma
    p.rolling_ball_um = args.rolling_ball_um
    p.gaussian_sigma_px = args.gaussian_sigma_px
    p.use_clahe = args.use_clahe
    p.clahe_clip_limit = args.clahe_clip

    p.min_area_um2 = args.min_area_um2
    p.max_area_um2 = args.max_area_um2
    p.exclude_touching_border = args.exclude_touching_border

    p.save_debug_figs = args.save_debug_figs
    p.csv_precision = args.csv_precision

    p.stardist_model = args.stardist_model
    p.sd_tiles = args.sd_tiles
    p.sd_prob = args.sd_prob
    p.sd_nms = args.sd_nms
    p.sd_overlap = args.sd_overlap
    p.sd_use_csbdeep_norm = args.sd_use_csbdeep_norm
    p.sd_target_tile = args.sd_target_tile
    p.sd_auto_tiles = args.sd_auto_tiles

    p.cpu_only = args.cpu_only

    p.pp_split = args.pp_split
    p.pp_min_peak_dist_px = args.pp_min_peak_dist_px
    p.pp_area_um2 = args.pp_area_um2
    p.pp_ecc = args.pp_ecc
    p.pp_max_new_per_obj = args.pp_max_new_per_obj

    return p


# ===================== MAIN =====================

def main():
    ap = build_argparser()
    args = ap.parse_args()
    params = params_from_args(args)

    if args.save_config:
        with open(args.save_config, "w", encoding="utf-8") as f:
            json.dump(asdict(params), f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] Kaydedildi: {args.save_config}")
        return

    try:
        run_pipeline(params, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Kullanıcı tarafından durduruldu.")
        sys.exit(130)


if __name__ == "__main__":
    main()
