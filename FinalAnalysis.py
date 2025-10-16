from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Plot axes and vowel set ----
F2_MIN, F2_MAX = 1000, 2250
F1_MIN, F1_MAX = 400, 800
LABEL_Y_OFFSET = 20
VOWELS = set(list("aAeEiIoOuUáéíóúÁÉÍÓÚɨɯ"))

def is_vowel_label(lbl: str) -> bool:
    if not lbl:
        return False
    return lbl.strip()[0] in VOWELS

# ----------------------------- I/O helpers -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---- DTW side ----

def read_word_metrics_csv(dtw_word_dir: Path) -> Dict[str, float]:
    out = {"bc_full": np.nan, "bd_full": np.nan, "bc_per_phone": np.nan, "bd_per_phone": np.nan}
    csv_path = dtw_word_dir / "word_metrics.csv"
    if not csv_path.exists():
        return out
    df = pd.read_csv(csv_path)
    df["mode"] = df["mode"].str.lower().str.strip()
    for mode, bc_key, bd_key in [("full", "bc_full", "bd_full"), ("per_phone", "bc_per_phone", "bd_per_phone")]:
        sub = df[df["mode"] == mode]
        if not sub.empty:
            out[bc_key] = float(pd.to_numeric(sub["bc"], errors="coerce").iloc[0])
            out[bd_key] = float(pd.to_numeric(sub["bd"], errors="coerce").iloc[0])
    return out


def read_phoneme_metrics_csv(dtw_word_dir: Path) -> pd.DataFrame:
    path = dtw_word_dir / "phoneme_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "word","seg_idx","r_lab","r_t0","r_t1","r_raw","r_pad",
            "t_lab","t_t0","t_t1","t_raw","t_pad","bc","bd"
        ])
    df = pd.read_csv(path)
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"r_lab","native_phone","native_label","real_label"}: ren[c] = "r_lab"
        if lc in {"t_lab","tts_phone","tts_label"}:     ren[c] = "t_lab"
        if lc in {"r_t0","native_t0","real_t0"}:       ren[c] = "r_t0"
        if lc in {"r_t1","native_t1","real_t1"}:       ren[c] = "r_t1"
        if lc in {"t_t0","tts_t0"}:                      ren[c] = "t_t0"
        if lc in {"t_t1","tts_t1"}:                      ren[c] = "t_t1"
        if lc in {"r_pad","native_padded"}:               ren[c] = "r_pad"
        if lc in {"t_pad","tts_padded"}:                  ren[c] = "t_pad"
    if ren:
        df = df.rename(columns=ren)
    for c in ["r_t0","r_t1","t_t0","t_t1","bc","bd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["r_pad","t_pad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if {"r_t0","r_t1"}.issubset(df.columns):
        df["real_dur_s"] = df["r_t1"] - df["r_t0"]
    else:
        df["real_dur_s"] = np.nan
    if {"t_t0","t_t1"}.issubset(df.columns):
        df["tts_dur_s"] = df["t_t1"] - df["t_t0"]
    else:
        df["tts_dur_s"] = np.nan
    return df

# ---- Praat side ----

def read_formant_tokens(praat_word_dir: Path, kind: str) -> pd.DataFrame:
    path = praat_word_dir / f"{kind} formants.csv"
    if not path.exists():
        return pd.DataFrame(columns=["phone","F1_Hz","F2_Hz","F3_Hz","source","word"]) 
    df = pd.read_csv(path)
    for col in ["F1_Hz","F2_Hz","F3_Hz"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source"] = kind
    return df


def normalize_formants_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["phone","F1_mean","F2_mean","F3_mean","N"]) 
    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith("f1_mean"): rename[c] = "F1_mean"
        if lc.startswith("f2_mean"): rename[c] = "F2_mean"
        if lc.startswith("f3_mean"): rename[c] = "F3_mean"
        if lc == "n":               rename[c] = "N"
    if rename:
        df = df.rename(columns=rename)
    for col in ["phone","F1_mean","F2_mean","N","F3_mean"]:
        if col not in df.columns:
            df[col] = np.nan
    return df[["phone","F1_mean","F2_mean","F3_mean","N"]]


def read_formant_summary(praat_word_dir: Path, kind: str) -> pd.DataFrame:
    path = praat_word_dir / f"{kind} formants_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["phone","F1_mean","F2_mean","F3_mean","N"]) 
    df = pd.read_csv(path)
    return normalize_formants_summary_columns(df)

# --------------------- Normalization helper ---------------------

def add_minmax_and_z_from_cost(df: pd.DataFrame, cost_col: str, out_prefix: str) -> pd.DataFrame:
    if df is None or df.empty or cost_col not in df.columns:
        return df
    out = df.copy()
    vals = pd.to_numeric(out[cost_col], errors="coerce")
    cmin, cmax = np.nanmin(vals), np.nanmax(vals)
    if np.isclose(cmin, cmax):
        out[f"{out_prefix}_minmax"] = 1.0
    else:
        out[f"{out_prefix}_minmax"] = 1.0 - (vals - cmin) / (cmax - cmin)
    mu, sd = np.nanmean(vals), np.nanstd(vals, ddof=0)
    out[f"{out_prefix}_z"] = 0.0 if np.isclose(sd, 0.0) else (vals - mu) / sd
    return out

# ---------------------------- Main pipeline ----------------------------

def aggregate(dtw_root: Path, praat_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    words_rows = []
    phone_acc = defaultdict(lambda: {
        "N_tokens": 0,
        "real_dur_sum": 0.0,
        "tts_dur_sum": 0.0,
        "bc_sum": 0.0,
        "bd_sum": 0.0,
        "pad_count": 0,
    })
    real_tokens_all, tts_tokens_all = [], []
    real_summ_all,  tts_summ_all  = [], []

    dtw_words = {p.name for p in dtw_root.iterdir() if p.is_dir()}
    praat_words = {p.name for p in praat_root.iterdir() if p.is_dir()}
    words = sorted(dtw_words | praat_words)

    for w in words:
        dtw_dir = dtw_root / w
        praat_dir = praat_root / w

        m = read_word_metrics_csv(dtw_dir)
        dfp = read_phoneme_metrics_csv(dtw_dir)
        if not dfp.empty:
            if {"r_lab","t_lab"}.issubset(dfp.columns):
                aligned = dfp[(dfp["r_lab"].astype(str) == dfp["t_lab"].astype(str))]
            else:
                aligned = dfp
            r_dur = aligned["real_dur_s"].sum() if "real_dur_s" in aligned else np.nan
            t_dur = aligned["tts_dur_s"].sum()  if "tts_dur_s"  in aligned else np.nan
        else:
            r_dur = np.nan
            t_dur = np.nan
        words_rows.append({
            "word": w,
            "real_dur_s": r_dur,
            "tts_dur_s": t_dur,
            "delta_dur_ms": (t_dur - r_dur) * 1000.0 if pd.notna(r_dur) and pd.notna(t_dur) else np.nan,
            "bc_full": m["bc_full"],
            "bd_full": m["bd_full"],
            "bc_per_phone": m["bc_per_phone"],
            "bd_per_phone": m["bd_per_phone"],
        })

        if not dfp.empty:
            if {"r_lab","t_lab"}.issubset(dfp.columns):
                aligned = dfp[(dfp["r_lab"].astype(str) == dfp["t_lab"].astype(str))].copy()
            else:
                aligned = dfp.copy()
            aligned = aligned.dropna(subset=["bc","bd"]) if {"bc","bd"}.issubset(aligned.columns) else aligned
            for _, r in aligned.iterrows():
                ph = str(r.get("r_lab", "")).strip()
                if not ph:
                    continue
                phone_acc[ph]["N_tokens"] += 1
                phone_acc[ph]["real_dur_sum"] += float(r.get("real_dur_s", np.nan)) if pd.notna(r.get("real_dur_s", np.nan)) else 0.0
                phone_acc[ph]["tts_dur_sum"]  += float(r.get("tts_dur_s",  np.nan)) if pd.notna(r.get("tts_dur_s",  np.nan)) else 0.0
                if pd.notna(r.get("bc", np.nan)):
                    phone_acc[ph]["bc_sum"] += float(r["bc"]) 
                if pd.notna(r.get("bd", np.nan)):
                    phone_acc[ph]["bd_sum"] += float(r["bd"]) 
                rp = int(r.get("r_pad", 0)) if pd.notna(r.get("r_pad", np.nan)) else 0
                tp = int(r.get("t_pad", 0)) if pd.notna(r.get("t_pad", np.nan)) else 0
                phone_acc[ph]["pad_count"] += 1 if (rp or tp) else 0

        if praat_dir.exists():
            rt = read_formant_tokens(praat_dir, "Real");  rt["word"] = w
            tt = read_formant_tokens(praat_dir, "TTS");   tt["word"] = w
            rs = read_formant_summary(praat_dir, "Real"); rs["word"] = w
            ts = read_formant_summary(praat_dir, "TTS");  ts["word"] = w
            if not rt.empty: real_tokens_all.append(rt)
            if not tt.empty: tts_tokens_all.append(tt)
            if not rs.empty: real_summ_all.append(rs)
            if not ts.empty: tts_summ_all.append(ts)

    words_df = pd.DataFrame(words_rows)

    phone_rows = []
    for ph, s in sorted(phone_acc.items()):
        N = s["N_tokens"]
        if N <= 0:
            continue
        phone_rows.append({
            "phone": ph,
            "N_tokens": N,
            "real_dur_mean_s": s["real_dur_sum"]/N if N else np.nan,
            "tts_dur_mean_s":  s["tts_dur_sum"]/N  if N else np.nan,
            "delta_dur_ms_mean": ((s["tts_dur_sum"]-s["real_dur_sum"]) * 1000.0 / N) if N else np.nan,
            "bc_mean": s["bc_sum"]/N if N else np.nan,
            "bd_mean": s["bd_sum"]/N if N else np.nan,
            "pad_rate": (s["pad_count"]/N) if N else np.nan,
        })
    timing_df = pd.DataFrame(phone_rows)

    # ---- Aggregate formants across words ----
    def summarize_formants_across_words(summ_list: list[pd.DataFrame]) -> pd.DataFrame:
        if not summ_list:
            return pd.DataFrame(columns=["phone","F1_mean","F2_mean","F3_mean","N"]) 
        tmp = pd.concat(summ_list, ignore_index=True)
        for c in ["F1_mean","F2_mean","F3_mean","N"]:
            if c not in tmp.columns:
                tmp[c] = np.nan
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp["N"] = tmp["N"].fillna(0.0)
        tmp["w"] = tmp["N"].clip(lower=0.0)
        sum_w = (tmp.groupby("phone", as_index=False)["N"].sum())
        denom = sum_w.set_index("phone")["N"].replace(0, np.nan)
        f1 = (tmp["F1_mean"].fillna(0.0) * tmp["w"]).groupby(tmp["phone"]).sum() / denom
        f2 = (tmp["F2_mean"].fillna(0.0) * tmp["w"]).groupby(tmp["phone"]).sum() / denom
        f3 = (tmp["F3_mean"].fillna(0.0) * tmp["w"]).groupby(tmp["phone"]).sum() / denom
        out = pd.DataFrame({
            "phone": sum_w["phone"].values,
            "F1_mean": f1.reindex(sum_w["phone"]).values,
            "F2_mean": f2.reindex(sum_w["phone"]).values,
            "F3_mean": f3.reindex(sum_w["phone"]).values,
            "N":       sum_w["N"].values,
        })
        return out[["phone","F1_mean","F2_mean","F3_mean","N"]]

    def summarize_tokens(tokens_list: list[pd.DataFrame]) -> pd.DataFrame:
        if not tokens_list:
            return pd.DataFrame(columns=["phone","F1_mean","F2_mean","F3_mean","N"]) 
        tmp = pd.concat(tokens_list, ignore_index=True)
        for col in ["F1_Hz","F2_Hz","F3_Hz"]:
            if col not in tmp.columns:
                tmp[col] = np.nan
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        grp = tmp.groupby("phone")
        out = grp.agg(
            F1_mean=("F1_Hz","mean"),
            F2_mean=("F2_Hz","mean"),
            F3_mean=("F3_Hz","mean"),
            N=("phone","size")
        ).reset_index()
        return out[["phone","F1_mean","F2_mean","F3_mean","N"]]

    real_agg_summ = summarize_formants_across_words(real_summ_all)
    tts_agg_summ  = summarize_formants_across_words(tts_summ_all)
    real_agg_tok  = summarize_tokens(real_tokens_all)
    tts_agg_tok   = summarize_tokens(tts_tokens_all)

    real_agg = pd.merge(real_agg_summ, real_agg_tok, on="phone", how="outer", suffixes=("_summ","_tok"))
    for c in ["F1_mean","F2_mean","F3_mean","N"]:
        if f"{c}_summ" in real_agg.columns and f"{c}_tok" in real_agg.columns:
            real_agg[c] = real_agg[f"{c}_summ"].combine_first(real_agg[f"{c}_tok"]) 
        elif f"{c}_summ" in real_agg.columns:
            real_agg[c] = real_agg[f"{c}_summ"]
        elif f"{c}_tok" in real_agg.columns:
            real_agg[c] = real_agg[f"{c}_tok"]
    real_agg = real_agg[["phone","F1_mean","F2_mean","F3_mean","N"]]

    tts_agg = pd.merge(tts_agg_summ, tts_agg_tok, on="phone", how="outer", suffixes=("_summ","_tok"))
    for c in ["F1_mean","F2_mean","F3_mean","N"]:
        if f"{c}_summ" in tts_agg.columns and f"{c}_tok" in tts_agg.columns:
            tts_agg[c] = tts_agg[f"{c}_summ"].combine_first(tts_agg[f"{c}_tok"]) 
        elif f"{c}_summ" in tts_agg.columns:
            tts_agg[c] = tts_agg[f"{c}_summ"]
        elif f"{c}_tok" in tts_agg.columns:
            tts_agg[c] = tts_agg[f"{c}_tok"]
    tts_agg = tts_agg[["phone","F1_mean","F2_mean","F3_mean","N"]]

    return words_df, timing_df, real_agg, tts_agg


def save_outputs(dtw_root: Path, praat_root: Path, final_root: Path) -> None:
    ensure_dir(final_root)
    plots_dir = final_root / "plots"
    ensure_dir(plots_dir)

    words_df, timing_df, real_agg, tts_agg = aggregate(dtw_root, praat_root)

    # ---- Normalization (BD only) ----
    if not words_df.empty:
        for col in ["bd_full","bd_per_phone"]:
            if col in words_df.columns:
                words_df = add_minmax_and_z_from_cost(words_df, col, f"{col}")

    if not timing_df.empty and "bd_mean" in timing_df.columns:
        timing_df = add_minmax_and_z_from_cost(timing_df, "bd_mean", "bd")

    # ---- Formant comparison table (merge Real vs TTS, add deltas) ----
    formants_merged = pd.merge(real_agg, tts_agg, on="phone", how="outer", suffixes=("_Real","_TTS"))
    for c in ["F1_mean","F2_mean","F3_mean"]:
        if f"{c}_Real" in formants_merged.columns and f"{c}_TTS" in formants_merged.columns:
            formants_merged[f"d{c.split('_')[0]}_Hz"] = formants_merged[f"{c}_TTS"] - formants_merged[f"{c}_Real"]
    if {"F1_mean_Real","F2_mean_Real","F1_mean_TTS","F2_mean_TTS"}.issubset(formants_merged.columns):
        dF1 = formants_merged["F1_mean_TTS"] - formants_merged["F1_mean_Real"]
        dF2 = formants_merged["F2_mean_TTS"] - formants_merged["F2_mean_Real"]
        formants_merged["euclid_F1F2_Hz"] = np.sqrt(dF1**2 + dF2**2)
    if "N_Real" not in formants_merged.columns:
        formants_merged = formants_merged.merge(real_agg[["phone","N"]].rename(columns={"N":"N_Real"}), on="phone", how="left")
    if "N_TTS" not in formants_merged.columns:
        formants_merged = formants_merged.merge(tts_agg[["phone","N"]].rename(columns={"N":"N_TTS"}), on="phone", how="left")

    # ---- Write CSVs ----
    words_csv   = final_root / "All Words.csv"
    timing_csv  = final_root / "All Phoneme Timing.csv"
    formant_csv = final_root / "All Phoneme Formants.csv"

    words_df.to_csv(words_csv, index=False)
    timing_df.to_csv(timing_csv, index=False)
    formants_merged.to_csv(formant_csv, index=False)

    # ---- Combined vowel space plot (grand means) ----
    plt.figure(figsize=(9, 5.5))
    if not real_agg.empty:
        for _, row in real_agg.iterrows():
            plt.scatter(row["F2_mean"], row["F1_mean"], s=80, marker="o", label=None)
            plt.text(row["F2_mean"], row["F1_mean"] - LABEL_Y_OFFSET, str(row["phone"]), fontsize=12, fontweight="bold", ha="center", va="top")
    if not tts_agg.empty:
        for _, row in tts_agg.iterrows():
            plt.scatter(row["F2_mean"], row["F1_mean"], s=80, marker="s", label=None)
            plt.text(row["F2_mean"], row["F1_mean"] - LABEL_Y_OFFSET, str(row["phone"]), fontsize=12, fontweight="bold", ha="center", va="top")
    plt.xlabel("F2 (Hz)"); plt.ylabel("F1 (Hz)")
    plt.title("Promedios de F1 y F2 de fonemas vocales — Real ○, TTS □")
    plt.xlim(F2_MIN, F2_MAX); plt.ylim(F1_MIN, F1_MAX)
    plt.legend(["Promedio Real (○)", "Promedio TTS (□)"], loc="best")
    plot_path = plots_dir / "VowelSpace_Combined.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Wrote:")
    for p in [words_csv, formant_csv, timing_csv, plot_path]:
        print("  -", p)


def main():
    base = Path(__file__).resolve().parent
    dtw_root   = base / "DTW Salida"
    praat_root = base / "Praat Salida"
    final_root = base / "Final Analysis"

    if not dtw_root.exists():
        raise SystemExit(f"DTW output not found: {dtw_root}")
    if not praat_root.exists():
        raise SystemExit(f"Praat output not found: {praat_root}")

    save_outputs(dtw_root, praat_root, final_root)


if __name__ == "__main__":
    main()
