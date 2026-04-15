"""
Worldribbon v2 — Within-Subject PCI Re-Analysis
PhysioNet EEGBCI: Eyes-Closed Rest (R02) vs Motor Execution (R03+R07)

Pipeline matches Ursachi 2026 (DOI: 10.3389/fnhum.2026.1781338) with one
amendment: artifact threshold raised to ±500 µV (raw unprocessed EEGBCI
signals peak at ~300 µV post-filter; 100 µV rejects all epochs).

Secondary analyses:
  - Motor imagery (R04+R08) vs execution (R03+R07)
  - Eyes-open (R01) vs eyes-closed (R02) rest
  - Frontal channel replication
"""

import mne
import numpy as np
from scipy.signal import welch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
np.random.seed(42)

PHI = (1 + np.sqrt(5)) / 2   # 1.618034…
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
EPOCH_LENGTH = 4.0            # seconds
ARTIFACT_THRESHOLD = 500e-6   # ±500 µV  (raw EEGBCI amplitude; see docstring)
MNE_BASE = Path('/home/runner/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0')

OUTPUT_DIR = Path('outputs/worldribbon_v2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)
CACHE_FILE = OUTPUT_DIR / 'cache.json'


# ── helpers ──────────────────────────────────────────────────────────────

def spectral_centroid(psd, freqs, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    p = psd[mask]
    f = freqs[mask]
    return float(np.sum(f * p) / np.sum(p)) if p.sum() > 0 else float('nan')


def pci(ratio):
    dp = abs(ratio - PHI)
    dh = abs(ratio - 2.0)
    if dp == 0: return 10.0
    if dh == 0: return -10.0
    return float(np.log(dh / dp))


def posterior_idx(raw):
    upper = [ch.upper().rstrip('.').replace('.', '') for ch in raw.ch_names]
    targets = ('O1', 'O2', 'OZ', 'POZ', 'P3', 'P4', 'PZ')
    return sorted({i for t in targets for i, c in enumerate(upper) if c == t})


def frontal_idx(raw):
    upper = [ch.upper().rstrip('.').replace('.', '') for ch in raw.ch_names]
    targets = ('F3', 'F4', 'FZ', 'FP1', 'FP2', 'F7', 'F8')
    return sorted({i for t in targets for i, c in enumerate(upper) if c == t})


def compute_metrics(raw, ch_idx, events=None, task_ids=None):
    """Return dict of metrics, or None if insufficient clean data."""
    if not ch_idx:
        return None
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ep = int(EPOCH_LENGTH * sfreq)

    if events is not None and task_ids:
        segs = [data[:, ev[0]:ev[0]+ep]
                for ev in events if ev[2] in task_ids
                and ev[0]+ep <= data.shape[1]]
    else:
        n = data.shape[1] // ep
        segs = [data[:, i*ep:(i+1)*ep] for i in range(n)]

    clean = [s for s in segs if np.max(np.abs(s)) < ARTIFACT_THRESHOLD]
    if len(clean) < 2:
        return None

    nperseg = ep
    psds = []
    for s in clean:
        freqs, p = welch(s[ch_idx], fs=sfreq,
                         nperseg=nperseg, noverlap=nperseg//2, window='hann')
        psds.append(p.mean(axis=0))
    mean_psd = np.mean(psds, axis=0)

    theta = spectral_centroid(mean_psd, freqs, THETA_BAND)
    alpha = spectral_centroid(mean_psd, freqs, ALPHA_BAND)
    if np.isnan(theta) or np.isnan(alpha) or theta == 0:
        return None

    r = alpha / theta
    return {
        'theta': theta, 'alpha': alpha, 'ratio': r,
        'pci': pci(r), 'convergence': abs(theta-8)+abs(alpha-8),
        'n_epochs': len(clean),
    }


def load_raw(sid, run):
    """Load (downloading if needed) a single run. Returns raw or None."""
    fpath = MNE_BASE / f'S{sid:03d}' / f'S{sid:03d}R{run:02d}.edf'
    if not fpath.exists():
        try:
            mne.datasets.eegbci.load_data(sid, [run], update_path=True, verbose=False)
        except Exception as e:
            return None
    if not fpath.exists():
        return None
    try:
        r = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)
        r.filter(1, 45, fir_design='firwin', verbose=False)
        return r
    except Exception:
        return None


def process_task_runs(sid, runs, ch_idx):
    """Pool task-epoch metrics across multiple runs."""
    results = []
    for run in runs:
        raw = load_raw(sid, run)
        if raw is None:
            continue
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            tids = [v for k, v in event_id.items() if k in ('T1', 'T2')]
        except Exception:
            continue
        m = compute_metrics(raw, ch_idx, events=events, task_ids=tids)
        if m:
            results.append(m)
    if not results:
        return None
    keys = ['theta', 'alpha', 'ratio', 'pci', 'convergence']
    return {k: float(np.mean([r[k] for r in results])) for k in keys} | \
           {'n_epochs': sum(r['n_epochs'] for r in results)}


# ── load cache ────────────────────────────────────────────────────────────

cache = {}
if CACHE_FILE.exists():
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    print(f"Loaded cache: {len(cache)} subjects already processed.")

print("=" * 70)
print("WORLDRIBBON v2: Within-Subject PCI Re-Analysis")
print(f"PhysioNet EEGBCI  |  PHI = {PHI:.6f}  |  threshold = ±500 µV")
print("=" * 70)

# ── per-subject loop ──────────────────────────────────────────────────────

for sid in range(1, 110):
    key = str(sid)
    if key in cache:
        print(f"S{sid:03d} [cached]")
        continue

    print(f"S{sid:03d} ", end='', flush=True)

    try:
        raw_ec = load_raw(sid, 2)   # eyes-closed rest
        if raw_ec is None:
            print("SKIP(no R02)")
            cache[key] = None
            continue

        post = posterior_idx(raw_ec)
        front = frontal_idx(raw_ec)

        if len(post) < 3:
            print(f"SKIP(post ch={len(post)})")
            cache[key] = None
            continue

        ec_post = compute_metrics(raw_ec, post)
        if ec_post is None:
            print("SKIP(rest clean<2)")
            cache[key] = None
            continue

        exec_post  = process_task_runs(sid, [3, 7], post)
        imag_post  = process_task_runs(sid, [4, 8], post)
        ec_front   = compute_metrics(raw_ec, front) if len(front) >= 3 else None

        exec_front = None
        if len(front) >= 3:
            exec_front = process_task_runs(sid, [3, 7], front)

        raw_eo = load_raw(sid, 1)
        eo_post = compute_metrics(raw_eo, post) if raw_eo is not None else None

        if exec_post is None:
            print("SKIP(no task)")
            cache[key] = None
            continue

        row = {
            'ec_post':    ec_post,
            'exec_post':  exec_post,
            'imag_post':  imag_post,
            'eo_post':    eo_post,
            'ec_front':   ec_front,
            'exec_front': exec_front,
        }
        cache[key] = row

        r0, r1 = ec_post['ratio'], exec_post['ratio']
        p0, p1 = ec_post['pci'], exec_post['pci']
        arrow = 'v' if p0 > p1 else '^'
        print(f"R={r0:.3f}->{r1:.3f}  PCI={p0:+.3f}->{p1:+.3f} {arrow}")

    except Exception as e:
        print(f"ERROR: {e}")
        cache[key] = None

    # save cache after every subject
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# ── assemble dataframe ────────────────────────────────────────────────────

rows = []
for sid in range(1, 110):
    key = str(sid)
    entry = cache.get(key)
    if not entry:
        continue
    ec  = entry['ec_post']
    ex  = entry['exec_post']
    if not ec or not ex:
        continue
    row = {'subject': sid,
           'theta_rest': ec['theta'],  'alpha_rest': ec['alpha'],
           'ratio_rest': ec['ratio'],  'pci_rest':   ec['pci'],
           'conv_rest':  ec['convergence'], 'n_epochs_rest': ec['n_epochs'],
           'theta_task': ex['theta'],  'alpha_task': ex['alpha'],
           'ratio_task': ex['ratio'],  'pci_task':   ex['pci'],
           'conv_task':  ex['convergence'], 'n_epochs_task': ex['n_epochs']}
    if entry.get('imag_post'):
        im = entry['imag_post']
        row.update({'pci_imagery': im['pci'], 'ratio_imagery': im['ratio']})
    if entry.get('eo_post'):
        eo = entry['eo_post']
        row.update({'pci_rest_eo': eo['pci'], 'ratio_rest_eo': eo['ratio']})
    if entry.get('ec_front') and entry.get('exec_front'):
        row.update({'pci_rest_frontal':  entry['ec_front']['pci'],
                    'ratio_rest_frontal': entry['ec_front']['ratio'],
                    'pci_task_frontal':  entry['exec_front']['pci'],
                    'ratio_task_frontal': entry['exec_front']['ratio']})
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / 'worldribbon_v2_data.csv', index=False)
print(f"\n{'='*70}")
print(f"VALID DATA: N={len(df)} subjects")
print(f"{'='*70}")

if len(df) < 5:
    print("Too few subjects — run the script again to continue downloading.")
    sys.exit(0)

# ── PRIMARY: EC rest vs motor execution ──────────────────────────────────

t_stat, p_value = stats.ttest_rel(df['pci_rest'], df['pci_task'])
wilcox_stat, wilcox_p = stats.wilcoxon(df['pci_rest'], df['pci_task'])
diff = df['pci_rest'] - df['pci_task']
d_paired = diff.mean() / (diff.std(ddof=1) + 1e-20)
r_corr, p_corr = stats.pearsonr(df['pci_rest'], diff)

print(f"\n── PRIMARY: Eyes-Closed Rest vs Motor Execution (posterior) ──")
print(f"  PCI_rest:  {df['pci_rest'].mean():.4f} ± {df['pci_rest'].std():.4f}")
print(f"  PCI_task:  {df['pci_task'].mean():.4f} ± {df['pci_task'].std():.4f}")
print(f"  ΔPCI:      {diff.mean():+.4f} ± {diff.std():.4f}")
print(f"  Paired t({len(df)-1}): t={t_stat:.4f}, p={p_value:.2e}")
print(f"  Wilcoxon: W={wilcox_stat:.1f}, p={wilcox_p:.2e}")
print(f"  Cohen's d (paired): {d_paired:.4f}")
print(f"  PCI_rest > PCI_task: {(diff>0).sum()}/{len(df)} ({(diff>0).mean()*100:.1f}%)")
print(f"  Phi-organized rest (PCI>0): {(df['pci_rest']>0).sum()}/{len(df)} ({(df['pci_rest']>0).mean()*100:.1f}%)")
print(f"  Phi-organized task (PCI>0): {(df['pci_task']>0).sum()}/{len(df)} ({(df['pci_task']>0).mean()*100:.1f}%)")
print(f"  R_rest: {df['ratio_rest'].mean():.4f} ± {df['ratio_rest'].std():.4f}")
print(f"  R_task: {df['ratio_task'].mean():.4f} ± {df['ratio_task'].std():.4f}")
print(f"  Baseline PCI vs ΔPCI: r={r_corr:.4f}, p={p_corr:.2e}")
prediction = p_value < 0.05 and diff.mean() > 0
print(f"  PREDICTION (PCI_rest > PCI_task): {'SUPPORTED' if prediction else 'NOT SUPPORTED'}")

# ── SECONDARY 1: execution vs imagery ────────────────────────────────────
if 'pci_imagery' in df.columns:
    dfm = df.dropna(subset=['pci_imagery'])
    if len(dfm) >= 10:
        t2, p2 = stats.ttest_rel(dfm['pci_task'], dfm['pci_imagery'])
        d2 = (dfm['pci_task'] - dfm['pci_imagery']).mean() / \
             (dfm['pci_task'] - dfm['pci_imagery']).std(ddof=1)
        print(f"\n── SECONDARY 1: Execution vs Imagery (N={len(dfm)}) ──")
        print(f"  PCI_exec:    {dfm['pci_task'].mean():.4f}")
        print(f"  PCI_imagery: {dfm['pci_imagery'].mean():.4f}")
        print(f"  t={t2:.4f}, p={p2:.2e}, d={d2:.4f}")

# ── SECONDARY 2: EO vs EC rest ────────────────────────────────────────────
if 'pci_rest_eo' in df.columns:
    dfeo = df.dropna(subset=['pci_rest_eo'])
    if len(dfeo) >= 10:
        t3, p3 = stats.ttest_rel(dfeo['pci_rest_eo'], dfeo['pci_rest'])
        d3 = (dfeo['pci_rest_eo'] - dfeo['pci_rest']).mean() / \
             (dfeo['pci_rest_eo'] - dfeo['pci_rest']).std(ddof=1)
        print(f"\n── SECONDARY 2: Eyes-Open vs Eyes-Closed Rest (N={len(dfeo)}) ──")
        print(f"  PCI_EO: {dfeo['pci_rest_eo'].mean():.4f}")
        print(f"  PCI_EC: {dfeo['pci_rest'].mean():.4f}")
        print(f"  t={t3:.4f}, p={p3:.2e}, d={d3:.4f}")

# ── SECONDARY 3: frontal replication ─────────────────────────────────────
if 'pci_rest_frontal' in df.columns:
    dff = df.dropna(subset=['pci_rest_frontal','pci_task_frontal'])
    if len(dff) >= 10:
        t4, p4 = stats.ttest_rel(dff['pci_rest_frontal'], dff['pci_task_frontal'])
        d4 = (dff['pci_rest_frontal'] - dff['pci_task_frontal']).mean() / \
             (dff['pci_rest_frontal'] - dff['pci_task_frontal']).std(ddof=1)
        print(f"\n── SECONDARY 3: Frontal replication (N={len(dff)}) ──")
        print(f"  PCI_rest_frontal: {dff['pci_rest_frontal'].mean():.4f}")
        print(f"  PCI_task_frontal: {dff['pci_task_frontal'].mean():.4f}")
        print(f"  t={t4:.4f}, p={p4:.2e}, d={d4:.4f}")

# ── dip test ──────────────────────────────────────────────────────────────
try:
    from diptest import diptest
    dip_r, dip_rp = diptest(df['ratio_rest'].values)
    dip_t, dip_tp = diptest(df['ratio_task'].values)
    print(f"\n── Hartigan Dip Test (bimodality) ──")
    print(f"  Rest: dip={dip_r:.4f}, p={dip_rp:.4f}")
    print(f"  Task: dip={dip_t:.4f}, p={dip_tp:.4f}")
except ImportError:
    pass

# ── figures ───────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'Worldribbon v2  |  PCI: Eyes-Closed Rest vs Motor Execution\n'
             f'PhysioNet EEGBCI, N={len(df)}, threshold=±500µV',
             fontsize=13, fontweight='bold')

# F1: paired PCI
ax = axes[0, 0]
for _, row in df.iterrows():
    c = 'steelblue' if row['pci_rest'] > row['pci_task'] else 'coral'
    ax.plot([0, 1], [row['pci_rest'], row['pci_task']],
            color=c, alpha=0.2, lw=0.7)
ax.scatter(np.zeros(len(df)), df['pci_rest'], color='steelblue', s=20, zorder=5, label='Rest (EC)')
ax.scatter(np.ones(len(df)),  df['pci_task'],  color='coral',     s=20, zorder=5, label='Exec')
ax.plot([0, 1], [df['pci_rest'].mean(), df['pci_task'].mean()],
        'k-', lw=3, label='Group mean')
ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6, label='PCI=0')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Rest (EC)', 'Motor Exec'])
ax.set_ylabel('Phi Coupling Index (PCI)')
verdict = 'SUPPORTED' if prediction else 'NOT SUPPORTED'
ax.set_title(f'Paired PCI\nt={t_stat:.3f}, p={p_value:.2e}, d={d_paired:.3f}  [{verdict}]')
ax.legend(fontsize=8)

# F2: ratio distributions
ax = axes[0, 1]
ax.hist(df['ratio_rest'], bins=25, alpha=0.5, color='steelblue',
        label='Rest (EC)', density=True, edgecolor='white')
ax.hist(df['ratio_task'], bins=25, alpha=0.5, color='coral',
        label='Motor Exec', density=True, edgecolor='white')
ax.axvline(PHI, color='gold',    lw=2.5, ls='-',  label=f'phi={PHI:.3f}')
ax.axvline(2.0, color='crimson', lw=2.5, ls='--', label='Harm=2.0')
ax.axvline(df['ratio_rest'].mean(), color='steelblue', lw=1.5, ls=':', alpha=0.8)
ax.axvline(df['ratio_task'].mean(), color='coral',     lw=1.5, ls=':', alpha=0.8)
ax.set_xlabel('Alpha/Theta Ratio'); ax.set_ylabel('Density')
ax.set_title(f'Ratio Distributions\nRest={df["ratio_rest"].mean():.3f}  Task={df["ratio_task"].mean():.3f}')
ax.legend(fontsize=8)

# F3: scatter
ax = axes[1, 0]
sc = ax.scatter(df['pci_rest'], df['pci_task'], alpha=0.6, s=35,
                c=df['ratio_rest'], cmap='RdYlGn_r', edgecolors='k', lw=0.3)
plt.colorbar(sc, ax=ax, label='Rest ratio')
lo = min(df['pci_rest'].min(), df['pci_task'].min()) - 0.3
hi = max(df['pci_rest'].max(), df['pci_task'].max()) + 0.3
ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.5, label='Identity')
ax.axhline(0, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('PCI (Rest EC)'); ax.set_ylabel('PCI (Motor Exec)')
ax.set_title('Rest vs Task PCI\n(color = rest ratio)')
ax.legend(fontsize=8); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

# F4: ΔPCI histogram
ax = axes[1, 1]
ax.hist(diff, bins=25, color='mediumpurple', edgecolor='black', alpha=0.75, lw=0.5)
ax.axvline(0, color='crimson', lw=2, ls='--', label='No change')
ax.axvline(diff.mean(), color='gold', lw=2, label=f'Mean={diff.mean():+.3f}')
ax.set_xlabel('ΔPCI (Rest − Task)'); ax.set_ylabel('Count')
ax.set_title(f'PCI Change (Rest − Task)\n{(diff>0).sum()}/{len(df)} subjects: rest > task')
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = OUTPUT_DIR / 'figures' / 'worldribbon_v2_results.png'
plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

# ── manuscript summary ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY FOR MANUSCRIPT")
print(f"{'='*70}")
print(f"""
Within-subject PCI comparison: eyes-closed rest vs motor execution
N = {len(df)} subjects, PhysioNet EEGBCI, posterior channels (O1/O2/Oz/P3/P4/Pz)
Artifact threshold: ±500 µV (post 1–45 Hz FIR)

Mean PCI (rest):     {df['pci_rest'].mean():.3f} ± {df['pci_rest'].std():.3f}
Mean PCI (task):     {df['pci_task'].mean():.3f} ± {df['pci_task'].std():.3f}
ΔPCI (rest − task):  {diff.mean():+.3f} ± {diff.std():.3f}

Paired t-test:  t({len(df)-1}) = {t_stat:.3f}, p = {p_value:.2e}
Wilcoxon:       W = {wilcox_stat:.1f},  p = {wilcox_p:.2e}
Cohen's d:      {d_paired:.3f} (paired)

{(diff>0).sum()}/{len(df)} subjects ({(diff>0).mean()*100:.1f}%) showed PCI_rest > PCI_task

Prediction (PCI_rest > PCI_task): {verdict}

Alpha/theta ratio:
  Rest:  {df['ratio_rest'].mean():.4f} ± {df['ratio_rest'].std():.4f}
  Task:  {df['ratio_task'].mean():.4f} ± {df['ratio_task'].std():.4f}
  phi={PHI:.4f},  harmonic=2.0000

Phi-organized (PCI > 0):
  Rest: {(df['pci_rest']>0).sum()}/{len(df)} ({(df['pci_rest']>0).mean()*100:.1f}%)
  Task: {(df['pci_task']>0).sum()}/{len(df)} ({(df['pci_task']>0).mean()*100:.1f}%)

Baseline PCI vs ΔPCI: r = {r_corr:.3f}, p = {p_corr:.2e}
""")
