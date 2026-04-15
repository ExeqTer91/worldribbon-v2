# Worldribbon v2 — Within-Subject PCI Re-Analysis

**PhysioNet EEGBCI · Eyes-Closed Rest vs Motor Execution · N = 98**

Replication and extension of the within-subject Phi Coupling Index (PCI)
analysis from Ursachi (2026), testing the prediction that phi-organization
decreases during motor engagement.

---

## Prediction

> **PCI_rest > PCI_task** — phi-organization, quantified as proximity of the
> alpha/theta frequency ratio to φ = 1.618…, should be higher at rest than
> during motor execution.

---

## Result: SUPPORTED

| | Eyes-Closed Rest | Motor Execution |
|---|---|---|
| Mean PCI | **1.082 ± 1.255** | **0.640 ± 1.254** |
| ΔPCI | +0.442 ± 1.347 | |
| Mean α/θ ratio | 1.686 ± 0.134 | 1.740 ± 0.120 |
| Phi-organized (PCI > 0) | 79.6% (78/98) | 71.4% (70/98) |

**Paired t(97) = 3.248, p = 0.0016**  
**Wilcoxon W = 1574, p = 0.0026**  
**Cohen's d = 0.328 (paired)**  
60/98 subjects (61.2%) showed PCI_rest > PCI_task.

---

## Secondary Analyses

| Comparison | N | t | p | d |
|---|---|---|---|---|
| Motor execution vs imagery | 97 | -1.64 | 0.104 | -0.167 |
| Eyes-open vs eyes-closed rest | 96 | -2.75 | 0.007 | -0.280 |
| Frontal replication (rest vs exec) | 98 | 7.24 | 1.07e-10 | 0.731 |

The frontal channel replication (F3/F4/Fz/Fp1/Fp2/F7/F8) shows a markedly
stronger effect (d = 0.731) than posterior channels, consistent with frontal
theta's established role in phi-organization (Ursachi, 2026).

The eyes-open vs eyes-closed rest comparison confirms that EC rest yields
higher PCI than EO rest (0.692 vs 1.084, p = 0.007), consistent with the
alpha-dependent nature of phi-organization.

**Note on regression-to-mean:** Baseline PCI vs ΔPCI: r = 0.538,
p = 1.15e-08. Subjects with higher rest PCI show larger task-induced
decreases, indicating regression-to-mean partially contributes to the
observed effect.

---

## Metric Definitions

```
R       = f_alpha / f_theta      (alpha/theta frequency ratio)
PCI     = log(|R − 2.0| / |R − 1.618...|)
          PCI > 0: closer to phi (phi-organized)
          PCI < 0: closer to harmonic 2:1
C       = |f_theta − 8| + |f_alpha − 8|   (convergence to 8 Hz)
```

Spectral centroids computed via Welch PSD (4-second Hann windows, 50%
overlap) on posterior channels (O1, O2, Oz, P3, P4, Pz).

---

## Dataset

**PhysioNet EEGBCI Motor/Movement Imagery**
- URL: https://physionet.org/content/eegmmidb/1.0.0/
- 109 subjects, 64 channels, 160 Hz
- Rest: **Run 2 (R02)** = eyes-closed baseline
- Task: **Runs 3 + 7 (R03, R07)** = motor execution (open/close fists)
- Secondary: Runs 4, 8 (imagery); Run 1 (eyes-open rest)

**Subjects excluded (task data unavailable):** S003, S009, S022, S054,
S061, S077, S096 and 4 others — 98/109 subjects with valid paired data.

---

## Pipeline

1. Bandpass filter 1–45 Hz (FIR, `firwin`)
2. Posterior channel selection (O1/O2/Oz/P3/P4/Pz)
3. Segment into 4-second epochs; artifact rejection threshold ±500 µV
   *(raw EEGBCI signals peak at ~300 µV post-filter; the 100 µV threshold
   in the original pipeline spec rejects all epochs and was amended here)*
4. PSD via Welch (4 s Hann window, 50% overlap)
5. Average across clean epochs and posterior channels
6. Compute spectral centroids → R → PCI

---

## Reproducing

```bash
pip install mne numpy scipy matplotlib pandas diptest
python worldribbon_v2.py
```

MNE will auto-download PhysioNet EEGBCI on first run (~650 MB for all 6
runs across 109 subjects). Results are cached per-subject in
`outputs/worldribbon_v2/cache.json` so the script is safely resumable.

**Outputs:**
- `outputs/worldribbon_v2_data.csv` — per-subject metrics
- `outputs/figures/worldribbon_v2_results.png` — 4-panel figure (300 DPI)

---

## Reference

Ursachi, A.-S. (2026). *Phi-organized neural oscillations: theta–alpha
frequency ratios as fixed golden-ratio architecture in human EEG.*
Frontiers in Human Neuroscience. DOI: 10.3389/fnhum.2026.1781338

---

## Context

This analysis is a within-subject replication satellite to the Schumann–EEG
bridge manuscript (Ursachi, in preparation), which demonstrates convergent
phi-organization in Schumann resonance frequency ratios (r31 ≈ φ²) and EEG
alpha/theta ratios. The current result — that PCI is higher at rest than
during motor engagement — is cited in Section 4.3 (Cross-Domain Implications)
as evidence that phi-organization is a dynamic state property, not a
fixed architectural constant.

---

*Author: Andrei-Sebastian Ursachi, Independent Researcher, Diepholz, Germany*  
*contact@andreiursachi.eu | ORCID: 0009-0002-6114-5011*
