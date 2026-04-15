# Worldribbon: Within-Subject PCI Analysis (PhysioNet EEGBCI)

Re-analysis of PhysioNet EEG Motor Imagery dataset (N=109) using a within-subject 
Phi-Convergence Index (PCI) design. Theta–alpha spectral centroid ratios are compared 
between **eyes-closed rest** and **motor execution** to test whether EEG resting states 
show greater phi-organisation (α/θ centroid ratio closer to φ=1.618).

---

## Version History

### v3 (Current) — Full-Spectrum Pipeline with ANCOVA
**N = 41 valid subjects** (±150 µV threshold after average re-reference, ≥5 clean epochs)

| Metric | Value |
|--------|-------|
| t(40) | **6.862** |
| p-value | **2.92 × 10⁻⁸** |
| Cohen's d | **1.072** |
| Direction | **85.4%** PCI_rest > PCI_task |
| Ratio rest | 1.696 |
| Ratio task | 1.821 |
| VERDICT | **SUPPORTED** (raw) |

**Critical methodological controls:**

| Control | Result |
|---------|--------|
| ANCOVA (baseline-controlled) | t≈0, p≈1.0 — **NOT significant** |
| FOOOF-corrected (N=39) | t=1.20, p=0.236, d=0.193 — not significant |
| Flat-spectrum null model | 97.6% outside 1/f CI (rest ratio ≠ pure aperiodic) |
| Frontal replication | d=0.753, p=2.1×10⁻⁵ — significant |
| Winsorized robustness | t=7.31, p=7.0×10⁻⁹ — robust |
| Imagery vs Execution | d=0.496, p=0.004 — condition-specific |
| Eyes-open vs EC rest | d=0.629, p=0.004 |

**Key finding:** The raw PCI effect (d=1.072) is entirely accounted for by 
regression-to-mean (baseline PCI → ΔPCI correlation: r=0.563). After ANCOVA, 
the effect vanishes (p=1.0). FOOOF-corrected analysis (removing aperiodic 1/f 
component) also shows no significant effect (p=0.236). However, 97.6% of subjects 
show rest spectral ratios outside the 1/f null model CI, suggesting genuine spectral 
organisation that differs from purely aperiodic noise.

**Pipeline improvements over v2:**
1. Average reference before artifact rejection
2. ±150 µV threshold on re-referenced data (v2: ±500 µV raw)
3. MIN_EPOCHS = 5 per condition
4. FOOOF aperiodic correction (fooof 1.1 / specparam)
5. ε=0.01 regularized PCI
6. Flat-spectrum null model (200 surrogates)
7. ANCOVA regression-to-mean control

**Figures:**
- `fig1_primary`: Primary PCI comparison (rest vs task) with violin/raincloud plots
- `fig2_fooof`: FOOOF-corrected PCI comparison
- `fig3_null`: Flat-spectrum null model with individual data points
- `fig4_frontal`: Frontal electrode replication

---

### v2 — Initial Replication
**N = 98 valid subjects** (±500 µV threshold, no re-referencing)

| Metric | Value |
|--------|-------|
| t(97) | **3.248** |
| p-value | **0.0016** |
| Cohen's d | **0.328** |
| Direction | **61.2%** PCI_rest > PCI_task |
| Frontal replication | d=0.731, p=1.07×10⁻¹⁰ |
| Baseline↔ΔPCI | r=0.538 |
| VERDICT | **SUPPORTED** |

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)**
- 109 subjects, 64-channel EEG, 160 Hz
- Runs: R02 (eyes-closed rest), R03+R07 (motor execution), R04+R08 (motor imagery), R01 (eyes-open)
- Reference: Goldberger et al. (2000). PhysioBank/PhysioToolkit/PhysioNet.
- DOI: https://doi.org/10.1161/01.CIR.101.23.e215

## Method

1. Load EDF files via MNE Python (`mne.datasets.eegbci`)
2. Band-pass filter 1–45 Hz (FIR, firwin)
3. Average re-reference (subtract channel mean per time point)
4. Epoch into 4-second non-overlapping windows
5. Artifact rejection: ±150 µV (v3) / ±500 µV (v2)
6. Welch PSD (nperseg = epoch length, 50% overlap)
7. Spectral centroids: θ=[4–8 Hz], α=[8–13 Hz]
8. FOOOF aperiodic correction (R²≥0.85 quality gate)
9. PCI = (α_centroid/θ_centroid − φ) / φ × 100
10. Paired t-test + Wilcoxon (rest vs. task, within-subject)
11. ANCOVA controlling for baseline PCI (regression-to-mean)
12. Flat-spectrum null: 200 surrogates from FOOOF aperiodic parameters

## Files

```
worldribbon_v2.py                        # v2 script
worldribbon_v3.py                        # v3 script (current)
outputs/worldribbon_v2/
  worldribbon_v2_data.csv               # v2 per-subject data
  figures/                              # v2 figures
outputs/worldribbon_v3/
  worldribbon_v3_data.csv               # v3 per-subject data (N=41)
  worldribbon_v3_summary.json           # v3 statistical summary
  figures/
    fig1_primary.png                    # PCI rest vs task
    fig2_fooof.png                      # FOOOF-corrected comparison
    fig3_null.png                       # Flat-spectrum null model
    fig4_frontal.png                    # Frontal electrode replication
```

## Interpretation

The Worldribbon project tests whether resting EEG shows greater phi-organisation 
(α/θ ratio closer to φ=1.618) compared to motor task EEG.

**v3 conclusion:** The large raw effect (d=1.072) is a regression-to-mean artifact: 
subjects with high rest PCI naturally have lower task PCI, and ANCOVA completely 
removes the effect (p=1.0). The FOOOF-corrected effect is small and non-significant 
(d=0.193). The 1/f null model shows that rest spectral ratios do cluster differently 
from pure aperiodic noise (97.6% outside CI), but this likely reflects the well-known 
theta–alpha peak structure in resting EEG rather than phi-specific organisation.

## License

MIT License. Data: PhysioNet Open Access (CC-BY).
Author: ExeqTer91
