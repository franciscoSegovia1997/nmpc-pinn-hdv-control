<div align="center">

# Lateral Control of an Autonomous Vehicle During Obstacle-Evasion Maneuvers
### A comparison of Stanley, NMPC-MD, NMPC-DD, and NMPC-PINN in CARLA 0.9.16

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![CARLA](https://img.shields.io/badge/CARLA-0.9.16-orange.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Status](https://img.shields.io/badge/status-validated-brightgreen.svg)

</div>

This repository presents the validation code and results from my master's thesis, **“Design of a Control System for a Mobile Vehicle in Emergency Obstacle-Evasion Scenarios”**, completed in the Master's Program in Control and Automation at the Pontifical Catholic University of Peru (PUCP). The work was validated in [CARLA](https://carla.org/) 0.9.16 using an identified model of the Fusorosa bus (m = 4800 kg, l_f = 3.018 m, l_r = 2.612 m, I_z = 21000 kg·m²).

The repository contains **the final validation results (Chapter 4 only)**: four lateral controllers compared under identical conditions, together with a sensitivity analysis (mass / friction / speed) evaluating each controller's generalization capability. It does not include the modeling development (Chapter 2) or controller design and tuning (Chapter 3); that research process remains in the private working repository.

## Controllers

| # | Controller | Description |
|---|---|---|
| 1 | **Stanley** | Geometric control law with curvature feedforward |
| 2 | **NMPC-MD** | NMPC with a dynamic model (5-DoF bicycle model, parameters identified for the Fusorosa) |
| 3 | **NMPC-DD** | NMPC with a data-driven model (MLP), numerically linearized outside CasADi |
| 4 | **NMPC-PINN** | NMPC with a physics-informed model (MLP + physical constraints) |

## Main result — Town07 (route unseen during training)

<div align="center">
<img src="assets/fig_metricas_barras.png" alt="Comparative metrics for the four controllers in Town07" width="85%">
</div>

| Controller | RMSE_cte [m] | MAE_cte [m] | E_max [m] | TV_δ [rad] |
|---|---|---|---|---|
| Stanley | 1.128 | 0.589 | 6.178 | 69.77 |
| NMPC-MD | 0.580 | 0.300 | 2.803 | 48.34 |
| NMPC-DD | **0.416** | 0.244 | 2.192 | 58.74 |
| NMPC-PINN | 0.449 | 0.276 | 1.871 | 47.32 |

<div align="center">
<img src="assets/fig_cte_comparacion.png" alt="Cross-track error (CTE) over the Town07 route" width="85%">
</div>

See [`resultados/06_comparacion_final_town07`](resultados/06_comparacion_final_town07) for the analysis notebook and complete figures.

## Key finding — generalization to unseen conditions

On the nominal Town07 route, NMPC-DD and NMPC-PINN perform very similarly. Their difference emerges in the **sensitivity analysis** (mass, friction coefficient, and cruise speed; see [`resultados/07_escenarios_sensibilidad`](resultados/07_escenarios_sensibilidad)):

<div align="center">
<img src="assets/fig_comparativa_cruzada.png" alt="Cross-comparison of sensitivity across scenarios" width="85%">
</div>

Because the DD model is purely statistical, it **diverges in combinations outside its training distribution** (see the `*_fail.npy` cases in `07_escenarios_sensibilidad/`). In contrast, **NMPC-PINN remains stable across all evaluated scenarios** due to the physical constraints incorporated during training. This is the quantitative generalization evidence supporting the thesis's main conclusion.

## Simulation videos

Recordings of the CARLA runs are available for the four controllers in Town07, the scenario variants (Scenarios 1/2), and the state-estimation variant (PINN-EKF):

📁 **[Video folder (Google Drive)](https://drive.google.com/drive/folders/1M352pA8DdquowHwVZXGbhakHKtNLHEvh?usp=drive_link)**

| Video | Controller / experiment |
|---|---|
| `controladorStanley_Tesis_V1.mp4` | Stanley — Town07 |
| `controladorNMPC-MD_Tesis_V2.mp4` | NMPC-MD — Town07 |
| `controladorNMPC-DD_Tesis_V3.mp4` | NMPC-DD — Town07 |
| `controladorNMPC-PINN_Tesis_V4.mp4` | NMPC-PINN — Town07 |
| `NMPC-*_Escenario1_V5-V8.mp4` | DD/PINN — sensitivity sweep, Scenario 1 |
| `NMPC-*-Escenario2_V9-V12.mp4` | DD/PINN — sensitivity sweep, Scenario 2 |
| `NMPC-PINN_V13.mp4` / `NMPC-PINN-EKF_V14.mp4` | PINN without/with state estimation (EKF) |

> The videos are not included in this repository: they total approximately 475 MB, and two files exceed GitHub's 100 MB per-file limit. If a permanent DOI-backed copy is preferred over relying on a personal Drive account, these videos are natural candidates for deposit on [Zenodo](https://zenodo.org/) alongside the larger `.npy`/`.keras` files.

## Repository structure

```text
repo/
├── assets/                          # Figures used in this README
├── common/                          # Reference copy of the CARLA client utilities
├── data/                            # Reference copy of the Town07 route dataset
└── resultados/
    ├── 01_stanley_town07/
    ├── 02_nmpc_dinamico_town07/
    ├── 03_nmpc_datadriven_town07/
    ├── 04_nmpc_pinn_town07/
    ├── 05_nmpc_pinn_town07_ekf/      # State-estimation (EKF) variant — robustness analysis
    ├── 06_comparacion_final_town07/  # Comparison of the four controllers + final figures
    └── 07_escenarios_sensibilidad/   # Mass/μ/speed sweep in Town04-Scenario 2 and Town06-Scenario 1
        ├── figuras/                  # Final comparison and sensitivity figures (.png/.pdf)
        └── *.ipynb, *.npy            # Analysis notebooks and sweep data
```

Each directory `01`–`05` is **self-contained**: it includes its own control script (`.py`), CARLA client utilities (`HUD.py`, `World.py`, etc.), route dataset (`traj_dataset_Town07.npy`), analysis notebook (`.ipynb`, with generated figures saved as outputs), final run data (`.npy`) and, where applicable, the trained model (`models/*.keras` + `scalers*.pkl`). The copies in `common/` and `data/` are reference copies, provided to make the shared files accessible without opening five experiment directories.

## Local reproduction

1. Install CARLA 0.9.16 and start the server (natively or with the official `carlasim/carla` image).
2. Set up the Python client environment:
   ```bash
   pip install -r requirements.txt
   ```
   The `carla` package is not available on PyPI. Install it from the wheel included with the CARLA 0.9.16 distribution under `PythonAPI/carla/dist/` (for example, `carla-0.9.16-cp312-cp312-win_amd64.whl` for Python 3.12 on Windows; use the wheel matching your Python version and operating system).
3. Enter the result directory you want to run (for example, `resultados/04_nmpc_pinn_town07`) and execute its control script (`python NMPCPINN_Final_Town07.py`) while the CARLA server is running. No additional files need to be copied; each directory contains everything its scripts require.
4. To review the analysis without running CARLA, open the corresponding `.ipynb`; it contains the saved figure outputs.


## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use this code, cite the thesis:

> Segovia Razo, A. F. (2026). *Design of a Control System for a Mobile Vehicle in Emergency Obstacle-Evasion Scenarios*. Master's thesis, Master's Program in Control and Automation, Pontifical Catholic University of Peru.
