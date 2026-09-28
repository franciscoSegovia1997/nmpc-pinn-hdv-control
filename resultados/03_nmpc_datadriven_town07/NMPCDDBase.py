#!/usr/bin/env python

"""NMPC Data-Driven (DD-1) en CARLA Town07.

Lateral    : NMPC CasADi+IPOPT con predictor DD-1 (MLP 2x128 softplus,
             Set D: [steer, ay, yr, vx], L=30, rolling window en CasADi).
             err_pos normalizado por L_WB^2 (adimensional).
             vx propagado con ALPHA_CL dentro del horizonte.
Longitudinal: PI + feedforward estatico (IMC lambda=1.5s).
Estado     : ground truth de CARLA (sin EKF).

Ganancias tuneadas (tuning_analysis_nmpc_dd.ipynb):
    N_H=15  W_POS=1.00  W_YAW=1.00  W_DU=0.25

Guarda: data_Town07_nmpc_dd.npy
"""

from __future__ import print_function

import argparse
import json
import logging
import os
import sys
import math
import multiprocessing as mp
from collections import deque

import numpy as np
import numpy.random as random
import casadi as ca

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from HUD import HUD
from World import World
from KeyboardControl import KeyboardControl

# Constantes fisicas del vehiculo
from pinn1_v7_utils import V_MIN, L_WB, LF, LR


# ==============================================================================
# -- Proceso independiente de gráficas en tiempo real -------------------------
# ==============================================================================

def _plots_worker(q, x_ref_list, y_ref_list, v_ref_list):
    """
    Proceso hijo — muestra 6 paneles matplotlib en tiempo real.
    Recibe mensajes via Queue:
      (closest_index, x_v, y_v, vx, steer_deg, cte)
    Termina al recibir None.
    X_ref, Y_ref y v_ref se dibujan completos desde el inicio como líneas estáticas.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np_p

    x_ref   = np_p.array(x_ref_list)
    y_ref   = np_p.array(y_ref_list)
    v_ref   = np_p.array(v_ref_list)
    N_ref   = len(x_ref)
    idx_ref = np_p.arange(N_ref)

    # ── Paleta clara ──────────────────────────────────────────────────────────
    BG     = '#f5f5f5'
    PANEL  = '#ffffff'
    C_REF  = '#9e9e9e'
    C_VEH  = '#1565c0'
    C_VEL  = '#e65100'
    C_CTE  = '#c62828'
    C_ST   = '#6a1b9a'
    C_GRID = '#cccccc'
    C_TXT  = '#424242'
    C_LBL  = '#212121'

    def _ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(PANEL)
        ax.set_title(title, fontsize=8, loc='left', color=C_LBL, pad=4)
        ax.set_xlabel(xlabel, fontsize=7.5, color=C_TXT)
        ax.set_ylabel(ylabel, fontsize=7.5, color=C_TXT)
        ax.tick_params(labelsize=7, colors=C_TXT)
        ax.grid(True, ls=':', alpha=0.7, color=C_GRID)
        for sp in ax.spines.values():
            sp.set_color(C_GRID)

    # ── Figura y layout ───────────────────────────────────────────────────────
    fig = plt.figure('NMPC-DD — Gráficas en Tiempo Real', figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    # (a) XY — columna izquierda completa
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_xy.set_facecolor(PANEL)
    ax_xy.plot(x_ref, y_ref, color=C_REF, lw=0.8, ls='--', alpha=0.55,
               label='Referencia')
    ax_xy.plot([x_ref[0]], [y_ref[0]], 's', color='#4caf50', ms=8, zorder=6,
               label='Inicio')
    line_xy, = ax_xy.plot([], [], color=C_VEH, lw=1.6, label='NMPC-DD', zorder=4)
    dot_xy,  = ax_xy.plot([], [], 'o', color='#ff4444', ms=7, zorder=7)
    ax_xy.set_aspect('equal', adjustable='datalim')
    ax_xy.set_title('(a) Trayectoria XY', fontsize=8, loc='left',
                    color=C_LBL, pad=4)
    ax_xy.set_xlabel('X [m]', fontsize=7.5, color=C_TXT)
    ax_xy.set_ylabel('Y [m]', fontsize=7.5, color=C_TXT)
    ax_xy.tick_params(labelsize=7, colors=C_TXT)
    ax_xy.grid(True, ls=':', alpha=0.7, color=C_GRID)
    for sp in ax_xy.spines.values(): sp.set_color(C_GRID)
    ax_xy.legend(fontsize=6.5, loc='lower right',
                 facecolor=PANEL, labelcolor=C_LBL, framealpha=0.85)
    ax_xy.invert_yaxis()

    # (b) X — referencia completa estática, vehículo crece en tiempo real
    ax_x = fig.add_subplot(gs[0, 1])
    ax_x.plot(idx_ref, x_ref, color=C_REF, lw=0.9, ls='--', label='X ref')
    line_xv, = ax_x.plot([], [], color=C_VEH, lw=1.2, label='X vehículo')
    _ax(ax_x, '(b) Posición X', 'Punto de ruta', 'X [m]')
    ax_x.legend(fontsize=6.5, facecolor=PANEL, labelcolor=C_LBL, framealpha=0.85)

    # (c) Y — referencia completa estática, vehículo crece en tiempo real
    ax_y = fig.add_subplot(gs[0, 2])
    ax_y.plot(idx_ref, y_ref, color=C_REF, lw=0.9, ls='--', label='Y ref')
    line_yv, = ax_y.plot([], [], color=C_VEH, lw=1.2, label='Y vehículo')
    _ax(ax_y, '(c) Posición Y', 'Punto de ruta', 'Y [m]')
    ax_y.legend(fontsize=6.5, facecolor=PANEL, labelcolor=C_LBL, framealpha=0.85)

    # (d) Velocidad — perfil de referencia completo estático, vehículo crece
    ax_v = fig.add_subplot(gs[1, 1])
    ax_v.plot(idx_ref, v_ref, color=C_VEL, lw=0.9, ls='--', label='v_ref')
    line_vv, = ax_v.plot([], [], color=C_VEH, lw=1.2, label='vx')
    _ax(ax_v, '(d) Velocidad longitudinal', 'Punto de ruta', 'v [m/s]')
    ax_v.legend(fontsize=6.5, facecolor=PANEL, labelcolor=C_LBL, framealpha=0.85)

    # (e) CTE
    ax_cte = fig.add_subplot(gs[1, 2])
    line_cte, = ax_cte.plot([], [], color=C_CTE, lw=1.1, alpha=0.9)
    ax_cte.axhline(0, color=C_REF, lw=0.9, ls='--')
    _ax(ax_cte, '(e) Error lateral (CTE)', 'Punto de ruta', 'CTE [m]')

    # (f) Steer — fila inferior, columnas 1 y 2
    ax_st = fig.add_subplot(gs[2, 1:])
    line_st, = ax_st.plot([], [], color=C_ST, lw=1.1, alpha=0.9)
    ax_st.axhline(0, color=C_REF, lw=0.9, ls='--')
    _ax(ax_st, '(f) Ángulo de dirección', 'Punto de ruta', 'Steer [°]')

    plt.tight_layout(pad=0.4)
    plt.ion()
    plt.show(block=False)

    # ── Buffers de datos (solo vehículo) ─────────────────────────────────────
    MAXPTS = 3000
    idxs   = []
    xs_v   = [];  ys_v   = []
    vxs    = []
    ctes   = [];  steers = []

    # ── Loop principal del proceso ────────────────────────────────────────────
    while True:
        got = False

        # Vaciar todos los mensajes disponibles sin bloquear
        while True:
            try:
                msg = q.get_nowait()
            except Exception:
                break

            if msg is None:
                plt.close('all')
                return

            cidx, xv, yv, vx, steer_deg, cte = msg
            idxs.append(cidx)
            xs_v.append(xv);   ys_v.append(yv)
            vxs.append(vx)
            ctes.append(cte);  steers.append(steer_deg)
            got = True

        # Truncar si los buffers crecen demasiado
        if len(idxs) > MAXPTS:
            idxs   = idxs[-MAXPTS:]
            xs_v   = xs_v[-MAXPTS:];  ys_v   = ys_v[-MAXPTS:]
            vxs    = vxs[-MAXPTS:]
            ctes   = ctes[-MAXPTS:];  steers = steers[-MAXPTS:]

        if got and idxs:
            # XY
            line_xy.set_xdata(xs_v);           line_xy.set_ydata(ys_v)
            dot_xy.set_xdata([xs_v[-1]]);       dot_xy.set_ydata([ys_v[-1]])
            ax_xy.relim();                       ax_xy.autoscale_view()

            # X — solo vehículo (ref es estática)
            line_xv.set_xdata(idxs);            line_xv.set_ydata(xs_v)
            ax_x.relim();                        ax_x.autoscale_view()

            # Y — solo vehículo
            line_yv.set_xdata(idxs);            line_yv.set_ydata(ys_v)
            ax_y.relim();                        ax_y.autoscale_view()

            # Velocidad — solo vehículo
            line_vv.set_xdata(idxs);            line_vv.set_ydata(vxs)
            ax_v.relim();                        ax_v.autoscale_view()

            # CTE
            line_cte.set_xdata(idxs);           line_cte.set_ydata(ctes)
            ax_cte.relim();                      ax_cte.autoscale_view()

            # Steer
            line_st.set_xdata(idxs);            line_st.set_ydata(steers)
            ax_st.relim();                       ax_st.autoscale_view()

            fig.canvas.draw_idle()

        try:
            fig.canvas.flush_events()
            plt.pause(0.016)   # ~60 Hz — proceso separado, sin conflicto con pygame
        except Exception:
            return   # ventana cerrada por el usuario — salir limpiamente


# ==============================================================================
# -- Diálogos tkinter ----------------------------------------------------------
# ==============================================================================

def show_start_dialog():
    """Muestra ventana de inicio. Bloquea hasta que el usuario presiona Iniciar."""
    import tkinter as tk

    root = tk.Tk()
    root.title("NMPC Data-Driven — Town07")
    root.resizable(False, False)
    root.configure(bg='#0d1b2a')
    root.attributes('-topmost', True)
    root.protocol("WM_DELETE_WINDOW", lambda: None)

    W, H = 540, 215
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f'{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}')

    tk.Label(root,
             text="NMPC Data-Driven  |  Bus Fusorosa  |  Town07",
             font=('Helvetica', 13, 'bold'),
             fg='#4fc3f7', bg='#0d1b2a').pack(pady=(24, 8))

    tk.Label(root,
             text="CARLA está listo.\n"
                  "El vehículo se encuentra en su posición inicial.\n"
                  "Presione Iniciar para comenzar la simulación.",
             font=('Helvetica', 11),
             fg='#cfd8dc', bg='#0d1b2a', justify='center').pack(pady=(0, 18))

    def _start():
        root.destroy()

    tk.Button(root,
              text="   ▶   Iniciar Simulación con NMPC Data-Driven   ",
              command=_start,
              font=('Helvetica', 11, 'bold'),
              bg='#2e7d32', fg='white',
              activebackground='#1b5e20', activeforeground='white',
              relief='flat', cursor='hand2',
              padx=16, pady=9).pack()

    root.mainloop()


def show_finish_dialog():
    """Muestra ventana de fin con cuenta regresiva 5→0."""
    import tkinter as tk

    root = tk.Tk()
    root.title("Simulación Finalizada")
    root.resizable(False, False)
    root.configure(bg='#0d1b2a')
    root.attributes('-topmost', True)

    W, H = 420, 200
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f'{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}')

    tk.Label(root,
             text="✔   Simulación Terminada",
             font=('Helvetica', 15, 'bold'),
             fg='#66bb6a', bg='#0d1b2a').pack(pady=(28, 10))

    count_var = tk.StringVar(value="Cerrando en 5 segundos...")
    tk.Label(root, textvariable=count_var,
             font=('Helvetica', 12),
             fg='#90a4ae', bg='#0d1b2a').pack(pady=4)

    bar_canvas = tk.Canvas(root, width=340, height=14,
                           bg='#1a2a3a', highlightthickness=0)
    bar_canvas.pack(pady=(12, 0))
    bar_rect = bar_canvas.create_rectangle(0, 0, 340, 14,
                                           fill='#2e7d32', outline='')

    def _countdown(n):
        if n <= 0:
            root.destroy()
            return
        count_var.set(f"Cerrando en {n} segundo{'s' if n != 1 else ''}...")
        bar_canvas.coords(bar_rect, 0, 0, int(340 * n / 5), 14)
        root.after(1000, _countdown, n - 1)

    root.after(100, _countdown, 5)
    root.mainloop()


# ==============================================================================
# -- Game Loop -----------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world           = None
    traffic_manager = None
    sim_completed   = [False]
    plot_proc       = None
    plot_q          = None

    # Log inicializado antes del try para que finally nunca falle
    log_x = [];          log_y = []
    log_speed = [];      log_target_speed = []
    log_time = [];       log_steer = [];       log_yaw = []
    log_crosstrack_error = []
    log_vx = [];  log_vy = [];  log_ax = [];  log_ay = []
    log_yaw_rate = [];   log_throttle = [];    log_brake = []
    log_roll = [];       log_pitch = []
    log_roll_rate = [];  log_pitch_rate = [];  log_steer_rad = []

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        client.load_world('Town07_Opt')

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()
        sim_world.unload_map_layer(carla.MapLayer.Buildings)
        sim_world.unload_map_layer(carla.MapLayer.Decals)
        sim_world.unload_map_layer(carla.MapLayer.Foliage)
        sim_world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        sim_world.unload_map_layer(carla.MapLayer.Particles)
        sim_world.unload_map_layer(carla.MapLayer.Props)
        sim_world.unload_map_layer(carla.MapLayer.StreetLights)
        sim_world.unload_map_layer(carla.MapLayer.Walls)

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('NMPC-DD Final — Town07')

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # ── CARGAR DATASET TOWN07 ──────────────────────────────────────────────
        dataset  = np.load("traj_dataset_Town07.npy", allow_pickle=True).item()
        x_traj   = dataset["x"]
        y_traj   = dataset["y"]
        yaw_traj = np.arctan2(np.sin(dataset["yaw"]), np.cos(dataset["yaw"]))
        v_traj   = dataset["v_18_25"]

        route_xy = np.stack((x_traj, y_traj), axis=1)
        N_PATH   = len(x_traj)
        spacing  = float(np.mean(np.linalg.norm(
            np.diff(route_xy, axis=0), axis=1)))

        print(f"Trayectoria Town07: {N_PATH} pts  spacing={spacing:.4f}m")
        print(f"v_18_25: min={v_traj.min():.1f} max={v_traj.max():.1f} m/s")

        # ── SPAWN ─────────────────────────────────────────────────────────────
        spawn_tf = carla.Transform(
            carla.Location(x=float(x_traj[0]), y=float(y_traj[0]), z=0.5),
            carla.Rotation(yaw=float(np.degrees(yaw_traj[0])))
        )
        world.player.set_transform(spawn_tf)

        # Renderizar frame inicial
        init_clock = pygame.time.Clock()
        world.world.tick()
        world.tick(init_clock)
        world.render(display)
        pygame.display.flip()

        # ── DIÁLOGO DE INICIO ─────────────────────────────────────────────────
        show_start_dialog()
        pygame.event.clear()

        # ── INICIAR PROCESO DE GRÁFICAS ───────────────────────────────────────
        try:
            plot_q    = mp.Queue(maxsize=60)
            plot_proc = mp.Process(
                target=_plots_worker,
                args=(plot_q, x_traj.tolist(), y_traj.tolist(), v_traj.tolist()),
                daemon=True,
                name='live-plots'
            )
            plot_proc.start()
            print("Proceso de gráficas iniciado (PID:", plot_proc.pid, ")")
        except Exception as e:
            print(f"[WARN] No se pudo iniciar el proceso de gráficas: {e}")
            plot_proc = None
            plot_q    = None

        # ── PARAMETROS GENERALES ──────────────────────────────────────────────
        dt            = 0.05
        closest_index = 0

        # PI LONGITUDINAL + FEEDFORWARD (IMC lambda=1.5s)
        Ks_ff    = 33.291680
        K0_ff    = -8.575180
        k_p      = 0.247137
        k_i      = 0.020025
        integral = 0.0

        # ── PARAMETROS NMPC-DD ────────────────────────────────────────────────
        N_H       = 15
        W_POS     = 1.00    # adimensional (err_pos / L_WB^2)
        W_YAW     = 1.00    # rad^2
        W_DU      = 0.25    # rad^2
        DELTA_MAX = 1.22    # rad
        ALPHA_CL  = float(np.exp(-dt / 1.5))

        # ── CARGAR MODELO DD-1 ────────────────────────────────────────────────
        _base = os.path.dirname(os.path.abspath(__file__))
        DD1_CFG_PATH = os.path.join(_base, 'models', 'DD1_tf',
                                    'DD1_tf_inference_config.json')
        DD1_W_PATH   = os.path.join(_base, 'models', 'DD1_tf',
                                    'DD1_tf_best_weights.npz')

        with open(DD1_CFG_PATH) as f:
            dd1_cfg = json.load(f)

        L_DD    = dd1_cfg['best_L']           # 30
        FEAT    = dd1_cfg['feature_names']    # [steer, ay, yr, vx]
        IN_DIM  = dd1_cfg['input_dim']        # 120
        FBOUNDS = dd1_cfg['feature_bounds']
        SY_MEAN  = np.array(dd1_cfg['scaler_Y_mean'],  dtype=np.float32)
        SY_SCALE = np.array(dd1_cfg['scaler_Y_scale'], dtype=np.float32)

        MID  = {f: (FBOUNDS[f][0] + FBOUNDS[f][1]) / 2.0 for f in FEAT}
        HALF = {f: (FBOUNDS[f][1] - FBOUNDS[f][0]) / 2.0 for f in FEAT}

        _nw = np.load(DD1_W_PATH)
        W1 = _nw['W1']; b1 = _nw['b1']
        W2 = _nw['W2']; b2 = _nw['b2']
        W3 = _nw['W3']; b3 = _nw['b3']

        # p layout: [vx,x,y,psi,vy,yr,u_prev | vref(N_H) | refs(3*N_H) | win(IN_DIM) | W_POS,W_YAW,W_DU]
        _N_P  = 10 + 4 * N_H + IN_DIM   # 190 para N_H=15
        IDX_W = 7  + 4 * N_H + IN_DIM   # 187

        print(f"NMPC-DD: N_H={N_H}  W_POS={W_POS}  W_YAW={W_YAW}  W_DU={W_DU}")
        print(f"DD-1: L={L_DD}  IN_DIM={IN_DIM}  _N_P={_N_P}  IDX_W={IDX_W}")

        # ── BUFFER DD-1 (rolling window normalizado) ──────────────────────────
        _vx0_n  = (0.0 - MID['vx']) / HALF['vx']
        _dd1_buf = deque([[0.0, 0.0, 0.0, _vx0_n]] * L_DD, maxlen=L_DD)

        def dd1_push(steer_rad, ay, yr, vx):
            _dd1_buf.append([
                (steer_rad - MID['steer']) / HALF['steer'],
                (ay        - MID['ay'])    / HALF['ay'],
                (yr        - MID['yr'])    / HALF['yr'],
                (vx        - MID['vx'])    / HALF['vx'],
            ])

        def dd1_window_flat():
            return np.array(_dd1_buf, dtype=np.float32).flatten()

        # ── HELPER: REFERENCIAS EN EL HORIZONTE ───────────────────────────────
        def get_horizon_refs(path_idx, vx0):
            vref_seq = np.empty(N_H)
            refs     = []
            idx = path_idx
            vx  = vx0
            for k in range(N_H):
                n_adv        = max(1, int(vx * dt / spacing))
                idx          = min(idx + n_adv, N_PATH - 1)
                vref_seq[k]  = float(v_traj[idx])
                refs.append((float(x_traj[idx]),
                              float(y_traj[idx]),
                              float(yaw_traj[idx])))
                vx = ALPHA_CL * vx + (1.0 - ALPHA_CL) * vref_seq[k]
            return vref_seq, refs

        # ── CONSTRUIR SOLVER NMPC-DD ──────────────────────────────────────────
        def build_nmpc_solver():
            # Pesos MLP como DM (constantes en el NLP)
            _W1_dm = ca.DM(W1);  _b1_dm = ca.DM(b1.reshape(-1, 1))
            _W2_dm = ca.DM(W2);  _b2_dm = ca.DM(b2.reshape(-1, 1))
            _W3_dm = ca.DM(W3);  _b3_dm = ca.DM(b3.reshape(-1, 1))
            _sY_sc = ca.DM(SY_SCALE.reshape(-1, 1))
            _sY_mn = ca.DM(SY_MEAN.reshape(-1, 1))

            def _softplus(z):
                return ca.fmax(z, 0.0) + ca.log(1.0 + ca.exp(-ca.fabs(z)))

            def _dd1_forward_ca(win_vec):
                h1  = _softplus(ca.mtimes(_W1_dm, win_vec) + _b1_dm)
                h2  = _softplus(ca.mtimes(_W2_dm, h1)      + _b2_dm)
                y_n = ca.mtimes(_W3_dm, h2)                + _b3_dm
                y_p = y_n * _sY_sc + _sY_mn
                return y_p[0, 0], y_p[1, 0]

            u_sym = ca.MX.sym('u', N_H)
            p     = ca.MX.sym('p', _N_P)

            vx_c   = p[0];  x_c  = p[1];  y_c   = p[2];  psi_c  = p[3]
            vy_c   = p[4];  yr_c = p[5];  u_prev = p[6]

            win_c    = ca.reshape(p[7 + 4*N_H : 7 + 4*N_H + IN_DIM], (IN_DIM, 1))
            W_POS_c  = p[IDX_W];  W_YAW_c = p[IDX_W + 1];  W_DU_c = p[IDX_W + 2]

            cost = 0.0
            for k in range(N_H):
                vref_k  = p[7 + k]
                x_ref   = p[7 + N_H + k * 3]
                y_ref   = p[7 + N_H + k * 3 + 1]
                yaw_ref = p[7 + N_H + k * 3 + 2]
                u_k     = u_sym[k]

                # Rama cinematica (vx < V_MIN)
                yr_kin  = (vx_c / L_WB) * ca.tan(u_k)
                x_kin   = x_c   + vx_c * ca.cos(psi_c) * dt
                y_kin   = y_c   + vx_c * ca.sin(psi_c) * dt
                psi_kin = psi_c + yr_kin * dt

                # Rama DD-1 (vx >= V_MIN): rolling window + inferencia MLP
                ay_c    = vx_c * yr_c
                s_n     = (u_k  - MID['steer']) / HALF['steer']
                a_n     = (ay_c - MID['ay'])    / HALF['ay']
                r_n     = (yr_c - MID['yr'])    / HALF['yr']
                v_n     = (vx_c - MID['vx'])    / HALF['vx']
                win_new = ca.vertcat(win_c[4:IN_DIM, :], ca.vertcat(s_n, a_n, r_n, v_n))

                vy_dd, yr_dd = _dd1_forward_ca(win_new)

                vx_g    = vx_c * ca.cos(psi_c) - vy_c * ca.sin(psi_c)
                vy_g    = vx_c * ca.sin(psi_c) + vy_c * ca.cos(psi_c)
                x_dyn   = x_c   + vx_g * dt
                y_dyn   = y_c   + vy_g * dt
                psi_dyn = psi_c + yr_c  * dt

                cond   = vx_c < V_MIN
                x_c    = ca.if_else(cond, x_kin,   x_dyn)
                y_c    = ca.if_else(cond, y_kin,   y_dyn)
                psi_c  = ca.if_else(cond, psi_kin, psi_dyn)
                vy_c   = ca.if_else(cond, 0.0,     vy_dd)
                yr_c   = ca.if_else(cond, yr_kin,  yr_dd)
                win_c  = win_new

                vx_c   = ALPHA_CL * vx_c + (1.0 - ALPHA_CL) * vref_k

                err_pos = ((x_c - x_ref)**2 + (y_c - y_ref)**2) / (L_WB**2)
                err_yaw = ca.atan2(ca.sin(psi_c - yaw_ref), ca.cos(psi_c - yaw_ref))
                du      = u_k - u_prev
                cost   += W_POS_c * err_pos + W_YAW_c * err_yaw**2 + W_DU_c * du**2
                u_prev  = u_k

            nlp  = {'x': u_sym, 'f': cost, 'p': p}
            opts = {
                'ipopt.print_level':           0,
                'print_time':                  0,
                'ipopt.max_iter':             50,
                'ipopt.tol':                1e-4,
                'ipopt.acceptable_tol':     1e-3,
                'ipopt.warm_start_init_point': 'yes',
                'error_on_fail':             False,
            }
            return ca.nlpsol('nmpc_dd', 'ipopt', nlp, opts)

        print("Compilando solver NMPC-DD (puede tardar 1-3 min)...")
        _nmpc_solver = build_nmpc_solver()
        print("Solver NMPC-DD listo")

        # ── FUNCION DE RESOLUCION NMPC ────────────────────────────────────────
        def solve_nmpc(state, path_idx, win_flat, u_warm=None, u_prev_applied=0.0):
            x, y, psi, vy, yr, vx = state
            vx_p   = max(float(vx), 1.0)
            u_last = float(u_prev_applied)

            u0 = np.clip(
                u_warm if u_warm is not None else np.full(N_H, u_last),
                -DELTA_MAX, DELTA_MAX)

            vref_seq, refs = get_horizon_refs(path_idx, vx_p)

            p_val = np.concatenate([
                [vx_p, x, y, psi, vy, yr, u_last],
                vref_seq,
                np.array(refs).flatten(),
                win_flat.flatten(),
                [W_POS, W_YAW, W_DU],
            ])

            try:
                sol   = _nmpc_solver(
                    x0=u0, lbx=[-DELTA_MAX]*N_H, ubx=[DELTA_MAX]*N_H, p=p_val)
                u_opt = np.clip(np.array(sol['x']).flatten(), -DELTA_MAX, DELTA_MAX)
            except Exception:
                u_opt = u0

            return float(u_opt[0]), u_opt

        # ── ESTADO INICIAL ────────────────────────────────────────────────────
        u_warm         = None
        steer_last_rad = 0.0
        simulation_time = 0.0
        print("Listo para correr en Town07")

        clock = pygame.time.Clock()

        # ── BUCLE PRINCIPAL ───────────────────────────────────────────────────
        while True:
            clock.tick(20)   # limita a 20 Hz → 1 s simulado = 1 s real (dt=0.05 s)
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            # LECTURA GROUND TRUTH CARLA
            velocity     = world.player.get_velocity()
            acceleration = world.player.get_acceleration()
            angular_vel  = world.player.get_angular_velocity()
            tf_          = world.player.get_transform()

            current_x   = tf_.location.x
            current_y   = tf_.location.y
            current_yaw = math.radians(tf_.rotation.yaw)
            roll_angle  = math.radians(tf_.rotation.roll)
            pitch_angle = math.radians(tf_.rotation.pitch)

            vx_local =  math.cos(current_yaw)*velocity.x + math.sin(current_yaw)*velocity.y
            vy_local = -math.sin(current_yaw)*velocity.x + math.cos(current_yaw)*velocity.y
            ax_local =  math.cos(current_yaw)*acceleration.x + math.sin(current_yaw)*acceleration.y
            ay_local = -math.sin(current_yaw)*acceleration.x + math.cos(current_yaw)*acceleration.y
            yaw_rate   = math.radians(angular_vel.z)
            roll_rate  = math.radians(angular_vel.x)
            pitch_rate = math.radians(angular_vel.y)

            control = carla.VehicleControl()

            # WAYPOINT MAS CERCANO
            current_pos  = np.array([current_x, current_y])
            search_start = max(closest_index - 10, 0)
            search_end   = min(closest_index + 300, N_PATH)
            local_dists  = np.linalg.norm(
                route_xy[search_start:search_end] - current_pos, axis=1)
            closest_index = search_start + np.argmin(local_dists)

            if closest_index >= N_PATH - 50:
                sim_completed[0] = True
                print("Trayectoria completada.")
                break

            # CONTROL LONGITUDINAL: PI + FF
            target_speed = float(v_traj[closest_index])
            vlon  = vx_local
            error = target_speed - vlon

            u_ff = float(np.clip((target_speed - K0_ff) / Ks_ff, 0.0, 1.0))
            integral += k_i * dt * error
            u_cmd  = u_ff + k_p * error + integral
            u_sat  = float(np.clip(u_cmd, -1.0, 1.0))
            if u_cmd != u_sat:
                integral -= k_i * dt * error

            if u_sat >= 0.0:
                control.throttle = float(np.clip(u_sat, 0.0, 0.90))
                control.brake    = 0.0
            else:
                control.throttle = 0.0
                control.brake    = float(np.clip(-u_sat, 0.0, 0.5))
            control.hand_brake = False

            # CONTROL LATERAL: NMPC DD-1
            win_flat  = dd1_window_flat()
            state_nmpc = [current_x, current_y, current_yaw,
                          vy_local, yaw_rate, vx_local]
            steer_rad, u_opt = solve_nmpc(state_nmpc, closest_index, win_flat,
                                          u_warm, steer_last_rad)

            u_warm         = np.append(u_opt[1:], u_opt[-1])
            steer_last_rad = steer_rad

            # Actualizar buffer DD-1 con el estado actual y el steer aplicado
            dd1_push(steer_rad, ay_local, yaw_rate, vx_local)

            cmd_steer = float(np.clip(steer_rad / DELTA_MAX, -1.0, 1.0))

            # CTE
            psi_r = yaw_traj[closest_index]
            crosstrack_error = float(
                (current_y - y_traj[closest_index]) * math.cos(psi_r) -
                (current_x - x_traj[closest_index]) * math.sin(psi_r))

            control.steer             = cmd_steer
            control.manual_gear_shift = False
            world.player.apply_control(control)

            # ── ENVIAR DATOS AL PROCESO DE GRÁFICAS (cada paso = 20 Hz) ──────
            if plot_q is not None:
                try:
                    plot_q.put_nowait((
                        int(closest_index),
                        float(current_x),
                        float(current_y),
                        float(vx_local),
                        float(np.degrees(np.clip(steer_rad, -DELTA_MAX, DELTA_MAX))),
                        float(crosstrack_error)
                    ))
                except Exception:
                    pass   # queue llena: descartar este punto, no bloquear

            # LOG
            log_x.append(current_x)
            log_y.append(current_y)
            log_speed.append(vlon)
            log_target_speed.append(target_speed)
            log_time.append(simulation_time)
            log_steer.append(cmd_steer)
            log_yaw.append(current_yaw)
            log_crosstrack_error.append(crosstrack_error)
            log_vx.append(vx_local)
            log_vy.append(vy_local)
            log_ax.append(ax_local)
            log_ay.append(ay_local)
            log_yaw_rate.append(yaw_rate)
            log_throttle.append(control.throttle)
            log_brake.append(control.brake)
            log_roll.append(roll_angle)
            log_pitch.append(pitch_angle)
            log_roll_rate.append(roll_rate)
            log_pitch_rate.append(pitch_rate)
            log_steer_rad.append(steer_rad)
            simulation_time += dt

    finally:
        # ── Guardar datos ─────────────────────────────────────────────────────
        if log_x:
            data_out = {
                "x":                np.array(log_x),
                "y":                np.array(log_y),
                "time":             np.array(log_time),
                "steer":            np.array(log_steer),
                "steer_rad":        np.array(log_steer_rad),
                "throttle":         np.array(log_throttle),
                "brake":            np.array(log_brake),
                "target_speed":     np.array(log_target_speed),
                "vx":               np.array(log_vx),
                "vy":               np.array(log_vy),
                "speed":            np.array(log_speed),
                "ax":               np.array(log_ax),
                "ay":               np.array(log_ay),
                "yaw":              np.array(log_yaw),
                "yaw_rate":         np.array(log_yaw_rate),
                "roll":             np.array(log_roll),
                "pitch":            np.array(log_pitch),
                "roll_rate":        np.array(log_roll_rate),
                "pitch_rate":       np.array(log_pitch_rate),
                "crosstrack_error": np.array(log_crosstrack_error),
            }
            np.save("data_Town07_nmpc_dd.npy", data_out)
            print("Datos guardados: data_Town07_nmpc_dd.npy ✅")
        else:
            print("Sin datos que guardar.")

        # ── Cerrar proceso de gráficas ────────────────────────────────────────
        if plot_q is not None and plot_proc is not None:
            try:
                plot_q.put_nowait(None)
            except Exception:
                pass
            plot_proc.join(timeout=3.0)
            if plot_proc.is_alive():
                plot_proc.terminate()
                plot_proc.join(timeout=1.0)

        # ── Limpiar CARLA y pygame ────────────────────────────────────────────
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            if traffic_manager is not None:
                traffic_manager.set_synchronous_mode(False)
            world.destroy()
        pygame.quit()

        # ── Diálogo de fin ────────────────────────────────────────────────────
        if sim_completed[0]:
            show_finish_dialog()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA NMPC Data-Driven DD-1 -- Town07')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int)
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='640x480')
    argparser.add_argument('--sync', action='store_true', default=True)
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*')
    argparser.add_argument('--generation', metavar='G', default='2')
    argparser.add_argument('-l', '--loop', action='store_true', dest='loop')
    argparser.add_argument('-a', '--agent', type=str,
                           choices=["Behavior", "Basic", "Constant"],
                           default="Behavior")
    argparser.add_argument('-b', '--behavior', type=str,
                           choices=["cautious", "normal", "aggressive"],
                           default='normal')
    argparser.add_argument('-s', '--seed', default=None, type=int)

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    mp.freeze_support()   # necesario en Windows para ejecutables empaquetados
    main()
