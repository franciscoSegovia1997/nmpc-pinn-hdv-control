#!/usr/bin/env python

"""NMPC + PINN1-v7 vehicle control in CARLA (Town07).
Lateral: NMPC CasADi+IPOPT con PINN1-v7 linealizado como parametros.
Longitudinal: PI + feedforward estatico (IMC lambda=1.5s).
Ganancias optimas del grid search (con err_pos/L_WB^2): N_H=15, W_POS=0.25, W_YAW=0.25, W_DU=1.00
"""

from __future__ import print_function

import argparse
import logging
import os
import sys
import math
import time
import multiprocessing as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import numpy.random as random
import tensorflow as tf
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
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import butter, sosfilt, sosfilt_zi, bessel

from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner

from pinn1_v7_utils import PINN1v7Predictor, V_MIN, L_WB, LF, LR


# ==============================================================================
# -- Proceso independiente de gráficas en tiempo real -------------------------
# ==============================================================================

def _plots_worker(q, x_ref_list, y_ref_list, v_ref_list):
    """
    Proceso hijo — muestra 8 paneles matplotlib en tiempo real.
    Recibe mensajes via Queue:
      (closest_index, x_v, y_v, vx, steer_deg, cte, cf, cr)
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
    fig = plt.figure('NMPC-PINN — Gráficas en Tiempo Real', figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)
    gs_left = gs[:, 0].subgridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.65)

    # (a) XY — mitad superior de la columna izquierda
    ax_xy = fig.add_subplot(gs_left[0])
    ax_xy.set_facecolor(PANEL)
    ax_xy.plot(x_ref, y_ref, color=C_REF, lw=0.8, ls='--', alpha=0.55,
               label='Referencia')
    ax_xy.plot([x_ref[0]], [y_ref[0]], 's', color='#4caf50', ms=8, zorder=6,
               label='Inicio')
    line_xy, = ax_xy.plot([], [], color=C_VEH, lw=1.6, label='NMPC-PINN', zorder=4)
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

    # (g) Cr — rigidez lateral trasera inferida por PINN
    ax_cr = fig.add_subplot(gs_left[1])
    line_cr, = ax_cr.plot([], [], color='#558b2f', lw=1.1, alpha=0.9)
    _ax(ax_cr, '(g) Cr inferido [N/rad]', 'Punto de ruta', 'Cr [N/rad]')

    # (h) Cf — rigidez lateral delantera inferida por PINN
    ax_cf = fig.add_subplot(gs_left[2])
    line_cf, = ax_cf.plot([], [], color='#00838f', lw=1.1, alpha=0.9)
    _ax(ax_cf, '(h) Cf inferido [N/rad]', 'Punto de ruta', 'Cf [N/rad]')

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
    cfs    = [];  crs    = []

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

            cidx, xv, yv, vx, steer_deg, cte, cf, cr = msg
            idxs.append(cidx)
            xs_v.append(xv);   ys_v.append(yv)
            vxs.append(vx)
            ctes.append(cte);  steers.append(steer_deg)
            cfs.append(cf);    crs.append(cr)
            got = True

        # Truncar si los buffers crecen demasiado
        if len(idxs) > MAXPTS:
            idxs   = idxs[-MAXPTS:]
            xs_v   = xs_v[-MAXPTS:];  ys_v   = ys_v[-MAXPTS:]
            vxs    = vxs[-MAXPTS:]
            ctes   = ctes[-MAXPTS:];  steers = steers[-MAXPTS:]
            cfs    = cfs[-MAXPTS:];   crs    = crs[-MAXPTS:]

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

            # Cr / Cf inferidos por PINN
            line_cr.set_xdata(idxs);            line_cr.set_ydata(crs)
            ax_cr.relim();                       ax_cr.autoscale_view()

            line_cf.set_xdata(idxs);            line_cf.set_ydata(cfs)
            ax_cf.relim();                       ax_cf.autoscale_view()

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
    root.title("NMPC PINN1-v7 — Town07")
    root.resizable(False, False)
    root.configure(bg='#0d1b2a')
    root.attributes('-topmost', True)
    root.protocol("WM_DELETE_WINDOW", lambda: None)

    W, H = 540, 215
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f'{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}')

    tk.Label(root,
             text="NMPC PINN1-v7  |  Bus Fusorosa  |  Town07",
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
              text="   ▶   Iniciar Simulación con NMPC PINN1-v7   ",
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
        pygame.display.set_caption('NMPC-PINN Final — Town07')

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # ── CARGAR DATASET TOWN07 ──────────────────────────────────────────────
        dataset  = np.load("traj_dataset_Town07.npy", allow_pickle=True).item()
        x_traj   = dataset["x"]
        y_traj   = dataset["y"]
        yaw_traj = np.arctan2(np.sin(dataset["yaw"]), np.cos(dataset["yaw"]))
        curv     = dataset["curvatura"]
        v_traj   = dataset["v_18_25"]

        route_xy = np.stack((x_traj, y_traj), axis=1)
        N_PATH   = len(x_traj)
        spacing  = float(np.mean(np.linalg.norm(
            np.diff(route_xy, axis=0), axis=1)))

        print(f"Trayectoria: {N_PATH} pts  spacing={spacing:.3f}m")
        print(f"Velocidad min={v_traj.min():.1f} max={v_traj.max():.1f} m/s")

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
        dt            = 0.05          # coincide con fixed_delta_seconds y PINN_DT
        closest_index = 0

        # PI LONGITUDINAL + FEEDFORWARD (IMC, lambda=1.5s, masa=5300kg)
        Ks_ff    = 33.291680
        K0_ff    = -8.575180
        k_p      = 0.247137
        k_i      = 0.020025
        integral = 0.0

        # ── NMPC PARAMETROS ───────────────────────────────────────────────────
        N_H       = 15
        W_POS     = 0.25
        W_YAW     = 0.25
        W_DU      = 1.0
        DELTA_MAX = 1.22
        EPS_LIN   = 5e-3
        MASS_VEH  = 5300.0
        ALPHA_CL  = float(np.exp(-dt / 1.5))
        _N_P      = 18 + 4 * N_H      # 78 para N_H=15

        # ── CARGAR PINN1-v7 ───────────────────────────────────────────────────
        pinn = PINN1v7Predictor(
            'models/PINN1_v7/PINN1_v7_best.keras',
            mass_default=MASS_VEH)
        print(f"PINN1-v7: {pinn.model.count_params():,} params  OK")

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(1, 160), dtype=tf.float32),
            tf.TensorSpec(shape=(1,   5), dtype=tf.float32),
        ])
        def pinn_infer(w, s):
            return pinn.model([w, s], training=False)

        # warmup del grafo tf.function
        _ = pinn_infer(
            np.zeros((1, 160), dtype=np.float32),
            np.array([[0., 0., 15., 0., MASS_VEH]], dtype=np.float32)).numpy()

        # ── HELPER: LINEALIZACION PINN ────────────────────────────────────────
        def linearize_pinn(fw_tf, vy0, yr0, vx, d0):
            def _call(vy, yr, d):
                out = pinn_infer(
                    fw_tf,
                    tf.constant([[vy, yr, vx, d, MASS_VEH]], dtype=tf.float32)
                ).numpy()
                return out[0, 0], out[0, 1]

            vy_n,   yr_n   = _call(vy0,           yr0,          d0)
            vy_pvy, yr_pvy = _call(vy0 + EPS_LIN, yr0,          d0)
            vy_pyr, yr_pyr = _call(vy0,           yr0 + EPS_LIN, d0)
            vy_pd,  yr_pd  = _call(vy0,           yr0,          d0 + EPS_LIN)

            c_  = np.array([vy_n, yr_n])
            A_  = np.array([
                [(vy_pvy - vy_n) / EPS_LIN, (vy_pyr - vy_n) / EPS_LIN],
                [(yr_pvy - yr_n) / EPS_LIN, (yr_pyr - yr_n) / EPS_LIN],
            ])
            B_  = np.array([(vy_pd - vy_n) / EPS_LIN,
                             (yr_pd - yr_n) / EPS_LIN])
            return A_, B_, c_, np.array([vy0, yr0]), float(d0)

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

        # ── CONSTRUIR SOLVER NMPC (una sola vez) ──────────────────────────────
        def build_nmpc_solver():
            # Layout de p (_N_P = 18 + 4*N_H):
            # [0..5]              : x,y,psi,vy,yr,vx
            # [6]                 : u_last
            # [7..10]             : A row-major
            # [11..12]            : B
            # [13..14]            : c
            # [15..16]            : lat0
            # [17]                : d0_lin
            # [18..18+N_H-1]      : vref_seq
            # [18+N_H..18+4*N_H-1]: refs x,y,yaw (3*N_H)
            u_sym = ca.MX.sym('u', N_H)
            p     = ca.MX.sym('p', _N_P)

            x_c    = p[0]; y_c = p[1]; psi_c = p[2]
            vy_c   = p[3]; yr_c = p[4]; vx_c  = p[5]
            u_prev = p[6]

            A_c    = ca.reshape(p[7:11], 2, 2).T   # col-major -> row-major
            B_c    = p[11:13]
            c_c    = p[13:15]
            lat0_c = p[15:17]
            d0_c   = p[17]

            cost = 0.0
            for k in range(N_H):
                vref_k  = p[18 + k]
                x_ref   = p[18 + N_H + k * 3]
                y_ref   = p[18 + N_H + k * 3 + 1]
                yaw_ref = p[18 + N_H + k * 3 + 2]
                u_k     = u_sym[k]

                # Propagacion de posicion (estado al inicio del paso k)
                vx_g  = vx_c * ca.cos(psi_c) - vy_c * ca.sin(psi_c)
                vy_g  = vx_c * ca.sin(psi_c) + vy_c * ca.cos(psi_c)
                x_c   = x_c   + vx_g * dt
                y_c   = y_c   + vy_g * dt
                psi_c = psi_c + yr_c * dt

                # Modelo lateral: PINN linealizado o cinematico (vx < V_MIN)
                lat_cur  = ca.vertcat(vy_c, yr_c)
                lat_next = c_c + A_c @ (lat_cur - lat0_c) + B_c * (u_k - d0_c)
                yr_kin   = (vx_c / L_WB) * ca.tan(u_k)

                vy_c = ca.if_else(vx_c < V_MIN, 0.0,    lat_next[0])
                yr_c = ca.if_else(vx_c < V_MIN, yr_kin, lat_next[1])

                # Modelo longitudinal (IMC)
                vx_c = ALPHA_CL * vx_c + (1.0 - ALPHA_CL) * vref_k

                # Costo: posicion XY + yaw + suavidad steer
                err_pos  = ((x_c - x_ref) ** 2 + (y_c - y_ref) ** 2) / (L_WB ** 2)
                err_yaw  = ca.atan2(ca.sin(psi_c - yaw_ref), ca.cos(psi_c - yaw_ref))
                du       = u_k - u_prev
                cost    += W_POS * err_pos + W_YAW * err_yaw ** 2 + W_DU * du ** 2
                u_prev   = u_k

            nlp  = {'x': u_sym, 'f': cost, 'p': p}
            opts = {
                'ipopt.print_level':           0,
                'print_time':                  0,
                'ipopt.max_iter':             50,
                'ipopt.tol':                1e-4,
                'ipopt.acceptable_tol':     1e-3,
                'ipopt.warm_start_init_point': 'yes',
            }
            return ca.nlpsol('nmpc', 'ipopt', nlp, opts)

        _nmpc_solver = build_nmpc_solver()
        print(f"Solver NMPC listo  N_H={N_H}  W_POS={W_POS}  W_YAW={W_YAW}  W_DU={W_DU}")

        # ── FUNCION DE RESOLUCION NMPC ────────────────────────────────────────
        def solve_nmpc(state, path_idx, u_warm=None):
            x, y, psi, vy, yr, vx = state

            pinn.freeze()
            fw_tf  = tf.constant(pinn._window_array(), dtype=tf.float32)
            u_last = float(u_warm[0]) if u_warm is not None else 0.0
            d0     = float(np.clip(u_last, -0.3, 0.3))

            A, B, c, lat0, d0_lin = linearize_pinn(fw_tf, vy, yr, vx, d0)
            vref_seq, refs        = get_horizon_refs(path_idx, vx)

            p_val = np.concatenate([
                [x, y, psi, vy, yr, vx],
                [u_last],
                A.flatten(order='C'), B, c, lat0, [d0_lin],
                vref_seq,
                np.array(refs).flatten(),
            ])

            u0  = np.clip(
                u_warm if u_warm is not None else np.full(N_H, u_last),
                -DELTA_MAX, DELTA_MAX)
            sol = _nmpc_solver(
                x0=u0, lbx=[-DELTA_MAX] * N_H, ubx=[DELTA_MAX] * N_H, p=p_val)
            pinn.unfreeze()

            u_opt = np.array(sol['x']).flatten()
            return float(u_opt[0]), u_opt

        # ── INICIALIZAR PINN BUFFER Y ESTADO CALIENTE ─────────────────────────
        pinn.reset(vx0=0.0, steer0=0.0, ay0=0.0, yr0=0.0)
        u_warm         = None
        steer_last_rad = 0.0
        simulation_time = 0.0
        print("PINN buffer inicializado  NMPC listo para correr")

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

            # LECTURA DE VARIABLES DINAMICAS
            velocity     = world.player.get_velocity()
            acceleration = world.player.get_acceleration()
            angular_vel  = world.player.get_angular_velocity()
            tf_          = world.player.get_transform()

            current_yaw  = math.radians(tf_.rotation.yaw)
            roll_angle   = math.radians(tf_.rotation.roll)
            pitch_angle  = math.radians(tf_.rotation.pitch)

            vx_local =  np.cos(current_yaw)*velocity.x + np.sin(current_yaw)*velocity.y
            vy_local = -np.sin(current_yaw)*velocity.x + np.cos(current_yaw)*velocity.y
            ax_local =  np.cos(current_yaw)*acceleration.x + np.sin(current_yaw)*acceleration.y
            ay_local = -np.sin(current_yaw)*acceleration.x + np.cos(current_yaw)*acceleration.y

            yaw_rate   = math.radians(angular_vel.z)
            roll_rate  = math.radians(angular_vel.x)
            pitch_rate = math.radians(angular_vel.y)

            control      = carla.VehicleControl()
            current_x    = tf_.location.x
            current_y    = tf_.location.y

            # WAYPOINT MAS CERCANO (desde CG)
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

            # VELOCIDAD OBJETIVO
            target_speed = float(v_traj[closest_index])

            # PI LONGITUDINAL + FEEDFORWARD
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

            # ── NMPC LATERAL ──────────────────────────────────────────────────
            # 1. Actualizar buffer PINN con mediciones del paso actual
            ay_meas = vx_local * yaw_rate
            pinn._push(steer_last_rad, ay_meas, yaw_rate, vx_local)

            # 2. Resolver NMPC
            state_nmpc = [current_x, current_y, current_yaw,
                          vy_local, yaw_rate, vx_local]
            steer_rad, u_opt = solve_nmpc(state_nmpc, closest_index, u_warm)

            # 3. Warm start para el siguiente paso
            u_warm         = np.append(u_opt[1:], u_opt[-1])
            steer_last_rad = steer_rad

            # Cf / Cr inferidos con el buffer actualizado
            _fw_now   = tf.constant(pinn._window_array(), dtype=tf.float32)
            _st_now   = tf.constant([[vy_local, yaw_rate, vx_local,
                                      steer_last_rad, MASS_VEH]], dtype=tf.float32)
            _out_cfcr = pinn_infer(_fw_now, _st_now).numpy()
            cf_val    = float(_out_cfcr[0, 2])
            cr_val    = float(_out_cfcr[0, 3])

            # 4. Normalizar a [-1, 1] para CARLA
            cmd_steer = float(np.clip(steer_rad / DELTA_MAX, -1.0, 1.0))

            # CTE para log (desde CG, convencion NMPC)
            psi_r = yaw_traj[closest_index]
            crosstrack_error = float(
                (current_y - y_traj[closest_index]) * np.cos(psi_r) -
                (current_x - x_traj[closest_index]) * np.sin(psi_r))

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
                        float(crosstrack_error),
                        cf_val,
                        cr_val,
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
            np.save("data_Town07_pinnnmpc_final.npy", data_out)
            print("Datos guardados en data_Town07_pinnnmpc_final.npy ✅")
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
        description='CARLA NMPC + PINN1-v7 Control Client')
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
