#!/usr/bin/env python

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import logging
import os
import numpy.random as random
import sys
import math
import multiprocessing as mp

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
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

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error
from agents.navigation.global_route_planner import GlobalRoutePlanner


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
    fig = plt.figure('Stanley — Gráficas en Tiempo Real', figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    # (a) XY — columna izquierda completa
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_xy.set_facecolor(PANEL)
    ax_xy.plot(x_ref, y_ref, color=C_REF, lw=0.8, ls='--', alpha=0.55,
               label='Referencia')
    ax_xy.plot([x_ref[0]], [y_ref[0]], 's', color='#4caf50', ms=8, zorder=6,
               label='Inicio')
    line_xy, = ax_xy.plot([], [], color=C_VEH, lw=1.6, label='Stanley', zorder=4)
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
    root.title("Controlador Stanley — Town07")
    root.resizable(False, False)
    root.configure(bg='#0d1b2a')
    root.attributes('-topmost', True)
    root.protocol("WM_DELETE_WINDOW", lambda: None)   # deshabilitar la X

    W, H = 540, 215
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f'{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}')

    tk.Label(root,
             text="Controlador Stanley  |  Bus Fusorosa  |  Town07",
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
              text="   ▶   Iniciar Simulación con Algoritmo Stanley   ",
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
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world            = None
    traffic_manager  = None
    sim_completed    = [False]
    plot_proc        = None
    plot_q           = None

    # Listas de log inicializadas antes del try para que finally nunca falle
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

        # CAMBIO DE MUNDO Y EXTRACCION DE OBJETOS
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
        # FIN DE MODIFICACION DEL MUNDO

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Stanley Final — Town07')

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # CARGAR DATASET TOWN07
        dataset  = np.load("traj_dataset_Town07.npy", allow_pickle=True).item()
        x_traj   = dataset["x"]
        y_traj   = dataset["y"]
        yaw_traj = dataset["yaw"]
        v_traj   = dataset["v_18_25"]

        route_xy = np.stack((x_traj, y_traj), axis=1)
        print(f"Trayectoria cargada: {len(route_xy)} puntos")
        print(f"Velocidad min: {v_traj.min():.1f} m/s  max: {v_traj.max():.1f} m/s")

        # SPAWN — primer punto de la trayectoria
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

        # PARAMETROS STANLEY
        wheelbase    = 3.01815
        k_e          = 3.00
        dt           = 1.0 / 20.0
        closest_index = 0
        vlon          = 0.0

        # PI LONGITUDINAL + FEEDFORWARD
        Ks_ff  = 33.291680
        K0_ff  = -8.575180
        k_p    = 0.247137
        k_i    = 0.020025
        integral = 0.0

        simulation_time = 0.0
        step_count      = 0

        clock = pygame.time.Clock()

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

            # LECTURA DE VARIABLES DINÁMICAS
            velocity     = world.player.get_velocity()
            acceleration = world.player.get_acceleration()
            angular_vel  = world.player.get_angular_velocity()
            tf           = world.player.get_transform()

            current_yaw = math.radians(tf.rotation.yaw)
            roll_angle  = math.radians(tf.rotation.roll)
            pitch_angle = math.radians(tf.rotation.pitch)

            vx_local =  np.cos(current_yaw)*velocity.x + np.sin(current_yaw)*velocity.y
            vy_local = -np.sin(current_yaw)*velocity.x + np.cos(current_yaw)*velocity.y
            ax_local =  np.cos(current_yaw)*acceleration.x + np.sin(current_yaw)*acceleration.y
            ay_local = -np.sin(current_yaw)*acceleration.x + np.cos(current_yaw)*acceleration.y
            yaw_rate  = math.radians(angular_vel.z)
            roll_rate  = math.radians(angular_vel.x)
            pitch_rate = math.radians(angular_vel.y)
            control    = carla.VehicleControl()

            # POSE ACTUAL
            tf          = world.player.get_transform()
            current_x   = tf.location.x
            current_y   = tf.location.y
            current_yaw = math.radians(tf.rotation.yaw)
            pos_x = current_x + math.cos(current_yaw) * wheelbase
            pos_y = current_y + math.sin(current_yaw) * wheelbase
            current_pos = np.array([pos_x, pos_y])

            # WAYPOINT MAS CERCANO
            search_start  = max(closest_index - 10, 0)
            search_end    = min(closest_index + 300, len(route_xy))
            local_dists   = np.linalg.norm(
                route_xy[search_start:search_end] - current_pos, axis=1)
            closest_index = search_start + np.argmin(local_dists)

            # FIN DE TRAYECTORIA
            if closest_index >= len(route_xy) - 50:
                sim_completed[0] = True
                print("Trayectoria completada ✅")
                break

            # VELOCIDAD OBJETIVO
            target_speed = float(v_traj[closest_index])

            vlon  = vx_local
            error = target_speed - vlon

            u_ff = float(np.clip((target_speed - K0_ff) / Ks_ff, 0.0, 1.0))
            integral += k_i * dt * error
            u_cmd = u_ff + k_p * error + integral
            u_sat = float(np.clip(u_cmd, -1.0, 1.0))
            if u_cmd != u_sat:
                integral -= k_i * dt * error

            if u_sat >= 0.0:
                control.throttle = float(np.clip(u_sat, 0.0, 0.90))
                control.brake    = 0.0
            else:
                control.throttle = 0.0
                control.brake    = float(np.clip(-u_sat, 0.0, 0.5))
            control.hand_brake = False

            # STANLEY — YAW PATH  (look-ahead 3.25 m = 163 puntos × 0.02 m)
            lookahead_idx = min(closest_index + 155, len(yaw_traj) - 1)
            yaw_path_raw = float(yaw_traj[lookahead_idx])
            yaw_path     = np.arctan2(np.sin(yaw_path_raw), np.cos(yaw_path_raw))

            yaw_diff_heading = yaw_path - current_yaw
            if yaw_diff_heading >  np.pi: yaw_diff_heading -= 2*np.pi
            if yaw_diff_heading < -np.pi: yaw_diff_heading += 2*np.pi

            # STANLEY — CROSSTRACK ERROR
            cte_step = 200
            idx1 = max(closest_index - cte_step, 0)
            idx2 = min(closest_index + cte_step, len(route_xy)-1)
            p1wy, p2wy = route_xy[idx1], route_xy[idx2]
            path_vec   = p2wy - p1wy
            veh_vec    = current_pos - p1wy
            path_norm  = np.linalg.norm(path_vec)
            if path_norm > 1e-9:
                cross = path_vec[0]*veh_vec[1] - path_vec[1]*veh_vec[0]
                crosstrack_error = -(cross / path_norm)
            else:
                crosstrack_error = 0.0

            v_eff               = max(vlon, 2.0)
            yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / v_eff)
            steer_expect        = yaw_diff_crosstrack + yaw_diff_heading
            steer_expect        = np.arctan2(np.sin(steer_expect), np.cos(steer_expect))

            cmd_steer = np.clip(steer_expect / 1.22, -1.0, 1.0)

            control.steer            = cmd_steer
            control.manual_gear_shift = False
            world.player.apply_control(control)

            # ── ENVIAR DATOS AL PROCESO DE GRÁFICAS (cada paso = 20 Hz) ─────────
            if plot_q is not None:
                try:
                    plot_q.put_nowait((
                        int(closest_index),
                        float(pos_x),
                        float(pos_y),
                        float(vx_local),
                        float(np.degrees(np.clip(steer_expect, -1.22, 1.22))),
                        float(crosstrack_error)
                    ))
                except Exception:
                    pass   # queue llena: descartar este punto, no bloquear

            # LOG
            log_x.append(pos_x)
            log_y.append(pos_y)
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
            log_steer_rad.append(steer_expect)
            simulation_time += dt
            step_count      += 1
            # FIN DATA RECORD

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
            np.save("data_town07_stanley_final.npy", data_out)
            print("Datos guardados en data_town07_stanley_final.npy")
        else:
            print("Sin datos que guardar.")

        # ── Cerrar proceso de gráficas ────────────────────────────────────────
        if plot_q is not None and plot_proc is not None:
            try:
                plot_q.put_nowait(None)   # señal de fin al worker
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
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',
        help='Window resolution (default: 640x480)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    mp.freeze_support()   # necesario en Windows para ejecutables empaquetados
    main()
