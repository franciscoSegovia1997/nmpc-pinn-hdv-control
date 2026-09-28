# -*- coding: utf-8 -*-
"""
pinn1_v7_utils.py
=================
Utilidades para cargar y usar PINN1-v7 fuera del notebook de entrenamiento.

Contiene:
  - Constantes fisicas y de normalizacion
  - Clases Keras custom (necesarias para cargar el .keras)
  - Funcion normalizar()
  - Funcion load_pinn1_v7()
  - Clase PINN1v7Predictor  — wrapper con buffer rolling para uso en tiempo real / NMPC
  - Funcion predict_batch() — inferencia vectorizada para grid search offline

Uso basico:
    from pinn1_v7_utils import PINN1v7Predictor
    pinn = PINN1v7Predictor("models/PINN1_v7/PINN1_v7_best.keras")
    pinn.reset(vx0=16.0)                          # inicializar buffer
    vy_next, yr_next = pinn.step(steer, ay, yr, vx, vy, mass=4800.0)
"""

import json
import numpy as np
from collections import deque
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

# ==============================================================================
# Constantes fisicas del vehiculo (Mitsubishi Fusorosa)
# ==============================================================================
IZ   = 21_000.0   # inercia de guinada [kg·m²]
LF   = 3.01815    # distancia CG — eje delantero [m]
LR   = 2.61195    # distancia CG — eje trasero   [m]
L_WB = LF + LR    # batalla total [m]
DT   = 0.05       # paso de tiempo CARLA [s]
V_MIN = 5.0       # velocidad minima valida para el modelo [m/s]

MASS_PATTERN = [4200., 4200., 4800., 4800., 5400., 5400., 6000., 6000.]

# ==============================================================================
# Configuracion del modelo (Set D, L=40)
# ==============================================================================
FEATURE_NAMES = ['steer', 'ay', 'yr', 'vx']   # Set D — 4 features
L_WINDOW      = 40                              # pasos de lookback (2 s a 20 Hz)
INPUT_DIM     = len(FEATURE_NAMES) * L_WINDOW  # 160

ENC_UNITS    = 96
N_ENC_LAYERS = 3

CF_BOUNDS = (0., 500_000.)
CR_BOUNDS = (0., 500_000.)

# Limites de normalizacion [-1, 1] para cada feature
FEATURE_BOUNDS = {
    'steer': (-1.22,  1.22),
    'ay':    (-20.0,  20.0),
    'yr':    (-2.0,    2.0),
    'vx':    ( 0.0,   30.0),
    'vy':    (-2.5,    2.5),
}


# ==============================================================================
# Normalizacion
# ==============================================================================
def normalizar(arr, feature: str) -> np.ndarray:
    """Mapea arr a [-1, 1] usando FEATURE_BOUNDS[feature]."""
    lo, hi = FEATURE_BOUNDS[feature]
    return (np.asarray(arr, dtype=np.float32) - (lo + hi) / 2.0) / ((hi - lo) / 2.0)


def desnormalizar(arr, feature: str) -> np.ndarray:
    """Inverso de normalizar."""
    lo, hi = FEATURE_BOUNDS[feature]
    return np.asarray(arr, dtype=np.float32) * ((hi - lo) / 2.0) + (lo + hi) / 2.0


# ==============================================================================
# Capas Keras custom — DEBEN estar definidas para cargar el .keras
# ==============================================================================
class PhysicsGuard(keras.Layer):
    """Restringe [Cf, Cr] al rango fisico valido usando sigmoid."""

    def __init__(self, cf_bounds, cr_bounds, **kwargs):
        kwargs.setdefault('name', 'cf_cr_guard')
        super().__init__(**kwargs)
        self.cf_min = float(cf_bounds[0]); self.cf_max = float(cf_bounds[1])
        self.cr_min = float(cr_bounds[0]); self.cr_max = float(cr_bounds[1])

    def call(self, z):
        Cf = self.cf_min + (self.cf_max - self.cf_min) * tf.sigmoid(z[:, 0])
        Cr = self.cr_min + (self.cr_max - self.cr_min) * tf.sigmoid(z[:, 1])
        return tf.stack([Cf, Cr], axis=1)

    def get_config(self):
        c = super().get_config()
        c.update({'cf_bounds': (self.cf_min, self.cf_max),
                  'cr_bounds': (self.cr_min, self.cr_max)})
        return c


class BicycleModelLayer(keras.Layer):
    """Dinamica de bicicleta linealizada — integra vy y yr un paso."""

    def __init__(self, lf, lr, Iz, dt, **kwargs):
        kwargs.setdefault('name', 'bicycle_model')
        super().__init__(**kwargs)
        self.lf = float(lf); self.lr = float(lr)
        self.Iz = float(Iz); self.dt = float(dt)

    def call(self, inputs):
        cf_cr, state_mass = inputs
        Cf = cf_cr[:, 0]; Cr = cf_cr[:, 1]
        vy_t    = state_mass[:, 0]
        yr_t    = state_mass[:, 1]
        vx_t    = tf.maximum(state_mass[:, 2], tf.constant(1e-3, dtype=tf.float32))
        steer_t = state_mass[:, 3]
        mass_t  = state_mass[:, 4]

        vy_dot = (-(Cf + Cr) / (mass_t * vx_t) * vy_t
                  - (self.lf * Cf - self.lr * Cr) / (mass_t * vx_t) * yr_t
                  - vx_t * yr_t
                  + Cf / mass_t * steer_t)

        yr_dot = (-(self.lf * Cf - self.lr * Cr) / (self.Iz * vx_t) * vy_t
                  - (self.lf**2 * Cf + self.lr**2 * Cr) / (self.Iz * vx_t) * yr_t
                  + self.lf * Cf / self.Iz * steer_t)

        return tf.stack([vy_t + self.dt * vy_dot,
                         yr_t + self.dt * yr_dot], axis=1)

    def get_config(self):
        c = super().get_config()
        c.update({'lf': self.lf, 'lr': self.lr, 'Iz': self.Iz, 'dt': self.dt})
        return c


class CombinedLoss(keras.losses.Loss):
    """Loss de entrenamiento Stage 2 — necesaria para cargar el .keras."""

    def __init__(self, var_vy, var_yr, lambda_param, cf_nom, cr_nom, cf_thresh, **kwargs):
        super().__init__(**kwargs)
        self._var_vy = float(var_vy); self._var_yr = float(var_yr)
        self._lambda = float(lambda_param)
        self._cf_nom = float(cf_nom);  self._cr_nom = float(cr_nom)
        self._thresh = float(cf_thresh)

    def call(self, y_true, y_pred):
        vy_p = y_pred[:, 0]; yr_p = y_pred[:, 1]
        Cf_p = y_pred[:, 2]; Cr_p = y_pred[:, 3]
        vy_t = y_true[:, 0]; yr_t = y_true[:, 1]
        Cf_t = y_true[:, 2]; Cr_t = y_true[:, 3]

        L_st = (tf.reduce_mean(tf.square(vy_p - vy_t)) / self._var_vy +
                tf.reduce_mean(tf.square(yr_p - yr_t)) / self._var_yr)

        valid = tf.cast(Cf_t > self._thresh, tf.float32)
        n_v   = tf.reduce_sum(valid) + 1e-6
        L_par = tf.reduce_sum(valid * (tf.square((Cf_p - Cf_t) / self._cf_nom) +
                                       tf.square((Cr_p - Cr_t) / self._cr_nom))) / n_v
        return L_st + self._lambda * L_par

    def get_config(self):
        c = super().get_config()
        c.update({'var_vy': self._var_vy, 'var_yr': self._var_yr,
                  'lambda_param': self._lambda, 'cf_nom': self._cf_nom,
                  'cr_nom': self._cr_nom, 'cf_thresh': self._thresh})
        return c


# Diccionario de objetos custom para keras.models.load_model
CUSTOM_OBJECTS = {
    'PhysicsGuard':       PhysicsGuard,
    'BicycleModelLayer':  BicycleModelLayer,
    'CombinedLoss':       CombinedLoss,
}


# ==============================================================================
# Carga del modelo
# ==============================================================================
def load_pinn1_v7(model_path: str = None) -> keras.Model:
    """
    Carga PINN1_v7_best.keras con los custom_objects necesarios.

    Parameters
    ----------
    model_path : str | None
        Ruta al .keras. Si es None usa la ruta por defecto
        relativa a este archivo.

    Returns
    -------
    keras.Model  con inputs [window(160,), state_mass(5,)]
                 y output  [vy_next, yr_next, Cf, Cr]
    """
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "PINN1_v7" / "PINN1_v7_best.keras"
    model = keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS)
    return model


def load_config(config_path: str = None) -> dict:
    """Carga PINN1_v7_config.json."""
    if config_path is None:
        config_path = Path(__file__).parent / "models" / "PINN1_v7" / "PINN1_v7_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


# ==============================================================================
# Wrapper con buffer rolling — uso en tiempo real y en loop NMPC
# ==============================================================================
class PINN1v7Predictor:
    """
    Wrapper sobre PINN1-v7 que mantiene el buffer de L=40 pasos
    y expone una API simple paso a paso.

    Uso tipico (loop de control 20 Hz):
        pinn = PINN1v7Predictor()
        pinn.reset(vx0=16.0)

        # cada paso:
        vy_next, yr_next = pinn.step(steer_rad, ay, yr, vx, vy, mass=4800.0)
        # propagar estado cinematico con vy_next, yr_next...
        pinn.commit()   # confirma el paso (avanza el buffer)

    Uso en NMPC (horizonte N con ventana congelada):
        pinn.freeze()   # guarda snapshot del buffer actual
        for i in range(N):
            vy_i1, yr_i1 = pinn.predict_frozen(steer_i, vy_i, yr_i, vx_i, mass)
        pinn.unfreeze() # restaura buffer real
    """

    def __init__(self, model_path: str = None, mass_default: float = 4800.0):
        self.model        = load_pinn1_v7(model_path)
        self.mass_default = float(mass_default)
        self._buffer      = deque(maxlen=L_WINDOW)   # buffer normalizado [L, 4]
        self._frozen_buf  = None                      # snapshot para horizon freeze
        self._ready       = False

    # ------------------------------------------------------------------
    def reset(self, vx0: float = 16.0, steer0: float = 0.0,
              ay0: float = 0.0, yr0: float = 0.0):
        """Inicializa el buffer con L copias del estado inicial."""
        row = [normalizar(steer0, 'steer'),
               normalizar(ay0,    'ay'),
               normalizar(yr0,    'yr'),
               normalizar(vx0,    'vx')]
        self._buffer.clear()
        for _ in range(L_WINDOW):
            self._buffer.append(row[:])
        self._ready = True

    # ------------------------------------------------------------------
    def _push(self, steer_rad: float, ay: float, yr: float, vx: float):
        """Agrega un nuevo paso al buffer (normalizado)."""
        self._buffer.append([
            float(normalizar(steer_rad, 'steer')),
            float(normalizar(ay,        'ay')),
            float(normalizar(yr,        'yr')),
            float(normalizar(vx,        'vx')),
        ])

    def _window_array(self) -> np.ndarray:
        """Devuelve el buffer aplanado como (1, 160)."""
        return np.array(self._buffer, dtype=np.float32).flatten().reshape(1, -1)

    # ------------------------------------------------------------------
    def step(self, steer_rad: float, ay: float, yr: float, vx: float,
             vy: float, mass: float = None) -> tuple:
        """
        Agrega el estado actual al buffer y devuelve [vy_{t+1}, yr_{t+1}].

        Parameters  (todos en unidades fisicas, SIN normalizar)
        ----------
        steer_rad : steering en radianes (de CARLA o NMPC output)
        ay        : aceleracion lateral [m/s²]
        yr        : yaw rate actual [rad/s]
        vx        : velocidad longitudinal [m/s]
        vy        : velocidad lateral actual [m/s]
        mass      : masa del vehiculo [kg] — default 4800

        Returns
        -------
        (vy_next [m/s], yr_next [rad/s])
        """
        if not self._ready:
            raise RuntimeError("Llama reset() antes de step().")
        if mass is None:
            mass = self.mass_default

        self._push(steer_rad, ay, yr, vx)

        X_win   = self._window_array()
        X_state = np.array([[vy, yr, vx, steer_rad, mass]], dtype=np.float32)

        pred = self.model.predict([X_win, X_state], verbose=0)
        return float(pred[0, 0]), float(pred[0, 1])   # vy_next, yr_next

    # ------------------------------------------------------------------
    def freeze(self):
        """Guarda snapshot del buffer para exploración del horizonte NMPC."""
        self._frozen_buf = list(self._buffer)

    def unfreeze(self):
        """Restaura el buffer real tras exploración del horizonte."""
        if self._frozen_buf is not None:
            self._buffer.clear()
            self._buffer.extend(self._frozen_buf)
            self._frozen_buf = None

    def predict_frozen(self, steer_rad: float, vy: float, yr: float,
                       vx: float, mass: float = None) -> tuple:
        """
        Prediccion dentro del horizonte NMPC usando ventana congelada.
        NO modifica el buffer real — usa el snapshot de freeze().
        """
        if mass is None:
            mass = self.mass_default

        X_win   = self._window_array()
        X_state = np.array([[vy, yr, vx, steer_rad, mass]], dtype=np.float32)
        pred    = self.model.predict([X_win, X_state], verbose=0)
        return float(pred[0, 0]), float(pred[0, 1])


# ==============================================================================
# Inferencia batch para grid search offline
# ==============================================================================
def predict_batch(model: keras.Model,
                  windows: np.ndarray,
                  states:  np.ndarray) -> np.ndarray:
    """
    Inferencia vectorizada sobre multiples muestras.

    Parameters
    ----------
    model   : modelo cargado con load_pinn1_v7()
    windows : (N, 160) float32 — ventanas normalizadas
    states  : (N, 5)   float32 — [vy, yr, vx, steer, mass]

    Returns
    -------
    (N, 4) float32 — [vy_next, yr_next, Cf, Cr]
    """
    return model.predict([windows, states], verbose=0, batch_size=512)


def build_window_matrix(history: np.ndarray) -> np.ndarray:
    """
    Construye la matriz de ventanas desde un array de historial.

    Parameters
    ----------
    history : (T, 4) float32 — columnas = [steer_n, ay_n, yr_n, vx_n]
              ya normalizadas con normalizar()

    Returns
    -------
    (T - L_WINDOW, L_WINDOW * 4) float32
    """
    T = len(history)
    if T <= L_WINDOW:
        raise ValueError(f"history debe tener al menos {L_WINDOW + 1} pasos.")
    windows = np.array(
        [history[t - L_WINDOW:t].flatten() for t in range(L_WINDOW, T)],
        dtype=np.float32
    )
    return windows


# ==============================================================================
# Test rapido de carga
# ==============================================================================
if __name__ == '__main__':
    print("Cargando PINN1-v7...")
    m = load_pinn1_v7()
    print(f"  Modelo cargado OK  |  params: {m.count_params():,}")

    cfg = load_config()
    print(f"  Config: Set={cfg['best_set']}  L={cfg['best_L']}  "
          f"TEST vy R2={cfg['test_metrics']['r2_vy']:.4f}  "
          f"yr R2={cfg['test_metrics']['r2_yr']:.4f}")

    p = PINN1v7Predictor()
    p.reset(vx0=16.0)
    vy1, yr1 = p.step(steer_rad=0.02, ay=0.1, yr=0.05, vx=16.0, vy=0.0)
    print(f"  Test step: vy_next={vy1:.4f} m/s  yr_next={yr1:.4f} rad/s")
    print("pinn1_v7_utils OK")
