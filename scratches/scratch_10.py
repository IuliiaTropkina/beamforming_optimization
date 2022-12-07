import numpy as np

from lib.numba_opt import jit_hardcore
from lib.vectors import vector, origin, norm

speed_of_light = 3e8
f = speed_of_light

TX_POSITIONS = np.array([[0.0,0,0], [100,0,0],[0, 20,0],[80, 20,0]],dtype=float)
pos_tgt = vector(70.0,0.0,0.0)
pos_rx = origin
v_tx = vector(0.0,.0,.0)
v_tgt = vector(-50.0, 0.0, 0)
v_rx = vector(0,0,0)

@jit_hardcore
def compute_deriv(pos_tx, pos_rx, pos_tgt, v_tx, v_rx, v_tgt, delta_t = 1e-5):
    old_dist = norm(pos_tx - pos_tgt) + norm(pos_rx - pos_tgt)
    pos_tgt_p = pos_tgt + v_tgt * delta_t
    pos_tx_p = pos_tx + v_tx*delta_t
    pos_rx_p = pos_rx + v_rx * delta_t
    new_dist = norm(pos_tx_p - pos_tgt_p) + norm(pos_rx_p - pos_tgt_p)
    return  (new_dist - old_dist) / delta_t

for pos_tx in TX_POSITIONS:
    deriv = compute_deriv(pos_tx, pos_rx, pos_tgt, v_tx, v_rx, v_tgt)

    doppler2 = f/speed_of_light * deriv

    print(pos_tx, doppler2)