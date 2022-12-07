import numpy as np
from lib.vectors import vector_normalize, norm
from scipy.spatial.transform import Rotation as R
import math
BITS = 3
def quantize_direction(dv:np.ndarray, bits:int=BITS)-> np.ndarray:
    ibase = 2**bits
    return np.round(dv * ibase)


def vector_rotate(vector, rot_angle, rotation_axis=np.array([0, 0, 1])):
    """Rotate relative to oy;
    vector - which rotate;
    rot_angle - rotation angle in radians"""
    lv = norm(vector)
    vector = vector / lv
    rotation_vector = rot_angle * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    return np.array(rotation.apply(vector) * lv)
COUNT = 0
c = 0
num_angle = 100
max_angle = 20
min_angle = 0
q_v_prev = None
for a in np.linspace(min_angle,max_angle,num_angle):
    a = a*math.pi/180
    v = np.array([1, 0, 0])
    v_rot = vector_rotate(v,a, rotation_axis = np.array([0,0,1]))
    q_v = quantize_direction(v_rot)
    print(q_v)
    if q_v_prev is not None:
        if not (q_v == q_v_prev).all():
            COUNT +=1
    if COUNT == 1:
        c+=1
    print(f"a: {a}, v: {q_v} ")
    q_v_prev = q_v.copy()

print(c*(max_angle-min_angle)/num_angle)