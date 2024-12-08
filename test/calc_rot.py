import numpy as np
import sympy as sp


def calc_state(vel: float, rot: float, pitch: float, yaw: float, delta: float, phase: float):
    vx = vel*np.cos(delta)
    vy = -vel*np.sin(delta)*np.sin(phase)
    vz = vel*np.sin(delta)*np.cos(phase)

    p = rot*np.cos(delta)
    q = rot*np.sin(delta)*np.sin(phase)
    r = rot*np.sin(delta)*np.cos(phase)

    delta_pitch = np.atan(vz/vx)
    delta_yaw = np.atan(vy/vx)

    pitch = pitch + delta_pitch
    yaw = yaw + delta_yaw

    return vx, vy, vz, pitch, yaw, q, r


x, y, z, p, vx, delta, theta = sp.symbols('x y z p vx delta theta')

R = sp.rot_axis2(sp.Symbol("delta"))
R_roll = sp.rot_axis1(sp.Symbol("theta"))

w = sp.Matrix([[p], [0], [0]])

w2 = R*w

w2 = R_roll*w2


v = sp.Matrix([[vx], [0], [0]])

v2 = R_roll*R*v

r = sp.Matrix([[x], [y], [z]])

v_rot = w2.cross(r)

v_tot = v + v_rot


sp.pprint(w2)

sp.pprint(v2)

substitutions = {
    x: 0,
    y: 0,
    z: 0,
    p: 24730,
    vx: 800.0,
    delta: 0.2*np.pi/180.0,
    theta: 45.0*np.pi/180
}


rot_vel = w2.subs(substitutions)
vel = v2.subs(substitutions)


print(rot_vel)
print(vel)

print(sp.python(w2))


vx, vy, vz, pitch, yaw, q, r = calc_state(
    vel=800, rot=24730, pitch=0.0, yaw=0.0, delta=0.2*np.pi/180.0, phase=180.0*np.pi/180)

print("Vx: ", vx)
print("Vy: ", vy)
print("Vz: ", vz)
print("pitch: ", pitch)
print("yaw: ", yaw)
print("q: ", q)
print("r: ", r)
