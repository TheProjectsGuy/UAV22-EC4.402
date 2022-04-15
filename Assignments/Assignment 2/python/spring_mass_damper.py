# A simple spring mass damper system
"""
    Given a mass, with a spring attached to it, and a viscosity 
    (friction) with the floor, derive a system response for moving the
    mass from one point to another.
"""

# %% Import everything
import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

# %%
t, s = sp.symbols(r"t, s")
# System parameters
m = sp.symbols(r"m", real=True, positive=True)
# Force parameters
kp = sp.symbols(r"k_p", real=True, positive=True)
xd = sp.symbols(r"x_d", real=True)


# %%
def L(f):
    return sp.laplace_transform(f, t, s, noconds=True)

def invL(F):
    return sp.inverse_laplace_transform(F, s, t)

# %%
X_s = (kp*xd)/((m*s**2 + kp)*s)

# %%
x_t = invL(X_s.apart(s))

# %%
tvals = np.linspace(0, 100, 1000)
kp_val = 2.0
m_val = 10
xd_val = 20
xt = lambda t_val: float(x_t.subs({t: t_val, m: m_val, xd: xd_val, 
    kp: kp_val}))
x_vals = np.array([xt(tv) for tv in tvals])
plt.figure(dpi=150)
plt.plot(tvals, x_vals, '.')

# %% Try acceleration
accX_s = (s**2)*X_s
accX_t = invL(accX_s)
ddxt = lambda t_val: float(accX_t.subs({t: t_val, m: m_val, 
    xd: xd_val, kp: kp_val}))
ddx_vals = np.array([ddxt(tv) for tv in tvals])
plt.figure(dpi=150)
plt.plot(tvals, ddx_vals, '.')

# %% Try damping
kd = sp.symbols(r"k_d", real=True, positive=True)
x_s = (kp * xd)/(s*(m*s**2 + kd*s + kp))
x_t = invL(x_s.apart(s))

# %%
kd_val = 5
kp_val = 1
xt = lambda t_val: float(x_t.subs({t: t_val, m: m_val, xd: xd_val, 
    kp: kp_val, kd:kd_val}))
x_vals = np.array([xt(tv) for tv in tvals])
plt.figure(dpi=150)
plt.plot(tvals, x_vals, '.')

# %% Closed loop system
C_s = kp + s*kd
G_s = 1/(m*s**2)
Xd_s = xd/s     # Step input (target xd)
TF_s = (C_s*G_s)/(1+C_s*G_s)
X_s = TF_s * Xd_s

# %%
x_t = invL(X_s.apart(s))

# %%
kd_val = 1.5
kp_val = 0.1
xt = lambda t_val: float(x_t.subs({t: t_val, m: m_val, xd: xd_val, 
    kp: kp_val, kd:kd_val}))
x_vals = np.array([xt(tv) for tv in tvals])
plt.figure(dpi=150)
plt.plot(tvals, x_vals, '.')

# %%
