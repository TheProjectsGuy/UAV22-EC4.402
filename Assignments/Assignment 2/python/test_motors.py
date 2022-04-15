
# %% Import everything
import sympy as sp

# %%
L, kt, ktau = sp.symbols(r"L, k_T, k_{\tau}", real=True, 
    positive=True)
M = sp.Matrix([
    [kt, kt, kt, kt],
    [0.5*L*ktau, -0.5*L*ktau, -0.5*L*ktau, 0.5*L*ktau],
    [0.5*L*ktau, 0.5*L*ktau, -0.5*L*ktau, -0.5*L*ktau],
    [ktau, -ktau, ktau, -ktau]
])
M_inv = M.inv()

# %%
