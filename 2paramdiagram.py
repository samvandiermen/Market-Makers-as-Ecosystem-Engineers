"""
Market makers as ecosystem engineers: two-parameter diagram in the (c, R)
plane (Figure 2 of the report).

Usage
    python 2paramdiagram.py    save figures/2paramdiagram.pdf

Positive equilibria satisfy R = Phi(M), with
    Phi(M) = (1+M) (1 + b M (1 + c M)) / (1 + c M).
For c > 1 + b, Phi dips below 1 before rising. Its minimum R_f(c) is where
the liquid market and the threshold appear or disappear together.

Regions:
    R > 1              liquid market only
    R_f(c) < R < 1     empty and liquid market both stable
    R < R_f(c)         empty market only
eps changes neither the position nor the stability of the equilibria
(for 0 < eps <= 1), so it does not appear in this diagram.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# LaTeX-like fonts, so the figure matches the report
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})

# ---------- Settings ----------
FIG_DIR = Path(__file__).parent / "figures"
b = 0.1
C_MAX, R_LO, R_HI = 8.0, 0.5, 1.5                 # plot range
GREEN, ORANGE, BLUE = "#0B7A4B", "#C2700F", "#3B6FD4"
INK, MUTED = "#1A1A1A", "#5B5B5B"                 # lines and axes

# ---------- Model ----------
def Phi(M, c):
    """Value of R for which M is a positive equilibrium."""
    return (1 + M) * (1 + b * M * (1 + c * M)) / (1 + c * M)

def R_fold(c):
    """Saddle-node value R_f(c): the minimum of Phi over M > 0 (1 if c <= 1+b)."""
    if c <= 1 + b:
        return 1.0
    # coarse log grid to find the minimum, then a fine grid around it
    M = np.logspace(-6, 3, 20000)
    i = Phi(M, c).argmin()
    lo, hi = M[max(i - 1, 0)], M[min(i + 1, len(M) - 1)]
    return Phi(np.linspace(lo, hi, 20000), c).min()

c = np.linspace(0, C_MAX, 800)
Rf = np.array([R_fold(x) for x in c])

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
fig.patch.set_facecolor("white")

# regions
ax.fill_between(c, 1.0, R_HI, color=GREEN, alpha=0.13, lw=0)
ax.fill_between(c, Rf, 1.0, color=ORANGE, alpha=0.18, lw=0)
ax.fill_between(c, R_LO, Rf, color=BLUE, alpha=0.10, lw=0)

# boundaries; R_f(c) only exists for c > 1+b
ax.axhline(1.0, color=INK, lw=1.6, zorder=3,
           label=r"empty market changes stability ($R=1$)")
ax.plot(c[c > 1 + b], Rf[c > 1 + b], color=INK, lw=1.6, ls="--", zorder=3,
        label=r"liquid market appears or disappears ($R=R_{\mathrm{f}}(c)$)")

# point where the two boundaries meet
ax.plot(1 + b, 1.0, "o", ms=7, mfc="white", mec=INK, mew=1.6, zorder=4)
ax.annotate(r"$(c,R)=(1+b,\,1)$", xy=(1 + b, 1.0), xytext=(1.9, 1.11),
            fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

# region labels
ax.text(5.6, 1.22, "liquid market only", ha="center", va="center",
        fontsize=9.5, color=GREEN, linespacing=1.4)
ax.text(5.0, 0.83, "empty or liquid market\n(depends on starting size)",
        ha="center", va="center", fontsize=9.5, color=ORANGE, linespacing=1.4)
ax.text(1.4, 0.68, "empty market only", ha="center", va="center",
        fontsize=9.5, color=BLUE)

# axes
ax.set(xlim=(0, C_MAX), ylim=(R_LO, R_HI),
       xlabel=r"liquidity response $c$", ylabel=r"survival number $R$")
ax.tick_params(colors=MUTED, labelsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
    ax.spines[side].set_linewidth(0.8)
ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95,
          edgecolor="#D8D8D8")

# ---------- Save and print reference values ----------
FIG_DIR.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIG_DIR / "2paramdiagram.pdf", bbox_inches="tight", facecolor="white")
print("Saved to", FIG_DIR / "2paramdiagram.pdf")

print("b =", b, " lines meet at c = 1+b =", 1 + b)
for x in (1.2, 1.5, 2.0, 2.4, 3.0, 5.0):
    print(f"  c = {x:7.4f}   R_fold = {R_fold(x):.4f}")

plt.show()