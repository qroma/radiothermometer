"""
Внесок шару тканини у радіометричний сигнал залежно від глибини та частоти.

Модель: плоска хвиля, однорідне півпростір, нормальне падіння, без відбиттів.
  1. Комплексна діелектрична проникність:
       eps*(w) = eps_inf + sum_n d_eps_n / (1 + (j*w*tau_n)^(1-alpha_n)) + sigma_i/(j*w*eps0)
  2. Стала поширення:  gamma = j*(w/c)*sqrt(eps*) = alpha + j*beta
       alpha = -Im{(w/c)*sqrt(eps*)}   [Нп/м]  - згасання амплітуди поля
  3. Потужність спадає як exp(-2*alpha*z); вагова функція сигналу  W(z) = 2*alpha*exp(-2*alpha*z)
  4. Малюнок показує відносний внесок шару:  exp(-2*alpha*z)  (поверхня = 1)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS0 = 8.854187817e-12   # Ф/м
C = 299792458.0          # м/с

# (eps_inf, [(d_eps, tau, alpha), ...], sigma_i)
TISSUES = {
    "м'язової": (4.0,
                 [(50, 7.234e-12, 0.1), (7000, 353.678e-9, 0.1),
                  (1.2e6, 318.31e-6, 0.1), (2.5e7, 2.274e-3, 0.0)], 0.2),
    "жирової (не інфільтрованої)": (2.5,
                 [(3, 7.958e-12, 0.2), (15, 15.915e-9, 0.1),
                  (3.3e4, 159.155e-6, 0.05), (1e7, 15.915e-3, 0.01)], 0.01),
}

def eps_complex(f, tissue):
    eps_inf, terms, sigma_i = tissue
    w = 2 * np.pi * f
    eps = eps_inf + 0j
    for d_eps, tau, al in terms:
        eps += d_eps / (1 + (1j * w * tau) ** (1 - al))
    return eps + sigma_i / (1j * w * EPS0)      # eps = eps' - j*eps''

def alpha_np_per_m(f, tissue):
    k = 2 * np.pi * f / C * np.sqrt(eps_complex(f, tissue))
    return -k.imag                               # Нп/м

def plot(tissue_name, tissue, fname, freqs=(0.3, 0.5, 1.0, 1.3, 2.0, 3.0, 5.0),
         zmax_cm=7.0):
    z = np.linspace(0, zmax_cm, 400)             # см
    cols = plt.cm.viridis(np.linspace(0.0, 0.92, len(freqs)))
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=170)
    for f, c in zip(freqs, cols):
        a = alpha_np_per_m(f * 1e9, tissue)
        hl = (f == 1.3)
        ax.plot(z, np.exp(-2 * a * z / 100), color="black" if hl else c,
                lw=3.2 if hl else 1.8,
                label=f"{f:g} ГГц" + (" (робоча)" if hl else "") + f"  (α≈{a:.1f} Нп/м)")
    ax.axhline(0.1, color="gray", ls="--", lw=1, label="рівень 10% від поверхневого внеску")
    ax.set_xlim(0, zmax_cm); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Глибина, см")
    ax.set_ylabel("Відносний внесок шару у сигнал, e^(−2αz)\n(поверхня = 1)")
    ax.set_title(f"Внесок шару {tissue_name} тканини у радіометричний сигнал\n"
                 "залежно від глибини для різних частот")
    ax.grid(True, ls=":", lw=.6)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout(); fig.savefig(fname)

if __name__ == "__main__":
    for name, t in TISSUES.items():
        a = alpha_np_per_m(1.3e9, t)
        print(f"{name}: 1.3 ГГц  alpha={a:.1f} Нп/м = {8.686*a/100:.2f} дБ/см, "
              f"1/(2alpha)={100/(2*a):.1f} см, глибина 90% = {100*np.log(10)/(2*a):.1f} см")
    plot("м'язової", TISSUES["м'язової"], "signal_vs_depth_muscle.png")
    plot("жирової", TISSUES["жирової (не інфільтрованої)"], "signal_vs_depth_fat.png")
