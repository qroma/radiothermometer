"""
Моделювання девіації Аллана (ADEV) вихідного сигналу модуляційного радіотермометра
до та після програмної компенсації асиметрії каналів комутатора.

Додано режим «плаваючого» коефіцієнта: κ̂(t) лінійно інтерполюється між калібруваннями
   (згладжування із затримкою виходу ≤ T_кал) на додаток до sample-and-hold.

"""
from pathlib import Path
import numpy as np
import allantools
from scipy.signal import lfilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

# ---------------- параметри (як у статті) ----------------
B = 300e6                 # смуга, Гц
p = 0.5                   # розподіл часу на вимірювання (оптимум)
T_ANT, T_ET = 310.0, 300.0
NF_DB = 1.5
T0 = 290.0
T_SYS = T_ANT + T0 * (10 ** (NF_DB / 10) - 1)   # ≈ 430 К
TAU0 = 5e-3               # один цикл антена+еталон, с
N = 2_000_000             # відліків -> сеанс 10 000 с
T_SESSION = N * TAU0
A1_0, A2_0 = 10 ** (-0.9 / 10), 10 ** (-1.1 / 10)
TAU_C = (90.0, 130.0)     # час кореляції дрейфу каналів, с
SIGMA_REL = 0.02          # відносне ст. відхилення дрейфу кожного каналу
T_CAL = 30.0              # інтервал калібрування, с
TAU_CAL_MEAS = 1.0        # тривалість одного калібрувального вимірювання, с
N_REAL = 20               # кількість реалізацій Монте-Карло


def ou(rng, n, dt, tau_c, sigma):
    phi = np.exp(-dt / tau_c)
    return lfilter([1.0], [1.0, -phi], rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n))


def one_run(seed, taus_req):
    rng = np.random.default_rng(seed)
    a1 = A1_0 * (1 + ou(rng, N, TAU0, TAU_C[0], SIGMA_REL))
    a2 = A2_0 * (1 + ou(rng, N, TAU0, TAU_C[1], SIGMA_REL))

    dT_cycle = T_SYS / np.sqrt(p * (1 - p) * B * TAU0)      # формула (5): вся різниця за цикл
    sigma_phase = dT_cycle / np.sqrt(2)                      # на кожну фазу
    V_ant = a1 / A2_0 * T_ANT + rng.normal(0, sigma_phase, N)   # у шкалі кельвінів
    V_et = a2 / A2_0 * T_ET + rng.normal(0, sigma_phase, N)
    raw = V_ant - V_et

    step = int(round(T_CAL / TAU0))
    idx = np.arange(0, N, step)
    s_cal = T_SYS / np.sqrt(B * TAU_CAL_MEAS)
    e1 = a1[idx] * T_ET / A2_0 + rng.normal(0, s_cal, idx.size)
    e2 = a2[idx] * T_ET / A2_0 + rng.normal(0, s_cal, idx.size)
    r = e1 / e2                                              # оцінки κ у моменти калібрування
    kappa_hold = np.repeat(r, step)[:N]                      # sample-and-hold
    kappa_interp = np.interp(np.arange(N), idx, r)           # «плаваючий» κ̂(t)
    corr_hold = V_ant / kappa_hold - V_et
    corr_interp = V_ant / kappa_interp - V_et

    out = []
    for x in (raw, corr_hold, corr_interp):
        t_, a_, _, _ = allantools.oadev(x, rate=1 / TAU0, data_type="freq", taus=taus_req)
        out.append(a_)
    return t_, out[0], out[1], out[2]


if __name__ == "__main__":
    taus_req = np.logspace(np.log10(TAU0), np.log10(T_SESSION / 10), 40)
    runs = [one_run(100 + s, taus_req) for s in range(N_REAL)]
    tau = runs[0][0]
    def stat(k):
        a = np.array([r[k] for r in runs])
        return np.median(a, 0), *np.percentile(a, [16, 84], axis=0)
    (m_r, lo_r, hi_r), (m_h, lo_h, hi_h), (m_i, lo_i, hi_i) = stat(1), stat(2), stat(3)

    lines = [f"T_sys = {T_SYS:.1f} K, сеанс = {T_SESSION:.0f} с, реалізацій = {N_REAL}, T_кал = {T_CAL:.0f} с"]
    for q in (1, 10, 30, 100, 300, 1000):
        i = int(np.argmin(abs(tau - q)))
        lines.append(f"tau={tau[i]:7.1f} с: без комп. {m_r[i]:.3f} К | hold {m_h[i]:.3f} К (×{m_r[i]/m_h[i]:.2f}) "
                     f"| інтерполяція {m_i[i]:.3f} К (×{m_r[i]/m_i[i]:.2f})")
    for name, m in (("без компенсації", m_r), ("hold", m_h), ("інтерполяція", m_i)):
        j = int(np.argmin(m)); lines.append(f"мінімум, {name}: {m[j]:.3f} К при tau={tau[j]:.3f} с")
    print("\n".join(lines)); (OUT / "adev_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    tw = np.array([TAU0, T_SESSION / 10])
    def base(fig_title):
        fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=170)
        ax.loglog(tw, 2 * T_SYS / np.sqrt(B * tw), "--", color="gray", lw=1.1,
                  label=r"білий шум: $2T_{sys}/\sqrt{B\tau}$ (теорія)")
        ax.axvline(T_CAL, color="gray", ls=":", lw=1)
        ax.set_xlabel(r"Час усереднення $\tau$, с"); ax.set_ylabel(r"Девіація Аллана $\sigma(\tau)$, К")
        ax.set_title(fig_title); ax.grid(True, which="both", ls=":", lw=0.5)
        return fig, ax
    sub = f"(медіана та інтервал 16–84% за {N_REAL} реалізаціями, T_кал = {T_CAL:.0f} с)"

    # рисунок 1: без компенсації vs hold
    fig, ax = base("Девіація Аллана до та після компенсації асиметрії комутатора\n" + sub)
    ax.fill_between(tau, lo_r, hi_r, color="#c0392b", alpha=0.18, lw=0)
    ax.fill_between(tau, lo_h, hi_h, color="#1f6f43", alpha=0.18, lw=0)
    ax.loglog(tau, m_r, "o-", ms=3, lw=1.4, color="#c0392b", label="Без компенсації")
    ax.loglog(tau, m_h, "s-", ms=3, lw=1.4, color="#1f6f43", label="З компенсацією (κ̂ стале між калібруваннями)")
    ax.legend(loc="lower left", fontsize=8.3, framealpha=0.95); fig.tight_layout()
    fig.savefig(OUT / "adev_comparison_corrected.png")

    # рисунок 2: + плаваючий коефіцієнт
    fig, ax = base("Девіація Аллана: сталий та «плаваючий» коефіцієнт κ̂\n" + sub)
    for lo, hi, c in ((lo_r, hi_r, "#c0392b"), (lo_h, hi_h, "#1f6f43"), (lo_i, hi_i, "#1f5fbf")):
        ax.fill_between(tau, lo, hi, color=c, alpha=0.15, lw=0)
    ax.loglog(tau, m_r, "o-", ms=3, lw=1.4, color="#c0392b", label="Без компенсації")
    ax.loglog(tau, m_h, "s-", ms=3, lw=1.4, color="#1f6f43", label="κ̂ стале між калібруваннями")
    ax.loglog(tau, m_i, "^-", ms=3.4, lw=1.6, color="#1f5fbf", label="«плаваючий» κ̂(t): лінійна інтерполяція")
    ax.legend(loc="lower left", fontsize=8.3, framealpha=0.95); fig.tight_layout()
    fig.savefig(OUT / "adev_comparison_interp.png")
