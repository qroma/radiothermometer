"""
Моделювання девіації Аллана (ADEV) вихідного сигналу модуляційного радіотермометра
до та після програмної компенсації асиметрії каналів комутатора.

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
    kappa_hat = np.repeat(e1 / e2, step)[:N]
    corr = V_ant / kappa_hat - V_et

    t_r, a_r, _, _ = allantools.oadev(raw, rate=1 / TAU0, data_type="freq", taus=taus_req)
    t_c, a_c, _, _ = allantools.oadev(corr, rate=1 / TAU0, data_type="freq", taus=taus_req)
    return t_r, a_r, a_c


if __name__ == "__main__":
    taus_req = np.logspace(np.log10(TAU0), np.log10(T_SESSION / 10), 40)
    runs = [one_run(100 + s, taus_req) for s in range(N_REAL)]
    tau = runs[0][0]
    raw = np.array([r[1] for r in runs]); cor = np.array([r[2] for r in runs])
    med_r, lo_r, hi_r = np.median(raw, 0), *np.percentile(raw, [16, 84], axis=0)
    med_c, lo_c, hi_c = np.median(cor, 0), *np.percentile(cor, [16, 84], axis=0)

    # --- підсумок ---
    lines = [f"T_sys = {T_SYS:.1f} K, сеанс = {T_SESSION:.0f} с, реалізацій = {N_REAL}"]
    for q in (1, 10, 30, 100, 300, 900):
        i = int(np.argmin(abs(tau - q)))
        lines.append(f"tau={tau[i]:7.1f} с: без компенсації {med_r[i]:.3f} К, з компенсацією {med_c[i]:.3f} К, "
                     f"виграш ×{med_r[i] / med_c[i]:.2f}")
    ir, ic = int(np.argmin(med_r)), int(np.argmin(med_c))
    lines.append(f"мінімум без компенсації: {med_r[ir]:.3f} К при tau={tau[ir]:.2f} с; "
                 f"з компенсацією: {med_c[ic]:.3f} К при tau={tau[ic]:.1f} с")
    wn0 = 2 * T_SYS / np.sqrt(B * TAU0)
    lines.append(f"теорія білого шуму при tau0: {wn0:.3f} К; дані (без компенсації): {med_r[0]:.3f} К")
    print("\n".join(lines)); (OUT / "adev_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # --- графік ---
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=170)
    ax.fill_between(tau, lo_r, hi_r, color="#c0392b", alpha=0.18, lw=0)
    ax.fill_between(tau, lo_c, hi_c, color="#1f6f43", alpha=0.18, lw=0)
    ax.loglog(tau, med_r, "o-", ms=3, lw=1.4, color="#c0392b", label="Без компенсації (медіана)")
    ax.loglog(tau, med_c, "s-", ms=3, lw=1.4, color="#1f6f43", label="З компенсацією (медіана)")
    tw = np.array([TAU0, T_SESSION / 10])
    ax.loglog(tw, 2 * T_SYS / np.sqrt(B * tw), "--", color="gray", lw=1.1,
              label=r"білий шум: $2T_{sys}/\sqrt{B\tau}$ (теорія)")
    ax.axvline(T_CAL, color="gray", ls=":", lw=1)
    ax.text(T_CAL * 1.08, ax.get_ylim()[0] * 1.6, "інтервал\nкалібрування", fontsize=7.5, color="gray", va="bottom")
    ax.set_xlabel(r"Час усереднення $\tau$, с")
    ax.set_ylabel(r"Девіація Аллана $\sigma(\tau)$, К")
    ax.set_title(f"Девіація Аллана до та після компенсації асиметрії комутатора\n"
                 f"(медіана та інтервал 16–84% за {N_REAL} реалізаціями)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout(); fig.savefig(OUT / "adev_comparison_corrected.png")
