"""
=============================
Simulation of a Dicke-type medical radiothermometer (1100-1400 MHz,
B = 300 MHz) used to verify, by means of the Allan deviation (ADEV),
the efficiency of the proposed software compensation of the HMC349
SPDT switch channel asymmetry (a1(T) != a2(T)).

Implements, step by step, the 10-step modeling algorithm described in
the article section "Моделювання ефективності компенсації асиметрії
каналів комутатора методом девіації Аллана":

    1. Input parameters of the model
    2. Reference (deterministic) antenna/reference-load powers
    3. Independent drift trajectories a1(t), a2(t)
    4. "Ideal" detector voltages in the antenna/reference phases (7),(8)
    5. Additive receiver white noise per the sensitivity formula (5)
    6. Uncompensated difference signal (9)
    7. Periodic asymmetry-coefficient estimate kappa (12)
    8. Compensated difference signal (13)
    9. Overlapping Allan deviation sigma(tau) for both realizations
    10. Comparative analysis of the two ADEV curves

Author: R. I. Maisakovskyi
"""

from pathlib import Path

import numpy as np
import allantools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "switch asymmetry sim output"
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. Input parameters of the model
# ---------------------------------------------------------------------
B = 300e6          # receiver bandwidth, Hz (1100-1400 MHz band)
p = 0.5            # antenna/reference duty cycle (optimal, symmetric)
k_B = 1.380649e-23 # Boltzmann constant, J/K

T_ant = 310.0      # equivalent brightness temperature of tissue, K
T_et = 300.0       # reference load temperature, K (fixed, known)

# Receiver noise figure -> equivalent system noise temperature.
# NF = 1.5 dB (value adopted in the article, "Чутливість" subsection)
NF_dB = 1.5
F_lin = 10 ** (NF_dB / 10)
T0 = 290.0
T_receiver = T0 * (F_lin - 1)      # ~120 K
T_sys = T_ant + T_receiver          # total system noise temperature, K

tau0 = 5e-3        # basic sample (one antenna+reference cycle), s
N = 1_000_000      # number of samples -> session length:
T_session = N * tau0
print(f"T_sys = {T_sys:.1f} K, session length = {T_session:.0f} s "
      f"({T_session/60:.1f} min), tau0 = {tau0*1e3:.1f} ms")

# Nominal (T-independent) channel transmission coefficients (linear units).
# Datasheet-consistent example: RF1 ~ 0.9 dB loss, RF2 ~ 1.1 dB loss.
a1_0 = 10 ** (-0.9 / 10)
a2_0 = 10 ** (-1.1 / 10)
G = 1.0e11         # lumped gain of the rest of the chain (arbitrary units,
                   # cancels out once the output is expressed in kelvins)

# Slow, independent drift of a1(T), a2(T): Ornstein-Uhlenbeck (mean
# reverting) process, representative of a bounded thermal drift of the
# switch die. Correlation time much longer than tau0, comparable to /
# shorter than the session length (matches the assumption formulated
# at the end of the "Калібрування" subsection).
tau_c1 = 90.0      # correlation time of channel RF1 drift, s
tau_c2 = 130.0     # correlation time of channel RF2 drift, s
sigma_rel = 0.02   # stationary relative std. dev. of each channel drift

# Calibration procedure parameters
T_cal_interval = 30.0   # interval between asymmetry-coefficient updates, s
tau_cal_meas = 1.0       # averaging time used for one calibration measurement, s


# ---------------------------------------------------------------------
# 2. Reference (deterministic) antenna / reference-load powers
# ---------------------------------------------------------------------
P_ant = k_B * T_ant * B     # ~ -89 dBm, per formula (2)
P_et = k_B * T_et * B


# ---------------------------------------------------------------------
# 3. Independent drift trajectories a1(t), a2(t) (discretized OU process)
# ---------------------------------------------------------------------
def ou_process(n, dt, tau_c, sigma_rel, rng):
    """Discrete Ornstein-Uhlenbeck process with stationary std = sigma_rel."""
    x = np.zeros(n)
    phi = np.exp(-dt / tau_c)
    sigma_step = sigma_rel * np.sqrt(1 - phi ** 2)
    noise = rng.normal(0.0, sigma_step, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x

drift1 = ou_process(N, tau0, tau_c1, sigma_rel, rng)
drift2 = ou_process(N, tau0, tau_c2, sigma_rel, rng)

a1_t = a1_0 * (1.0 + drift1)
a2_t = a2_0 * (1.0 + drift2)
kappa_true = a1_t / a2_t   # true (unknown to the receiver) asymmetry


# ---------------------------------------------------------------------
# 4. "Ideal" (noiseless) detector voltages, expressions (7) and (8)
# ---------------------------------------------------------------------
V_ant_ideal = G * a1_t * P_ant
V_et_ideal = G * a2_t * P_et


# ---------------------------------------------------------------------
# 5. Additive receiver white noise per the sensitivity formula (5)
# ---------------------------------------------------------------------
# dT_rms(tau0) = T_sys / sqrt(p (1-p) B tau0)  -- expressed directly in
# kelvins and mapped onto each voltage sample through the same
# calibration constant G * a2_0 * k_B * B that converts (9) into a
# temperature-equivalent scale (see step 6).
dT_rms_tau0 = T_sys / np.sqrt(p * (1 - p) * B * tau0)
calib_const = G * a2_0 * k_B * B     # V per kelvin, cf. expression (9)
sigma_V_noise = calib_const * dT_rms_tau0

V_ant = V_ant_ideal + rng.normal(0.0, sigma_V_noise, size=N)
V_et = V_et_ideal + rng.normal(0.0, sigma_V_noise, size=N)


# ---------------------------------------------------------------------
# 6. Uncompensated difference signal, expression (9)
#    dV = V_ant - V_et = G a2(T) [kappa(T) P_ant - P_et]
#    Converted to an equivalent temperature scale by dividing by the
#    same calibration constant, so that a perfectly compensated,
#    perfectly symmetric system reads exactly (T_ant - T_et).
# ---------------------------------------------------------------------
dV_raw = V_ant - V_et
T_meas_raw = dV_raw / calib_const + T_et   # equivalent-temperature scale


# ---------------------------------------------------------------------
# 7. Periodic estimate of the asymmetry coefficient, expression (12)
#    kappa = V_et1 / V_et2, obtained by momentarily switching the same
#    reference load to both channels. Each estimate is itself noisy
#    because it comes from an averaged measurement of finite duration
#    tau_cal_meas.
# ---------------------------------------------------------------------
cal_step = max(1, int(round(T_cal_interval / tau0)))
sigma_V_cal = calib_const * (T_sys / np.sqrt(p * (1 - p) * B * tau_cal_meas))

kappa_hat = np.empty(N)
current_kappa = 1.0
for i in range(N):
    if i % cal_step == 0:
        V_et1 = G * a1_t[i] * P_et + rng.normal(0.0, sigma_V_cal)
        V_et2 = G * a2_t[i] * P_et + rng.normal(0.0, sigma_V_cal)
        current_kappa = V_et1 / V_et2
    kappa_hat[i] = current_kappa


# ---------------------------------------------------------------------
# 8. Compensated difference signal, expression (13)
#    dV_corr = V_ant / kappa_hat - V_et
# ---------------------------------------------------------------------
dV_corr = V_ant / kappa_hat - V_et
T_meas_corr = dV_corr / calib_const + T_et


# ---------------------------------------------------------------------
# 9. Overlapping Allan deviation, sigma(tau), for both realizations
# ---------------------------------------------------------------------
rate = 1.0 / tau0
tau_max = T_session / 3
taus_req = np.logspace(np.log10(tau0), np.log10(tau_max), 50)

tau_raw, adev_raw, adev_err_raw, n_raw = allantools.oadev(
    T_meas_raw, rate=rate, data_type="freq", taus=taus_req)
tau_corr, adev_corr, adev_err_corr, n_corr = allantools.oadev(
    T_meas_corr, rate=rate, data_type="freq", taus=taus_req)


# ---------------------------------------------------------------------
# 10. Comparative analysis
# ---------------------------------------------------------------------
def slope(tau, adev, i0, i1):
    x = np.log10(tau[i0:i1])
    y = np.log10(adev[i0:i1])
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def interp_log(tau, adev, tau_q):
    """log-log linear interpolation of adev(tau) at a query point tau_q."""
    return 10 ** np.interp(np.log10(tau_q), np.log10(tau), np.log10(adev))

# slope over the first decade (short tau, white-noise region, both curves
# should follow close to the theoretical -0.5 law before drift sets in)
i_short = max(3, np.searchsorted(tau_raw, tau0 * 8))
slope_raw_short = slope(tau_raw, adev_raw, 0, i_short)
slope_corr_short = slope(tau_corr, adev_corr, 0, i_short)

# slope over the last decade (long tau): compensated should return to the
# -0.5 law, uncompensated remains drift-dominated / not yet averaging down
i_long0 = int(len(tau_raw) * 0.8)
slope_raw_long = slope(tau_raw, adev_raw, i_long0, len(tau_raw) - 1)
slope_corr_long = slope(tau_corr, adev_corr, i_long0, len(tau_corr) - 1)

idx_min_raw = int(np.argmin(adev_raw))
idx_min_corr = int(np.argmin(adev_corr))
tau_opt_raw, dT_opt_raw = tau_raw[idx_min_raw], adev_raw[idx_min_raw]
tau_opt_corr, dT_opt_corr = tau_corr[idx_min_corr], adev_corr[idx_min_corr]

# worst-case degradation of each curve across the simulated tau range
degradation_raw = adev_raw.max() / adev_raw.min()
degradation_corr = adev_corr.max() / adev_corr.min()

# improvement factor at several practically relevant integration times
tau_points = [10.0, 30.0, 100.0, 300.0, 900.0]
gains = []
for tp in tau_points:
    if tp <= tau_raw[-1]:
        vr = interp_log(tau_raw, adev_raw, tp)
        vc = interp_log(tau_corr, adev_corr, tp)
        gains.append((tp, vr, vc, vr / vc))

print("\n--- Results -------------------------------------------------")
print(f"Slope (short tau, uncompensated): {slope_raw_short:+.2f}  (theory: -0.50)")
print(f"Slope (short tau, compensated):    {slope_corr_short:+.2f}  (theory: -0.50)")
print(f"Slope (long tau,  uncompensated): {slope_raw_long:+.2f}")
print(f"Slope (long tau,  compensated):    {slope_corr_long:+.2f}")
print(f"tau_opt, uncompensated: {tau_opt_raw:.2f} s,  ADEV_min = {dT_opt_raw*1000:.1f} mK")
print(f"tau_opt, compensated:   {tau_opt_corr:.1f} s, ADEV_min = {dT_opt_corr*1000:.1f} mK")
print(f"Peak-to-floor degradation, uncompensated: x{degradation_raw:.1f}")
print(f"Peak-to-floor degradation, compensated:    x{degradation_corr:.1f}")
for tp, vr, vc, g in gains:
    print(f"At tau = {tp:6.0f} s: dT_raw = {vr*1000:8.1f} mK, dT_corr = {vc*1000:7.1f} mK, gain x{g:5.2f}")

with open(OUTPUT_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
    f.write("Switch asymmetry compensation -- Allan deviation simulation results\n")
    f.write("=====================================================================\n\n")
    f.write(f"T_sys = {T_sys:.1f} K, session = {T_session:.0f} s, tau0 = {tau0*1e3:.1f} ms, N = {N}\n")
    f.write(f"Nominal channel losses: RF1 = 0.9 dB, RF2 = 1.1 dB "
            f"(kappa_nominal = {a1_0/a2_0:.4f})\n")
    f.write(f"Drift amplitude (rel. std): {sigma_rel*100:.1f}%, "
            f"correlation times: tau_c1={tau_c1:.0f}s, tau_c2={tau_c2:.0f}s\n")
    f.write(f"Calibration interval: {T_cal_interval:.0f} s\n\n")
    f.write(f"Slope (short tau, uncompensated): {slope_raw_short:+.2f} (theory -0.50)\n")
    f.write(f"Slope (short tau, compensated):    {slope_corr_short:+.2f} (theory -0.50)\n")
    f.write(f"Slope (long tau, uncompensated):  {slope_raw_long:+.2f}\n")
    f.write(f"Slope (long tau, compensated):     {slope_corr_long:+.2f}\n")
    f.write(f"tau_opt (uncompensated) = {tau_opt_raw:.2f} s, ADEV_min = {dT_opt_raw*1000:.1f} mK\n")
    f.write(f"tau_opt (compensated)   = {tau_opt_corr:.1f} s, ADEV_min = {dT_opt_corr*1000:.1f} mK\n")
    f.write(f"Peak-to-floor degradation, uncompensated: x{degradation_raw:.1f}\n")
    f.write(f"Peak-to-floor degradation, compensated:    x{degradation_corr:.1f}\n")
    for tp, vr, vc, g in gains:
        f.write(f"At tau = {tp:.0f} s: dT_raw = {vr*1000:.1f} mK, dT_corr = {vc*1000:.1f} mK, gain x{g:.2f}\n")

np.savetxt(OUTPUT_DIR / "adev_uncompensated.csv",
           np.column_stack([tau_raw, adev_raw]),
           delimiter=",", header="tau_s,adev_K", comments="")
np.savetxt(OUTPUT_DIR / "adev_compensated.csv",
           np.column_stack([tau_corr, adev_corr]),
           delimiter=",", header="tau_s,adev_K", comments="")

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=150)
ax.loglog(tau_raw, adev_raw, "o-", ms=3, lw=1.3, color="#c0392b",
          label="Без компенсації, вираз (9)")
ax.loglog(tau_corr, adev_corr, "s-", ms=3, lw=1.3, color="#1f6f43",
          label="З компенсацією, вираз (13)")

ref_tau = np.array([tau0, tau0 * 200])
ref_adev = adev_corr[1] * (ref_tau / ref_tau[0]) ** (-0.5)
ax.loglog(ref_tau, ref_adev, "--", lw=1.0, color="gray",
          label=r"нахил $-0.5$ (білий шум)")

ax.set_xlabel(r"Час усереднення $\tau$, с")
ax.set_ylabel(r"Девіація Аллана $\sigma(\tau)$, К")
ax.set_title("Девіація Аллана вихідного сигналу\nдо та після компенсації асиметрії комутатора")
ax.grid(True, which="both", ls=":", lw=0.5)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "adev_comparison.png", dpi=200)
print(f"\nSaved output files to: {OUTPUT_DIR}")
