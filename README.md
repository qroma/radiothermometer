# Radiothermometer Modeling

This repository contains exploratory Python models for radiothermometry. The
scripts examine how tissue temperature profiles, microwave radiation,
frequency-dependent attenuation, antenna characteristics, and RF switch
asymmetry can affect the measured signal. The models use simplified example
parameters and are intended for simulation and visualization.

## Scripts

- `radiotermometer.py` models a temperature profile through tissue, estimates
  Rayleigh-Jeans radiation and signal attenuation at 1 GHz, and displays plots
  of temperature, radiation, and signal intensity.
- `radiotermometer with multiple GHz.py` compares signal attenuation with depth
  across frequencies from 1 to 10 GHz using a frequency-dependent absorption
  model.
- `radiotermometer antena.py` adds a localized Gaussian temperature hotspot
  and compares frequency response and example working-depth ranges for several
  antenna types.
- `switch asymmetry adev sim.py` simulates RF channel drift and receiver noise,
  then compares uncompensated and periodically compensated measurements using
  overlapping Allan deviation. It writes a plot, two CSV data files, and a
  text summary to the `switch asymmetry sim output` folder beside the script.

## Requirements and Running

The scripts use NumPy and Matplotlib. The switch-asymmetry simulation also
requires AllanTools (`allantools`). Install these packages in the Python
environment selected for this project in VS Code, then run a script with
**Run Python File**. The radiothermometer scripts display plots; the switch
simulation saves its generated files in its output folder.
