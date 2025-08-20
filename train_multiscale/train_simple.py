"""
train_baseline.py

A simplified, single-stage training script to establish a stable baseline.

This script reverts to the exact parameters and optical problem from the
original `example_of_usage.py` script. The goal is to ensure the core
optimization process is working correctly on a small, known problem
before attempting to scale up.

The optical task is a simple plane-to-plane inverting imaging system.

Written by Andrew G. York, licensed CC-BY 4.0.
Extended and adapted by Gemini.
"""

import time
import numpy as np
from scipy.ndimage import zoom

# Assuming beam_propagation.py is in the same directory or your PYTHONPATH
from train_multiscale.beam_propagation_multiscale import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial,
    from_tif, to_tif, plot_loss_history, gaussian_beam_2d
)

##############################################################################
## CONFIGURATION
##
## This section defines the parameters for our single baseline simulation,
## matching the original `example_of_usage.py`.
##############################################################################

# --- Define the physical properties of the optic ---
# All units are in micrometers (microns)
PHYSICAL_WIDTH_UM = 25.4  # Corresponds to a range of -12.7 to +12.7 um
PHYSICAL_DEPTH_UM = 25.4  # Corresponds to a range of 0 to 25.4 um
WAVELENGTH_UM = 1.0       # Wavelength of light to simulate

# --- Define the grid resolution ---
N_XY = 128
N_Z = 128

# --- Define training parameters ---
ITERATIONS = 2000
STEP_SIZE = 100
SMOOTHING_SIGMA = 5.0

# --- Define the materials for the optic ---
AIR = FixedIndexMaterial(1.0)
POLYMER = FixedIndexMaterial(1.5)

# --- Define parameters for the imaging task ---
# The original script used a radius of 3 um on a 25.4 um wide grid.
# 3.0 / (25.4 / 2) = 0.236
INPUT_RADIUS_FRACTION = 0.24
DIVERGENCE_ANGLE_DEGREES = 15 # Controls how tightly focused the input points are


##############################################################################
## OPTICAL TASK DEFINITION
##############################################################################

def get_training_pair(coords, wavelength, x0, y0, divergence_angle_degrees):
    """
    Generates an energy-conserving input/output pair for the imaging task.
    """
    x, y, _ = coords.xyz
    divergence_angle_rad = np.deg2rad(divergence_angle_degrees)
    w = wavelength / (np.pi * divergence_angle_rad)
    input_field = gaussian_beam_2d(
        x=x, y=y, x0=x0, y0=y0, phi=0, theta=0,
        wavelength=wavelength, w=w)
    desired_output_field = input_field[::-1, ::-1].copy()
    return input_field.squeeze(), desired_output_field.squeeze()


##############################################################################
## MAIN TRAINING SCRIPT
##############################################################################

def main():
    """
    Main function to run the single-stage baseline training.
    """
    print("--- Starting Baseline Training ---")

    # 1. Setup coordinates for the simulation.
    coords = Coordinates(
        xyz_i=(-PHYSICAL_WIDTH_UM/2, -PHYSICAL_WIDTH_UM/2, 0),
        xyz_f=(+PHYSICAL_WIDTH_UM/2, +PHYSICAL_WIDTH_UM/2, PHYSICAL_DEPTH_UM),
        n_xyz=(N_XY, N_XY, N_Z)
    )
    print(f"Physical Size: {PHYSICAL_WIDTH_UM:.2f} x {PHYSICAL_WIDTH_UM:.2f} x {PHYSICAL_DEPTH_UM:.2f} um")
    print(f"Grid Resolution: {coords.n_xyz}")
    print(f"Voxel Dimensions: {coords.d_xyz[0]:.3f}, {coords.d_xyz[1]:.3f}, {coords.d_xyz[2]:.3f} um")

    # 2. Initialize the Refractive3dOptic object.
    ro = Refractive3dOptic(coords)
    ro.set_materials((AIR, POLYMER))

    # 3. Initialize the optic's concentration profile from random noise.
    print("Initializing optic with random noise.")
    initial_concentration = 0.5 + np.random.randn(*coords.n_xyz[::-1]) * 0.01
    ro.set_3d_concentration(initial_concentration)

    # 4. Run the optimization loop.
    loss_history = []
    max_radius = (PHYSICAL_WIDTH_UM / 2.0) * INPUT_RADIUS_FRACTION
    print(f"Running {ITERATIONS} training iterations...")
    for i in range(ITERATIONS):
        start_time = time.perf_counter()

        # Generate a random point within the circular input area.
        r = max_radius * np.sqrt(np.random.random_sample())
        phi = 2 * np.pi * np.random.random_sample()
        x0, y0 = r * np.cos(phi), r * np.sin(phi)

        # Generate the training pair for the random (x0, y0).
        input_field, desired_output_field = get_training_pair(
            coords, WAVELENGTH_UM, x0, y0, DIVERGENCE_ANGLE_DEGREES
        )
        ro.set_2d_input_field(input_field, WAVELENGTH_UM)
        ro.set_2d_desired_output_field(desired_output_field)

        # Perform one step of gradient descent.
        ro.gradient_update(
            step_size=STEP_SIZE,
            smoothing_sigma=SMOOTHING_SIGMA
        )
        loss_history.append((x0, y0, ro.loss))

        end_time = time.perf_counter()
        if np.isnan(ro.loss):
            print(f"ERROR: Loss is NaN at iteration {i+1}. Aborting.")
            break

        print(f"Iter {i+1}/{ITERATIONS} | Loss: {ro.loss:.6f} | Time: {(end_time - start_time)*1000:.1f} ms")

        # Periodically save intermediate results for visualization.
        if (i + 1) % 100 == 0:
            print("Saving intermediate TIFs and loss plot...")
            ro.update_attributes()
            to_tif('baseline_concentration_3d.tif', ro.concentration)
            to_tif('baseline_output_amplitude_2d.tif', np.abs(ro.calculated_field[-1]))
            plot_loss_history(loss_history, 'baseline_loss_history.png')

    # 5. Save the final, converged result for this stage.
    print("Baseline training complete. Saving final concentration profile.")
    ro.update_attributes(delete_tensors=True)
    output_filename = "baseline_final_concentration.tif"
    to_tif(output_filename, ro.concentration)

    print(f"\n{'='*80}\nBaseline training finished!\n")
    print(f"The final optic design is saved in:\n{output_filename}\n{'='*80}")


if __name__ == '__main__':
    main()
