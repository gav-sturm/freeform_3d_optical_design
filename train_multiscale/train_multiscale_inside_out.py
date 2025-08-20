"""
train_multiscale.py

An N-stage, "inside-out" training script for designing large-scale
freeform 3D refractive optics.

This script implements a robust, physically-grounded scaling strategy.
It begins with a small, stable baseline simulation and progressively
"grows" the optic's physical size and grid resolution in stages.

The process is as follows:
1.  **Start Small:** The first stage runs the known-good baseline
    simulation to create a small, converged "seed" optic.
2.  **Expand and Pad:** For each subsequent stage, a larger simulation
    grid is created. The previously trained smaller optic is placed in
    the center of this new grid, and the surrounding area is padded.
3.  **Continue Training:** This padded optic is used as the starting
    point for the next stage, allowing the optimizer to learn how to
    extend the solution outwards.

This process ensures that the simulation is physically valid and stable
at every stage of the scaling process.

Written by Andrew G. York, licensed CC-BY 4.0.
Extended and adapted by Gemini.
"""

import time
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path # Import Path for modern path handling
import re

# Assuming beam_propagation.py is in the same directory or your PYTHONPATH
from train_multiscale.beam_propagation_multiscale import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial,
    from_tif, to_tif, plot_loss_history, gaussian_beam_2d
)

##############################################################################
## CONFIGURATION
##
## This section defines the parameters for the entire multi-stage training
## process. Instead of a long, hardcoded list, we now programmatically
## generate the stages for a more concise and flexible setup.
##############################################################################

def generate_training_schedule(
    num_stages,
    start_width_um, end_width_um,
    start_n_xy, end_n_xy,
    iterations_per_stage,
    start_smoothing, end_smoothing
):
    """
    Programmatically generates a list of training stage configurations.

    This function creates a smooth transition from a small, low-res
    simulation to a large, high-res one over a specified number of stages.
    """
    # Use logarithmic spacing for physical size to make smaller steps at the beginning.
    widths = np.logspace(np.log10(start_width_um), np.log10(end_width_um), num_stages)
    # Keep voxel size constant across stages, using stage 1 as reference
    # Target voxel size based on stage 1 settings
    dxy0 = start_width_um / float(start_n_xy)
    # Derive grid points per stage to maintain constant voxel size
    n_xys = np.round(widths / dxy0).astype(int)
    smoothing_sigmas = np.linspace(start_smoothing, end_smoothing, num_stages)

    stages = []
    for i in range(num_stages):
        stage = {
            'name': f'Stage{i+1:02d}_Size{widths[i]:.0f}um',
            'physical_width_um': widths[i],
            'n_xy': n_xys[i],
            'n_z': n_xys[i], # Keep n_z the same as n_xy for simplicity
            'iterations': iterations_per_stage,
            'step_size': 100, # Keep step size constant for stability
            'smoothing_sigma': smoothing_sigmas[i],
        }
        stages.append(stage)
    return stages

# --- Generate a 10-stage training schedule ---
TRAINING_STAGES = generate_training_schedule(
    num_stages=10,
    start_width_um=25.4,   # Start with the baseline size
    end_width_um=1000.0,   # End at 1mm
    start_n_xy=128,        # Start with baseline resolution
    end_n_xy=2048,         # End with high resolution
    iterations_per_stage=1000, # Fewer iterations per stage are needed
    start_smoothing=5.0,
    end_smoothing=2.0
)

# --- Define the physical properties of the optic ---
WAVELENGTH_UM = 1.0           # Wavelength of light to simulate

# --- Define the materials for the optic ---
AIR = FixedIndexMaterial(1.0)
POLYMER = FixedIndexMaterial(1.5)

# --- Define parameters for the imaging task ---
INPUT_RADIUS_FRACTION = 0.24
DIVERGENCE_ANGLE_DEGREES = 15


##############################################################################
## HELPER FUNCTIONS
##############################################################################

# Early stopping configuration: end a stage when loss stays below threshold
# for a number of consecutive iterations. The fixed iteration count remains as
# a safety cap.
CONVERGENCE_LOSS_THRESHOLD = 0.10
CONVERGENCE_PATIENCE = 10

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

def save_diagnostic_outputs(ro, output_dir, loss_history):
    """
    Saves a comprehensive set of TIF files and plots for diagnostics
    into a stage-specific subfolder.
    """
    print(f"Saving intermediate TIFs and loss plot to {output_dir}...")
    ro.update_attributes()

    # Save the optic's physical structure
    to_tif(output_dir / 'concentration_3d.tif', ro.concentration)
    to_tif(output_dir / 'concentration_xz_slice.tif',
           ro.concentration[:, ro.coordinates.ny//2, :])

    # Save the fields for the most recent training example
    to_tif(output_dir / 'input_amplitude_2d.tif', np.abs(ro.input_field))
    to_tif(output_dir / 'input_phase_2d.tif', np.angle(ro.input_field))
    to_tif(output_dir / 'desired_output_amplitude_2d.tif', np.abs(ro.desired_output_field))
    to_tif(output_dir / 'desired_output_phase_2d.tif', np.angle(ro.desired_output_field))
    to_tif(output_dir / 'calculated_output_amplitude_2d.tif', np.abs(ro.calculated_field[-1]))
    to_tif(output_dir / 'calculated_output_phase_2d.tif', np.angle(ro.calculated_field[-1]))

    # Save 3D diagnostic volumes (names match attributes in beam_propagation.py)
    to_tif(output_dir / 'desired_output_field_3d.tif', np.abs(ro.desired_output_field_3d))
    to_tif(output_dir / 'calculated_output_field_3d.tif', np.abs(ro.calculated_output_field_3d))
    to_tif(output_dir / 'error_3d.tif', ro.error_3d)

    # Save optimization state
    to_tif(output_dir / 'gradient_3d.tif', ro.gradient)

    # Save the loss history plot
    plot_loss_history(loss_history, output_dir / 'loss_history.png')


##############################################################################
## MULTI-STAGE TRAINING INFRASTRUCTURE
##############################################################################

def run_training_stage(stage_config, input_concentration_file=None):
    """
    Executes a single stage of the multi-scale training process.
    """
    stage_name = stage_config['name']
    # Create a dedicated subfolder for this stage's outputs
    output_dir = Path('output') / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}\n--- Starting Stage: {stage_name} ---\n{'='*80}")

    # 1. Setup coordinates for the current stage's physical size and resolution.
    width = stage_config['physical_width_um']
    depth = width # Maintain a 1:1 aspect ratio for simplicity
    n_xy = stage_config['n_xy']
    n_z = stage_config['n_z']
    
    coords = Coordinates(
        xyz_i=(-width/2, -width/2, 0),
        xyz_f=(+width/2, +width/2, depth),
        n_xyz=(n_xy, n_xy, n_z)
    )
    print(f"Physical Size: {width:.2f} x {width:.2f} x {depth:.2f} um")
    print(f"Grid Resolution: {coords.n_xyz}")
    print(f"Voxel Dimensions: {coords.d_xyz[0]:.3f}, {coords.d_xyz[1]:.3f}, {coords.d_xyz[2]:.3f} um")

    # 2. Initialize the Refractive3dOptic object.
    ro = Refractive3dOptic(coords)
    ro.set_materials((AIR, POLYMER))

    # 3. Initialize the optic's concentration profile.
    if input_concentration_file is None:
        # First stage: start from scratch with random noise.
        print("Initializing optic with random noise.")
        initial_concentration = 0.5 + np.random.randn(*coords.n_xyz[::-1]) * 0.01
    else:
        # Subsequent stages: load the previous result and resample (trilinear) to match
        # the physical extent of the previous stage in the new grid's voxel spacing.
        print(f"Initializing from previous result (trilinear resample): {input_concentration_file}")
        try:
            prev_concentration = from_tif(input_concentration_file)
            # Determine shapes
            new_shape = coords.n_xyz[::-1]  # (z, y, x)

            # Parse previous stage physical width from the directory name, e.g. 'Stage01_Size25um'
            prev_stage_dir = Path(input_concentration_file).parent.name
            m = re.search(r"Size([0-9]+(?:\.[0-9]+)?)um", prev_stage_dir)
            if m is None:
                raise ValueError(f"Could not parse previous stage width from '{prev_stage_dir}'")
            prev_width_um = float(m.group(1))

            # New voxel sizes (x, y, z) in um/voxel
            dx, dy, dz = coords.d_xyz
            # Compute the number of voxels the previous physical width should occupy in the new grid
            target_nx = int(max(1, np.round(prev_width_um / dx)))
            target_ny = int(max(1, np.round(prev_width_um / dy)))
            target_nz = int(max(1, np.round(prev_width_um / dz)))

            # Ensure the target core fits inside the new grid
            target_nz = min(target_nz, new_shape[0])
            target_ny = min(target_ny, new_shape[1])
            target_nx = min(target_nx, new_shape[2])

            # Compute trilinear zoom factors from previous shape to target core shape
            prev_z, prev_y, prev_x = prev_concentration.shape
            zoom_factors = (
                target_nz / float(prev_z),
                target_ny / float(prev_y),
                target_nx / float(prev_x),
            )

            # Trilinear interpolation (order=1)
            resampled_prev = zoom(prev_concentration, zoom=zoom_factors, order=1, mode='nearest')
            resampled_prev = np.asarray(resampled_prev, dtype=np.float64)
            # Clip to valid range
            resampled_prev = np.clip(resampled_prev, 0.0, 1.0)

            # Build initial full volume filled with neutral 0.5, then paste centered
            initial_concentration = np.full(new_shape, 0.5, dtype=np.float64)
            insert_shape = resampled_prev.shape  # (z, y, x)
            start_idx_new = [(n - i) // 2 for n, i in zip(new_shape, insert_shape)]
            end_idx_new = [s + i for s, i in zip(start_idx_new, insert_shape)]
            initial_concentration[
                start_idx_new[0]:end_idx_new[0],
                start_idx_new[1]:end_idx_new[1],
                start_idx_new[2]:end_idx_new[2]
            ] = resampled_prev
        except FileNotFoundError:
            print(f"ERROR: Could not find file {input_concentration_file}. Aborting.")
            raise
            
    ro.set_3d_concentration(initial_concentration)

    # 4. Run the optimization loop for this stage.
    loss_history = []
    iterations = stage_config['iterations']
    max_radius = (width / 2.0) * INPUT_RADIUS_FRACTION
    print(f"Running up to {iterations} iterations (early stop if loss < {CONVERGENCE_LOSS_THRESHOLD} for {CONVERGENCE_PATIENCE} consecutive iters)...")
    consecutive_below_threshold = 0
    for i in range(iterations):
        start_time = time.perf_counter()

        r = max_radius * np.sqrt(np.random.random_sample())
        phi = 2 * np.pi * np.random.random_sample()
        x0, y0 = r * np.cos(phi), r * np.sin(phi)

        input_field, desired_output_field = get_training_pair(
            coords, WAVELENGTH_UM, x0, y0, DIVERGENCE_ANGLE_DEGREES
        )
        ro.set_2d_input_field(input_field, WAVELENGTH_UM)
        ro.set_2d_desired_output_field(desired_output_field)

        ro.gradient_update(
            step_size=stage_config['step_size'],
            smoothing_sigma=stage_config['smoothing_sigma']
        )
        loss_history.append((x0, y0, ro.loss))

        end_time = time.perf_counter()
        if np.isnan(ro.loss):
            print(f"ERROR: Loss is NaN at iteration {i+1}. Aborting.")
            return f"{stage_name}_FAILED.tif"

        print(f"Iter {i+1}/{iterations} | Loss: {ro.loss:.6f} | Time: {(end_time - start_time)*1000:.1f} ms")

        if (i + 1) % 100 == 0: # Save less frequently
            save_diagnostic_outputs(ro, output_dir, loss_history)

        # Early stopping check
        if ro.loss < CONVERGENCE_LOSS_THRESHOLD:
            consecutive_below_threshold += 1
        else:
            consecutive_below_threshold = 0
        if consecutive_below_threshold >= CONVERGENCE_PATIENCE:
            print(f"Converged: loss < {CONVERGENCE_LOSS_THRESHOLD} for {CONVERGENCE_PATIENCE} consecutive iterations at iter {i+1}.")
            break

    # 5. Save the final, converged result for this stage.
    print(f"Stage '{stage_name}' complete. Saving final concentration profile.")
    ro.update_attributes(delete_tensors=True)
    output_filename = output_dir / "final_concentration.tif"
    to_tif(output_filename, ro.concentration)

    return output_filename


def main():
    """
    Main function to orchestrate the entire multi-stage training process.
    """
    print("Starting multi-scale optic design process.")
    previous_stage_output_file = None
    for stage_config in TRAINING_STAGES:
        previous_stage_output_file = run_training_stage(
            stage_config,
            input_concentration_file=previous_stage_output_file
        )
        if "FAILED" in previous_stage_output_file.name:
            print("Stopping due to numerical instability.")
            break

    print(f"\n{'='*80}\nMulti-scale training finished!\n")
    if "FAILED" not in previous_stage_output_file.name:
        print(f"The final optic design is saved in:\n{previous_stage_output_file}\n{'='*80}")


if __name__ == '__main__':
    main()
