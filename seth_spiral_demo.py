import time
import numpy as np
from beam_propagation import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial,
    TrainingData_for_2dImaging, from_tif, to_tif, plot_loss_history)
from pointmapping import MyTrainingData_withMapping

def example_of_usage():
    """Example code: design a 3D refractive optic with specified input/output.

    Consider copy-pasting this example code to get you started.

    In this example, the input/output is simple plane-to-plane imaging
    (with inversion). This is the same input-output you'd expect from a
    pair of ideal lenses which are cofocal and coaxial.

    We start with some (suboptimal) 3D refractive optic, and we generate
    "training data": 2D arrays of complex numbers that represent the
    amplitude and phase of optical inputs to our 3D optic. For each
    input, we specify the output that we WISH our optic would deliver,
    and then calculate the output it ACTUALLY delivers, for our current
    3D refractive optic. We use the difference between desired and
    calculated output to calculate our "loss", and use gradients of this
    loss to update our 3D refractive optic.
    """

    # Specify our coordinate system, organized via a Coordinates object:
    coords = Coordinates(xyz_i=(-12.7, -12.7,     0),
                         xyz_f=(+12.7, +12.7, +25.4),
                         n_xyz=(  128,   128,   128))
    print("Voxel dimensions: %0.3f, %0.3f, %0.3f"%(coords.d_xyz))

    # Use these coordinates to initialize an instance of Refractive3dOptic
    # that will simulate how light changes as it passes through our
    # refractive optic:
    ro = Refractive3dOptic(coords)

    # Each voxel of our refractive optic is a mixture of materials:
    air     = FixedIndexMaterial(1)
    polymer = FixedIndexMaterial(1.5)
    ro.set_materials((air, polymer))

    # Initialize our optic.
    try: # If there's a concentration saved to disk, pick up where we left off:
        fname = '01_concentration.tif'
        initial_concentration = from_tif(fname)
        ro.set_3d_concentration(initial_concentration)
        print("Using initial concentration from:", fname)
    except FileNotFoundError:
        print("Using default concentration (50/50 mixture at each voxel).")

    def map_reverse(xp, yp) -> tuple[float, float]:
        r = np.hypot(xp, yp)
        theta = np.atan2(xp, yp)

        # to swirl, add a monotonic function of r to theta then convert back.

        theta += r / 4
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return (x, y)

    # Make a source to generate training data. In this case, the
    # training data is for a simple plane-to-plane inverting imaging
    # system:
    data_source = MyTrainingData_withMapping(coords, 3, map_reverse)

    wavelength = 1
    divergence_angle_degrees = 15
    loss_history = []
    for iteration in range(int(1e6)): # Run for a loooong time
        start_time = time.perf_counter()
        
        # Use our data source to generate random input/output pairs:
        x0, y0 = data_source.random_point_in_a_circle()
        input_field, desired_output_field = data_source.input_output_pair(
            x0, y0, wavelength, divergence_angle_degrees)
        ro.set_2d_input_field(input_field, wavelength)
        ro.set_2d_desired_output_field(desired_output_field)

        # Simulate propagation through our 3D refractive optic,
        # calculate loss, and calculate a gradient that hopefully will
        # reduce the loss:
        ro.gradient_update(
            step_size=100,
            z_planes=(1, 2, 3),
            smoothing_sigma=5)
        loss_history.append((x0, y0, ro.loss))

        end_time = time.perf_counter()
        print("At iteration", iteration, "the loss is %0.4f"%(ro.loss),
              "(%0.2f ms elapsed)"%(1000*(end_time - start_time)))

        # Every so often, output some intermediate state, so we can
        # monitor our progress. You can use ImageJ
        # ( https://imagej.net/ij/ ) to view the TIF files:
        if iteration % 5 == 0:
            ro.update_attributes()
            print("Saving TIFs etc...", end='')
            to_tif('composition.tif',          ro.composition)
            to_tif('concentration.tif',        ro.concentration)
            to_tif('input_field.tif',          ro.input_field)
            to_tif('desired_output_field_amplitude.tif', np.real(ro.desired_output_field))
            to_tif('desired_output_field_phase.tif', np.angle(ro.desired_output_field))
            to_tif('calculated_field_amplitude.tif',
                   np.abs(ro.calculated_field))
            to_tif('calculated_field_phase.tif',
                   np.angle(ro.calculated_field))
            to_tif('desired_output_field_3d.tif',
                   np.abs(ro.desired_output_field_3d))
            to_tif('calculated_output_field_3d.tif',
                   np.abs(ro.calculated_output_field_3d))
            to_tif('error_3d.tif', ro.error_3d)
            to_tif('gradient.tif', ro.gradient)
            plot_loss_history(loss_history, '10_loss_history.png')
            print("done.")

if __name__ == '__main__':
    example_of_usage()
