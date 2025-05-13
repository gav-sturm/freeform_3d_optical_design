import numpy as np
import torch # For calculating gradients
# These imports are only used for the demo code:
# from scipy import ndimage as ndi
# import matplotlib.pyplot as plt
# import tifffile 

def main():
    """Example code: design a 3D refractive optic with specified input/output

    Techniques for fabricating freeform 3D refractive optics are rapidly
    maturing. How shall we design these optics? Gradient search, just
    like training a neural network!

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
    coords = Coordinates(xyz_i=(-10, -10,   0),
                         xyz_f=(+10, +10, +20),
                         n_xyz=(101, 101, 101))
    print("Voxel dimensions:", coords.d_xyz)

    # Use these coordinates to initialize an instance of BeamPropagation
    # that will simulate how light changes as it passes through our
    # refractive object:
    bp = BeamPropagation(coords)

    # Initialize our object. For now, we'll start with all zeros:
    try:
        initial_density = from_tif('1_density.tif')
    except FileNotFoundError:
        print("Using default initial density.")
        nx, ny, nz = coords.n_xyz
        x,   y,  z = coords.xyz
        r = np.sqrt(x**2 + y**2)
        initial_density = np.zeros((nz, ny, nx))
        initial_density[:, :, :] = 0.45 / (np.cosh(0.2*r)**2)
    bp.set_3d_density(initial_density)

    # Make a source to generate training data. In this case, the
    # training data is for a simple plane-to-plane inverting imaging
    # system:
    data_source = TrainingData_for_2dImaging(coords, radius=3)

    wavelength = 1
    loss_history = []
    for iteration in range(10000):
        # Use our data source to generate random input/output pairs:
        x0, y0 = data_source.random_point_in_a_circle()
        if iteration == 0: x0, y0 = 0, 0
        input_field, desired_field = (
            data_source.input_output_pair(x0, y0, wavelength))
        bp.set_2d_input_field(input_field, wavelength)
        bp.set_2d_desired_output_field(desired_field)

        # Simulate propagation through our 3D refractive object,
        # calculate loss, and calculate a gradient that hopefully will
        # reduce the loss:
        bp.calculate_3d_field()
        bp.calculate_loss(z_planes=(1, 2, 3))
        bp.calculate_gradient()

        # Output some intermediate state, so we can monitor our progress:
        print("At iteration", iteration, "the loss is", bp.loss)
        loss_history.append((x0, y0, bp.loss))
        if iteration % 10 == 0:
            print("Saving TIFs etc...", end='')
            to_tif('1_density.tif', bp.density)
            to_tif('2_input_field.tif', bp.input_field)
            to_tif('3_desired_field.tif', bp.desired_field)
            to_tif('4_calculated_field.tif', np.abs(bp.calculated_field))
            to_tif('5_error.tif', bp.error_3d)
            to_tif('6_gradient.tif', bp.gradient)
            plot_loss_history(loss_history, '7_loss_history.png')
            print("done.")

        # Update our 3D refractive object, using our calculated gradient:
        step_size = 1
        update = step_size * smooth(bp.gradient)
        bp.set_3d_density(bp.density - update)

##############################################################################
## The following blocks of code are the heart of the module. You should
## import the module and use these classes, similar to the demo code
## above.
##############################################################################

class Coordinates:
    """A convenience class for keeping track of the coordinates of our voxels.

     - xyz_i: a 3-element tuple storing the coordinates of our initial voxel
     - xyz_f: a 3-element tuple storing the coordinates of our final voxel
     - n_xyz: a 3-element tuple specifying how many voxels we have in x, y, z

    There's nothing complicated here, but this is the type of detail I
    tend to get wrong if I don't make a convenience class to organize it.
    """
    def __init__(self, xyz_i, xyz_f, n_xyz):
        # Sanitize and organize...
        # - Inputs:
        xi, yi, zi = map(float, xyz_i)
        xf, yf, zf = map(float, xyz_f)
        nx, ny, nz = map(  int, n_xyz)
        # If your simulation cross section is too small, you're
        # dominated by edge effects; we demand at least 1x21x21 pixels:
        assert nx > 20
        assert ny > 20
        assert nz > 0
        # - Position of voxels:
        self.xyz = (np.linspace(xi, xf, nx).reshape( 1,  1, nx),
                    np.linspace(yi, yf, ny).reshape( 1, ny,  1),
                    np.linspace(zi, zf, nz).reshape(nz,  1,  1))
        self.x, self.y, self.z = self.xyz
        # - Shape of voxels:
        dx, dy, dz = (xf-xi)/(nx-1), (yf-yi)/(ny-1), (zf-zi)/(nz-1)
        self.d_xyz = dx, dy, dz
        self.dx, self.dy, self.dz = self.d_xyz
        # - Number of voxels:
        self.n_xyz = nx, ny, nz
        self.nx, self.ny, self.nz = self.n_xyz
        return None

class BeamPropagation:
    """Simulate propagation of a light through a 3D refractive object.
    """
    def __init__(self, coordinates):
        assert isinstance(coordinates, Coordinates)
        self.coordinates = coordinates
        return None

    def set_3d_density(self, density):
        """`density` is a 3D numpy array describing our refractive object.

        You can think of `density` as the index of refraction vs 3D
        position for our object, but this neglects dispersion. The
        propagation simulation wants to know phase shifts at each voxel
        due to our refractive object, which depends on the density of
        the object and also the wavelength of the light.

        Our current beam propagation model is only accurate if `density`
        is very smoothly varying, so caveat emptor.
        """
        nx, ny, nz = self.coordinates.n_xyz
        assert density.shape == (nz, ny, nx)
        assert np.isrealobj(density)
        self.density = density.astype('float64').copy() # TODO: redundant?
        self._invalidate(( # Remove these attributes, if they exist:
            '_density_tensor', '_calculated_field_tensor', 'calculated_field',
            'error_3d', '_loss_tensor', 'loss', 'gradient'))
        return None

    def set_2d_input_field(self, input_field, wavelength):
        """What light are we shining on our refractive object?

        `input_field` is a 2D numpy array of complex numbers, specifying
        the amplitude and phase of the input light vs. 2D position at
        the input plane of our refractive object.

        `wavelength` is a positive number in the same units as our
        Coordinates object (e.g. microns). It'll be used to calculate
        how the light spreads out as it propagates through each layer of
        our refractive object, and also used to convert density to phase
        shifts.
        """
        nx, ny, nz = self.coordinates.n_xyz
        assert input_field.shape == (ny, nx)
        assert input_field.dtype == 'complex128'
        assert wavelength > 0
        self.input_field = input_field
        self.wavelength = wavelength
        self._invalidate(( # Remove these attributes, if they exist:
            'desired_field', '_calculated_field_tensor', 'calculated_field',
            'error_3d', '_loss_tensor', 'loss', 'gradient'))
        return None

    def set_2d_desired_output_field(self, desired_field):
        """What light do we wish would exit our refractive object?

        `desired_field` is a 2D numpy array of complex numbers,
        specifying the amplitude and phase of the light vs. 2D position
        that we WISH would be produced at the output plane of our
        refractive object. We use this to calculate loss (aggregate
        error between desired and calculated fields), and we take
        gradients of this loss to update our object to (hopefully) get
        closer to yielding our desired output.
        """
        self._require('input_field', 'set_2d_input_field')
        self._require('wavelength', 'set_2d_input_field')
        nx, ny, nz = self.coordinates.n_xyz
        assert desired_field.shape == (ny, nx)
        assert desired_field.dtype == 'complex128'
        self.desired_field = desired_field
        self._invalidate(( # Remove these attributes, if they exist:
            'error_3d', '_loss_tensor', 'loss', 'gradient'))
        return None

    def calculate_3d_field(self):
        """Propagate the input field through each z-slice of the volume.
        """
        self._require('input_field', 'set_2d_input_field')
        self._require('wavelength', 'set_2d_input_field')
        self._require('density', 'set_3d_density')
        nx, ny, nz = self.coordinates.n_xyz
        dx, dy, dz = self.coordinates.d_xyz
        input_field, wavelength = self.input_field, self.wavelength
        # How do amplitude and phase change from one slice to the next?
        amplitude_mask = self._apodization_amplitude_mask(edge_pixels=5)
        phase_mask     = self._propagation_phase_mask(dz)
        # Store the calculated 3D field:
        calculated_field = np.zeros((nz+1, ny, nx), dtype='complex128')
        calculated_field[0, :, :] = input_field
        # Use Torch operations in case we want to calculate gradients:
        fft, ifft, exp   = torch.fft.fftn, torch.fft.ifftn, torch.exp
        calculated_field = torch.from_numpy(calculated_field)
        amplitude_mask   = torch.from_numpy(amplitude_mask)
        phase_mask       = torch.from_numpy(phase_mask)
        # `density` is the quantity we ultimately want to update via
        # gradient search, so we need `requires_grad`:
        density          = torch.from_numpy(self.density)
        density.requires_grad_(True)
        phase_shifts     = self._density_to_phase_shifts(density, wavelength)
        # Iterate over the slices, fft, multiply, ifft, multiply.
        for i in range(nz):
            calculated_field[i+1, :, :] = (
                amplitude_mask * exp(1j * phase_shifts[i, :, :]) *
                ifft(fft(calculated_field[i, :, :]) * phase_mask))
        # Save results as attributes, not return values:
        self._density_tensor          = density
        self._calculated_field_tensor = calculated_field
        self.calculated_field        = calculated_field.detach().numpy()
        self._invalidate(( # Remove these attributes, if they exist:
            'error_3d', '_loss_tensor', 'loss', 'gradient'))
        return None

    def calculate_loss(self, z_planes=(1, 2, 3)):
        """How well does our calculated field match our desired field?

        This function doesn't return a value; it populates the
        attributes `loss`, `_loss_tensor`, and `error_3d`.
        """
        self._require('desired_field', 'set_2d_desired_output_field')
        self._require('_calculated_field_tensor', 'calculate_3d_field')
        # We want gradients, so we'll calculate our loss using Torch:
        desired_output = torch.from_numpy(self.desired_field)
        calculated_output = self._calculated_field_tensor[-1, :, :]
        # Since our fields are complex, we have to decide how to
        # penalize both intensity and phase errors. My favorite way to
        # do this is to simulated propagation in free space for both the
        # calculated and the desired fields, and compare the intensity
        # mismatch at multiple different z-planes:
        desired_output_3d    = [desired_output]
        calculated_output_3d = [calculated_output]
        for dz in z_planes:
            d_at_dz = self._freespace_propagation(desired_output,    dz)
            c_at_dz = self._freespace_propagation(calculated_output, dz)
            desired_output_3d.append(   d_at_dz)
            calculated_output_3d.append(c_at_dz)

        loss = torch.zeros(1)
        error_3d = [] # Useful for visualization
        for d, c in zip(desired_output_3d, calculated_output_3d):
            desired_intensity    = d.abs()**2
            calculated_intensity = c.abs()**2
            intensity_error = (calculated_intensity - desired_intensity)
            error_3d.append(intensity_error.detach().clone().numpy())
            worst_case_intensity_error = (desired_intensity +
                                          calculated_intensity).sum()
            loss += intensity_error.abs().sum() / worst_case_intensity_error
        loss = loss / (len(z_planes) + 1)
        # Save our results as attributes, not return values:
        self.error_3d = np.array(error_3d)
        self._loss_tensor = loss
        self.loss = loss.detach().numpy()[0]
        self._invalidate(('gradient',)) # Remove this attribute, if it exists.
        return None

    def calculate_gradient(self):
        self._require('_loss_tensor', 'calculate_loss')
        self._require('_density_tensor', 'calculate_3d_field')
        self._loss_tensor.backward()
        gradient_tensor = self._density_tensor.grad
        self.gradient = gradient_tensor.numpy()
        return None

    def _propagation_phase_mask(self, distance):
        self._require('wavelength', 'set_2d_input_field')
        nx, ny, nz = self.coordinates.n_xyz
        dx, dy, dz = self.coordinates.d_xyz
        # Spatial frequencies as a function of position in FFT space:
        k  = 2*np.pi/self.wavelength
        kx = 2*np.pi*np.fft.fftfreq(nx, dx).reshape( 1, nx)
        ky = 2*np.pi*np.fft.fftfreq(ny, dy).reshape(ny,  1)
        with np.errstate(invalid='ignore'):
            kz = np.sqrt(k**2 - kx**2 - ky**2)
        phase_mask = np.nan_to_num(np.exp(1j*kz*distance))
        return phase_mask

    def _apodization_amplitude_mask(self, edge_pixels=5, edge_value=0.5):
        # We want absorptive boundary conditions, not reflective or
        # periodic boundary conditions, so we smoothly reduce the
        # transmission amplitude near the lateral edges of the
        # simulation volume.
        edge_pixels = int(edge_pixels)
        assert edge_pixels > 0
        assert edge_value >= 0
        assert edge_value <= 1
        nx, ny, nz = self.coordinates.n_xyz
        assert nx > 2*edge_pixels
        assert ny > 2*edge_pixels
        def linear_taper(n):
            mask = np.ones(n, dtype='float64')
            mask[  0:edge_pixels] = np.linspace(edge_value, 1, edge_pixels)
            mask[-edge_pixels:] = np.linspace(1, edge_value, edge_pixels)
            return mask
        amplitude_mask = (linear_taper(nx).reshape(1, nx) *
                          linear_taper(ny).reshape(ny, 1))
        return amplitude_mask

    def _density_to_phase_shifts(self, density, wavelength):
        # This function will likely be overridden by the user, keep it simple.
        #
        # TODO: this should account for how phase shifts depend on
        # wavelength (i.e., dispersion).
        #
        # For now, just return a copy:
        return density + 0*wavelength

    def _freespace_propagation(self, field, distance):
        # Like 'calculate_3d_propagation()', but for a single step, with
        # no edge absorption and no phase shifts. We use this internally
        # to calculate the loss function.
        nx, ny, nz = self.coordinates.n_xyz
        assert field.shape == (ny, nx)
        phase_mask = self._propagation_phase_mask(distance)
        if isinstance(field, np.ndarray):     # Numpy input, return an array
            assert np.iscomplexobj(field)
            fft, ifft = np.fft.fftn, np.fft.ifftn
        elif isinstance(field, torch.Tensor): # Torch input, return a tensor
            assert torch.is_complex(field)
            phase_mask = torch.from_numpy(phase_mask)
            fft, ifft = torch.fft.fftn, torch.fft.ifftn
        field_after_propagation = ifft(fft(field) * phase_mask)
        return field_after_propagation

    def _invalidate(self, iterable_of_attribute_names):
        # Many of the methods above need to invalidate (i.e. delete)
        # multiple attributes. This makes it a little more convenient:
        for attr in iterable_of_attribute_names:
            if hasattr(self, attr):
                delattr(self, attr)
        return None

    def _require(self, attribute_name, prerequisite_function_name):
        # Many of the methods above need to be called in the expected
        # order, to create attributes that later methods depend on.
        # Check for a required attribute, and try to print a useful
        # error message if it's not present:
        if not hasattr(self, attribute_name):
            raise AttributeError(
                "No attribute `%s`. Did you call `%s()` yet?"%(
                    attribute_name, prerequisite_function_name))
        return None

##############################################################################
## The following utility code is used for the demo in the 'main' block,
## it's not critical to the module, and probably shouldn't be referenced
## in your code when you import this module.
##############################################################################

def to_tif(filename, x):
    import tifffile as tf
    if hasattr(x, 'detach'):
        x = x.detach()
    x = np.asarray(x).real.astype('float32')
    if x.ndim == 3:
        x = np.expand_dims(x, axis=(0, 2))
    tf.imwrite(filename, x, imagej=True)

def from_tif(filename):
    import tifffile as tf
    return tf.imread(filename)

def smooth(x):
    from scipy import ndimage as ndi
    return ndi.gaussian_filter(x, sigma=(0, 5, 5))

def plot_loss_history(loss_history, filename):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy import ndimage as ndi

    loss_history = np.asarray(loss_history)
    x0, y0, loss = loss_history.T
    r = np.sqrt(x0**2 + y0**2)
    smooth_loss = ndi.gaussian_filter(loss, sigma=10)
    fig = plt.figure()
    plt.scatter(range(len(loss)), loss, s=7, c=r)
    plt.plot(smooth_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.colorbar(label="Input radial position")
    plt.grid('on', alpha=0.1)
    plt.savefig(filename)
    plt.close(fig)
    return None

class TrainingData_for_2dImaging:
    """An example of how to generate training data for an imaging optic.

    This generates input/output pairs that image a pointlike source at
    the 2d input plane to an inverted (but otherwise identical) image of
    the input plane to the output plane.
    """
    def __init__(self, coordinates, radius):
        assert isinstance(coordinates, Coordinates)
        self.coordinates = coordinates
        assert radius > 0
        self.radius = radius
        return None

    def random_input_output_pair(self, wavelength):
        x0, y0 = self.random_point_in_a_circle()
        return self.input_output_pair(x0, y0, wavelength)

    def random_point_in_a_circle(self):
        # Local nicknames:
        R, sin, cos, pi, sqrt = self.radius, np.sin, np.cos, np.pi, np.sqrt
        rand = np.random.random_sample
        # Simple math:
        r, phi = R*sqrt(rand()), 2*pi*rand()
        x, y = r*cos(phi), r*sin(phi)
        return x, y

    def input_output_pair(self, x0, y0, wavelength):
        # Input beam is a focused point:
        input_field = self.gaussian_beam_2d(
            x0=x0, y0=y0, phi=0, theta=0,
            wavelength=wavelength, w=0.7*wavelength)
        # Desired output beam is an inverted image of the same point:
        desired_field = self.gaussian_beam_2d(
            x0=-x0, y0=-y0, phi=0, theta=0,
            wavelength=wavelength, w=0.7*wavelength)
        return input_field, desired_field

    def plane_wave_2d(self, x0, y0, phi, theta, wavelength):
        # Local nicknames:
        x, y, z = self.coordinates.xyz
        exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
        # Simple math:
        k = 2*pi/wavelength
        kx = k*sin(theta)*cos(phi)
        ky = k*sin(theta)*sin(phi)
        field = exp(1j*(kx*(x-x0) + ky*(y-y0)))
        return field.squeeze()
        
    def gaussian_beam_2d(self, x0, y0, phi, theta, wavelength, w):
        x, y, z = self.coordinates.xyz
        r_sq = (x-x0)**2 + (y-y0)**2
        phase = self.plane_wave_2d(x0, y0, phi, theta, wavelength)
        amplitude = np.exp(-r_sq / w**2)
        field = phase * amplitude
        return field.squeeze()

if __name__ == '__main__':
    main()
