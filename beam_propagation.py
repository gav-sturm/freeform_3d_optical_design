import numpy as np
import torch # For calculating gradients
# These imports are only used for the demo code:
# from pathlib import Path
# from scipy import ndimage as ndi
# import matplotlib.pyplot as plt
# import tifffile 

"""
v1.0.3

Techniques for fabricating freeform 3D refractive optics are rapidly
maturing. By 'freeform', I don't just mean the shape - I mean optics
where at each voxel, we can specify the refractive index of the
material. This unlocks crazy new possibilities for optical design - but
how shall we design these optics?

Using gradient search, just like training a neural network!

This module defines a `BeamPropagation` class for designing freeform 3D
refractive optics, and includes some example code for how to use this
class in the `main()` block below.

Written by Andrew G. York, licensed CC-BY 4.0.

Inspired and informed by conversations with Shwetadwip Chowdhury, Tanner
Fadero, Dakota Britton, and (presumably) others I'm forgetting. Credit
them for what's good here, and blame me for what's bad. Please tell me
if I should add your name to this list!
"""

def main():
    """Example code: design a 3D refractive optic with specified input/output.

    You can execute this code by running this module, but for "normal"
    use, you should import this module, and write your own code,
    copy-pasting this example code to get you started.

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
    # If you're copy-pasting this example code, you'll need to put some
    # import statements at the start of your script, probably something
    # like this:
    #
    # import numpy as np
    # from beam_propagation import Coordinates, BeamPropagation
    #
    # You'll also probably want to copy-paste the function definitions
    # from the end of this module (e.g. `to_tif()`, `from_tif()`, etc),
    # along with the if __name__ == '__main__': block.

    # Specify our coordinate system, organized via a Coordinates object:
    coords = Coordinates(xyz_i=(-10, -10,   0),
                         xyz_f=(+10, +10, +20),
                         n_xyz=(101, 101, 101))
    print("Voxel dimensions:", coords.d_xyz)

    # Use these coordinates to initialize an instance of BeamPropagation
    # that will simulate how light changes as it passes through our
    # refractive object:
    bp = BeamPropagation(coords)

    # Each voxel of our refractive index is a mixture of materials:
    air     = FixedIndexMaterial(1)
    polymer = FixedIndexMaterial(1.5)
    bp.set_materials((air, polymer))

    # Initialize our object.
    try: # If there's an object saved to disk, pick up where we left off:
        initial_density = from_tif('1_density.tif')
    except FileNotFoundError:
        # If not, initialize either an empty array or a simple
        # pseudoparabolic GRIN lens:
        print("Using default initial density.")
        nx, ny, nz = coords.n_xyz
        x,   y,  z = coords.xyz
        r = np.sqrt(x**2 + y**2)
        initial_density = np.zeros((nz, ny, nx))
        # If you want to initialize with a pretty good initial guess,
        # uncomment this line:
        ##initial_density[:, :, :] = 0.45 / (np.cosh(0.2*r)**2)
    bp.set_3d_density(initial_density)

    # Make a source to generate training data. In this case, the
    # training data is for a simple plane-to-plane inverting imaging
    # system:
    data_source = TrainingData_for_2dImaging(coords, radius=3)

    wavelength = 1
    divergence_angle_degrees = 15
    loss_history = []
    for iteration in range(10000):
        # Use our data source to generate random input/output pairs:
        x0, y0 = data_source.random_point_in_a_circle()
        if iteration == 0: x0, y0 = 0, 0
        input_field, desired_field = data_source.input_output_pair(
            x0, y0, wavelength, divergence_angle_degrees)
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
        step_size = 10
        update = step_size * smooth(bp.gradient)
        bp.set_3d_density(bp.density - update)

##############################################################################
## The following blocks of code are the heart of the module. You should
## import the module and use these classes, similar to the demo code
## above.
##############################################################################

class BeamPropagation:
    """Simulate propagation of a light through a 3D refractive object.
    """
    def __init__(self, coordinates, try_cuda=True):
        assert isinstance(coordinates, Coordinates)
        self.coordinates = coordinates
        assert try_cuda in (True, False)
        self.device = torch.device('cpu')
        if try_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        return None

    def set_materials(self, material_list):
        """What materials are we mixing to control the index of refraction?

        The index of refraction at each voxel is the weighted average of
        two (or more) materials, our 'base' material, and our
        'mixer(s)'. Use this function to specify those materials.

        For real materials, the index of refraction depends on the
        wavelength, and you probably want to use a SellmeierMaterial
        object. For example, if each voxel was a mixture of air and
        fused silica, we could write:
        
            air = SellemeierMaterial(
                B=(0.05792105, 0.00167917),
                C=(238.0185, 57.362))
            fused_silica = SellmeierMaterial(
                B=(0.6961663, 0.4079426, 0.8974794),
                C=(0.004679148, 0.01351206, 97.934))
            material_list = [air, fused_silica]

        For simpler simulations that neglect dispersion, you could use a
        fictitious but convenient FixedIndexMaterial:

            air =          FixedIndexMaterial(1)
            fused_silica = FixedIndexMaterial(1.46)
            material_list = [air, fused_silica]
        """
        # For now, we only allow binary mixtures:
        assert len(material_list) == 2
        for m in material_list:
            assert hasattr(m, 'get_index')
        self.material_list = material_list
        self._invalidate(( # Remove these attributes, if they exist:
            'density', '_density_tensor', '_calculated_field_tensor',
            'calculated_field', 'error_3d', '_loss_tensor', 'loss', 'gradient'))
        return None

    def set_3d_density(self, density):
        """`density` is a 3D numpy array describing our refractive object.

        A density of 0 corresponds to a voxel that's entirely the 'base'
        material. A density of 1 corresponds to a voxel that's entirely
        the 'mixer' material.

        Our current beam propagation model is only accurate if `density`
        is very smoothly varying, so caveat emptor.
        """
        self._require('material_list', 'set_materials')
        nx, ny, nz = self.coordinates.n_xyz
        assert density.shape == (nz, ny, nx)
        assert np.isrealobj(density)
        self.density = density.astype('float64', copy=True)
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
        Coordinates object (e.g. microns). Note that we're specifying
        the wavelength of light our input field in *vacuum*, not in our
        base material. This is used to calculate how the light spreads
        out as it propagates through each layer of our refractive
        object, and also used to convert density to index of refraction.

        If you're simulating dispersion using a SellmeierMaterial, then
        the units of `wavelength` need to be microns.
        """
        self._require('material_list', 'set_materials')
        nx, ny, nz = self.coordinates.n_xyz
        assert input_field.shape == (ny, nx)
        assert input_field.dtype == 'complex128'
        assert wavelength > 0
        self.input_field = input_field
        self.wavelength = wavelength
        self.index_list = [m.get_index(wavelength) for m in self.material_list]
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

        I think doi.org/10.1364/AO.17.003990 is the OG reference for the
        algorithm we're currently using here to simulate propagation.
        """
        self._require('input_field', 'set_2d_input_field')
        self._require('wavelength', 'set_2d_input_field')
        self._require('density', 'set_3d_density')
        # How do amplitude and phase change from one slice to the next?
        # Regardless of the object, we want to propagate "between"
        # slices in a homogenous medium with absorbing boundary
        # conditions:
        amplitude_mask = self._apodization_amplitude_mask(edge_pixels=5)
        phase_mask     = self._propagation_phase_mask(self.coordinates.dz)
        # Use Torch tensors so we can calculate gradients:
        fft, ifft, exp = torch.fft.fftn, torch.fft.ifftn, torch.exp
        amplitude_mask = self._to_torch(amplitude_mask)
        phase_mask     = self._to_torch(phase_mask)
        input_field    = self._to_torch(self.input_field)
        self._density_tensor = self._to_torch(self.density) # List of 2D tensors
        # Propagate the light through the object, one slice at a time:
        calculated_field = [input_field]
        for d in self._density_tensor:
            # The density is the quantity we ultimately want to update
            # via gradient search, so it `requires_grad`:
            d.requires_grad_(True)
            # Convert the density to phase shifts:
            phase_shifts = self._density_to_phase_shifts(d)
            # fft, multiply, ifft, multiply:
            calculated_field.append(
                amplitude_mask * exp(1j * phase_shifts) *
                ifft(phase_mask * fft(calculated_field[-1])))
        # Save results as attributes, not return values:
        self._calculated_field_tensor = calculated_field # List of 2D tensors
        self.calculated_field = self._to_numpy(calculated_field) # 3D array
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
        desired_output = self._to_torch(self.desired_field)
        calculated_output = self._calculated_field_tensor[-1]
        # Since our fields are complex, we have to decide how to
        # penalize both intensity and phase errors. My favorite way to
        # do this is to simulate propagation in free space for both the
        # calculated and the desired fields, and compare the intensity
        # mismatch at multiple different z-planes:
        desired_output_3d    = [desired_output]    # List of 2D tensors
        calculated_output_3d = [calculated_output] # List of 2D tensors
        for dz in z_planes:
            d_at_dz = self._freespace_propagation(desired_output,    dz)
            c_at_dz = self._freespace_propagation(calculated_output, dz)
            desired_output_3d.append(   d_at_dz)
            calculated_output_3d.append(c_at_dz)

        loss = torch.zeros(1, device=self.device)
        error_3d = [] # Useful for visualization
        for d, c in zip(desired_output_3d, calculated_output_3d):
            desired_intensity    = d.abs()**2
            calculated_intensity = c.abs()**2
            intensity_error = (calculated_intensity - desired_intensity)
            error_3d.append(intensity_error)
            worst_case_intensity_error = (desired_intensity +
                                          calculated_intensity).sum()
            loss += intensity_error.abs().sum() / worst_case_intensity_error
        loss = loss / (len(z_planes) + 1)
        # Save our results as attributes, not return values:
        self.error_3d = self._to_numpy(error_3d)
        self._loss_tensor = loss
        self.loss = self._to_numpy(loss)[0]
        self._invalidate(('gradient',)) # Remove this attribute, if it exists.
        return None

    def calculate_gradient(self):
        self._require('_loss_tensor', 'calculate_loss')
        self._require('_density_tensor', 'calculate_3d_field')
        self._loss_tensor.backward()
        self.gradient = self._to_numpy([d.grad for d in self._density_tensor])
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

    def _density_to_phase_shifts(self, density):
        """Convert our `density` tensor to phase shifts at each voxel.

        The propagation simulation wants to know phase shifts at each
        voxel due to our refractive object, which depends on the
        `density` at each voxel, the materials that we're mixing, the
        size of the voxel, and the wavelength of the propagating light.
        """
        # `density` must be a tensor, to allow autograd:
        assert isinstance(density, torch.Tensor)
        # Local nicknames for the scalars:
        dz, wavelength, pi = self.coordinates.dz, self.wavelength, np.pi

        # The index of refraction is a weighted average of our
        # materials. For now, we only implement binary mixtures:
        assert len(self.index_list) == 2
        index_1, index_2 = self.index_list
        index_minus_1 = (index_1 - 1) + (index_2 - index_1)*density
        
        # The default material is nondispersive, meaning the phase
        # shifts scale with dz and inversely with wavelength:
        phase_shifts = 2*pi * dz * index_minus_1 / wavelength
        
        return phase_shifts # This is a pytorch Tensor (which allows autograd)

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
            phase_mask = self._to_torch(phase_mask)
            fft, ifft = torch.fft.fftn, torch.fft.ifftn
        field_after_propagation = ifft(phase_mask * fft(field))
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

    def _to_torch(self, x):
        """We convert 2D numpy arrays directly to 2D tensors, but we
        convert 3D numpy arrays to lists of 2D tensors rather than a 3D
        tensor, to avoid quadratic complexity during backpropagation.
        """
        assert x.ndim in (2, 3)
        if x.ndim == 2:
            return torch.from_numpy(x).to(self.device)
        elif x.ndim == 3:
            return [torch.from_numpy(x[i, :, :]).to(self.device)
                    for i in range(x.shape[0])]

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor): # Convert directly to a 2D array
            assert x.ndim in (1, 2)
            return x.cpu().detach().numpy()
        elif isinstance(x, list): # Convert to a 3D numpy array
            # This is kinda janky but seems correct at least.
            x0 = x[0].cpu().detach().numpy()
            nz = len(x)
            ny, nx = x0.shape
            out = np.zeros((nz, ny, nx), dtype=x0.dtype)
            out[0, :, :] = x0
            for i in range(1, nz):
                out[i, :, :] = x[i].cpu().detach().numpy()
            return out

class SellmeieirMaterial:
    """The Sellmeier equation is an empirical relationship between
    refractive index and wavelength for a particular transparent medium.

    https://en.wikipedia.org/wiki/Sellmeier_equation
    """
    def __init__(self, B=(0, 0, 0), C=(0, 0, 0)):
        """Three terms is typical (one per resonance), but we might as
        well allow any number of resonances.
        """
        assert len(B) == len(C)
        B = [float(x) for x in B]
        C = [float(x) for x in C]
        self.B = B
        self.C = C
        return None

    def get_index(self, wavelength_um):
        lamda_sq = wavelength_um**2
        index_sq = 1
        for b, c in zip(self.B, self.C):
            index_sq += b * lamda_sq / (lamda_sq - c)
        index = np.sqrt(index_sq)
        return index

class FixedIndexMaterial:
    """A conceptually simple material with no dispersion.

    In real life, the index of refraction for a material depends on the
    wavelength. However, sometimes it's nice to simulate simple things,
    so here's a (fictitious) material that has the same index of
    refraction for all wavelengths.
    """
    def __init__(self, index):
        index = float(index)
        self.index = index
        return None

    def get_index(self, wavelength_um):
        return self.index
            
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

##############################################################################
## The following utility code is used for the demo in the 'main' block,
## it's not critical to the module, and probably shouldn't be referenced
## in your code when you import this module.
##############################################################################

def output_directory():
    # Put all the files that the demo code makes in their own folder:
    from pathlib import Path
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    return output_dir

def to_tif(filename, x):
    import tifffile as tf
    if hasattr(x, 'detach'):
        x = x.detach()
    x = np.asarray(x).real.astype('float32')
    if x.ndim == 3:
        x = np.expand_dims(x, axis=(0, 2))
    tf.imwrite(output_directory() / filename, x, imagej=True)

def from_tif(filename):
    import tifffile as tf
    return tf.imread(output_directory() / filename)

def smooth(x):
    from scipy import ndimage as ndi
    return ndi.gaussian_filter(x, sigma=(0, 5, 5))

def plot_loss_history(loss_history, filename):
    import matplotlib as mpl
    mpl.use('agg') # Prevents a memory leak from repeated plotting
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
    plt.savefig(output_directory() / filename)
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

    def random_point_in_a_circle(self):
        # Local nicknames:
        R, sin, cos, pi, sqrt = self.radius, np.sin, np.cos, np.pi, np.sqrt
        rand = np.random.random_sample
        # Simple math:
        r, phi = R*sqrt(rand()), 2*pi*rand()
        x, y = r*cos(phi), r*sin(phi)
        return x, y

    def input_output_pair(
        self,
        x0,
        y0,
        wavelength,
        divergence_angle_degrees,
        phi=0,
        theta=0,
        ):
        x0, y0 = float(x0), float(y0)
        wavelength = float(wavelength)
        divergence_angle, pi = np.deg2rad(divergence_angle_degrees), np.pi
        w = wavelength / (pi*divergence_angle)
        # Input beam is a focused point:
        x, y, _ = self.coordinates.xyz
        input_field = gaussian_beam_2d(
            x=x, y=y, x0=x0, y0=y0, phi=phi, theta=theta,
            wavelength=wavelength, w=w)
        # Desired output beam is an inverted image of the same point:
        desired_field = input_field[::-1, ::-1].copy()
        return input_field, desired_field
        
def gaussian_beam_2d(x, y, x0, y0, phi, theta, wavelength, w):
    # Local nicknames:
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    # Simple math:
    k = 2*pi/wavelength
    kx = k*sin(theta)*cos(phi)
    ky = k*sin(theta)*sin(phi)
    r_sq = (x-x0)**2 + (y-y0)**2
    phase = exp(1j*(kx*(x-x0) + ky*(y-y0)))
    amplitude = exp(-r_sq / w**2)
    field = phase * amplitude
    return field.squeeze()

if __name__ == '__main__':
    main()
