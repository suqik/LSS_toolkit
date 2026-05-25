"""
Angular power spectrum measurements.
"""

import numpy as np
import healpy as hp

from .database import LSS_CURVEMESH


def get_fullsky_power(mesh, config=None):
    """Measure full-sky angular power spectrum via healpy.anafast.

    Parameters
    ----------
    mesh : LSS_CURVEMESH
        Healpix map.
    config : dict, optional
        Configuration with keys:
        - lmax (int): max ell, default 3*nside-1
        - iter (int): iteration count, default 0

    Returns
    -------
    dict
        With keys: 'ell' (1D array) and 'cl' (1D array).
    """
    if config is None:
        config = {}
    lmax = config.get('lmax', 3 * mesh.nside - 1)
    cl = hp.anafast(mesh.data, lmax=lmax, iter=config.get('iter', 0))
    ell = np.arange(len(cl))
    return {'ell': ell, 'cl': cl}


def get_partsky_power(mesh, mask, config=None):
    """Measure partial-sky angular power spectrum via pymaster (NaMaster).

    Parameters
    ----------
    mesh : LSS_CURVEMESH
        Healpix map.
    mask : LSS_CURVEMESH
        Healpix mask map.
    config : dict, optional
        Configuration with keys:
        - lmax (int): max multipole
        - nside (int): nside for the field (default mesh.nside)
        - bin_width (int): delta ell per bandpower bin, default 20

    Returns
    -------
    dict
        With keys: 'ell' (bandpower centres), 'cl' (decoupled spectrum),
        'cl_coupled' (pseudo-Cl), 'cl_err' (errors).
    """
    try:
        import pymaster as nmt
    except ImportError:
        raise ImportError(
            "pymaster (NaMaster) is required for partial-sky power spectra. "
            "Install with: pip install pymaster"
        )

    if config is None:
        config = {}
    lmax = config.get('lmax', 3 * mesh.nside - 1)
    nside = config.get('nside', mesh.nside)
    bin_width = config.get('bin_width', 20)

    f = nmt.NmtField(mask.data, [mesh.data], nside=nside)

    n_bins = lmax // bin_width
    b = nmt.NmtBin.from_nside_linear(nside, bin_width, n_bins)

    ell_uncoupled, cl_coupled = nmt.compute_coupled_cell(f, f)
    ell_binned = b.get_effective_ells()
    cl_decoupled = b.decouple_cell(cl_coupled)[0]
    cl_err = np.sqrt(b.decouple_cell(
        np.diag(np.diag(cl_coupled[0]))
    )[0] * 2.0 / (2 * ell_binned + 1))

    return {'ell': ell_binned, 'cl': cl_decoupled,
            'cl_coupled': cl_coupled[0], 'cl_err': cl_err}
