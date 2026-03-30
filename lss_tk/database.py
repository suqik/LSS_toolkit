'''
Define a Unified data format for measurements
'''
import astropy.cosmology
import astropy.units as u
import numpy as np
import healpy as hp
import astropy
from astropy.cosmology import z_at_value

class LSSdata(object):
    def __init__(
            self, cosmo_obj:astropy.cosmology.FLRW,
            cart_x=None, cart_y=None, cart_z=None, cart_units="Mpc_h",
            sph_ra=None, sph_dec=None, sph_z=None, sph_units="deg",
            **kwargs
            ):
        
        ### initial cosmology
        self.cosmo_obj = cosmo_obj
        
        # Define the structure of the data array
        dtype = [
            ('cart_x', 'f8'), 
            ('cart_y', 'f8'), 
            ('cart_z', 'f8'),
            ('sph_ra', 'f8'),
            ('sph_dec', 'f8'),
            ('sph_z', 'f8'),
            ('weight', 'f8'),
            ('mass', 'f8'),
            ('radius', 'f8'),
            ('e_1', 'f8'),
            ('e_2', 'f8')
        ]
        
        # Determine number of entries
        n_entries = 0
        if cart_x is not None:
            n_entries = len(cart_x) if hasattr(cart_x, '__len__') else 1
        elif sph_ra is not None:
            n_entries = len(sph_ra) if hasattr(sph_ra, '__len__') else 1
        elif 'weight' in kwargs or 'mass' in kwargs or 'radius' in kwargs or 'e_1' in kwargs or 'e_2' in kwargs:
            # If only other properties are provided, assume one entry
            n_entries = 1 if not any([hasattr(v, '__len__') for v in kwargs.values()]) else len(list(kwargs.values())[0])
        
        # Initialize the structured array
        if n_entries > 0:
            self.data = np.zeros(n_entries, dtype=dtype)
            
            # Set cartesian coordinates
            if cart_x is not None:
                self.data['cart_x'] = cart_x
                self.data['cart_y'] = cart_y if cart_y is not None else 0.0
                self.data['cart_z'] = cart_z if cart_z is not None else 0.0
                self.cart_units = cart_units

                if self.data['cart_x'].any() or (hasattr(self.data['cart_x'], '__len__') and len(self.data['cart_x']) > 0):
                    assert self.cart_units in ["Mpc_h", "Mpc", "kpc", "kpc_h", "Gpc", "Gpc_h"], "Not supported distance units"
                    dist_prefac = self.__dist_prefactor(self.cosmo_obj.h, self.cart_units)
                    self.data['cart_x'] *= dist_prefac
                    self.data['cart_y'] *= dist_prefac
                    self.data['cart_z'] *= dist_prefac
                    self.cart_units = "Mpc_h"
            else:
                self.cart_units = "Mpc_h"  # Default unit
                
            # Set spherical coordinates
            if sph_ra is not None:
                self.data['sph_ra'] = sph_ra
                self.data['sph_dec'] = sph_dec if sph_dec is not None else 0.0
                self.data['sph_z'] = sph_z if sph_z is not None else 0.0
                self.sph_units = sph_units

                if self.data['sph_ra'].any() or (hasattr(self.data['sph_ra'], '__len__') and len(self.data['sph_ra']) > 0):
                    assert self.sph_units in ["rad", "deg", "arcmin", "arcsec"], "Not supported angle units"
                    ang_prefac = self.__ang_prefactor(self.sph_units)
                    self.data['sph_ra'] *= ang_prefac
                    self.data['sph_dec'] *= ang_prefac
                    self.sph_units = "deg"
            else:
                self.sph_units = "deg"  # Default unit
                
            # Set other properties from kwargs
            for key, value in kwargs.items():
                if key in ['weight', 'mass', 'radius', 'e_1', 'e_2']:
                    if value is not None:
                        self.data[key] = value
                else:
                    # Store non-standard attributes as regular class attributes
                    setattr(self, key, value)
            
            # Set default values for unset fields
            if 'weight' not in kwargs:
                self.data['weight'] = 1.0
            if 'mass' not in kwargs:
                self.data['mass'] = np.full(n_entries, np.nan)
            if 'radius' not in kwargs:
                self.data['radius'] = np.full(n_entries, np.nan)
            if 'e_1' not in kwargs:
                self.data['e_1'] = np.full(n_entries, np.nan)
            if 'e_2' not in kwargs:
                self.data['e_2'] = np.full(n_entries, np.nan)
        else:
            # Handle empty case
            self.data = np.zeros(0, dtype=dtype)
            self.cart_units = "Mpc_h"
            self.sph_units = "deg"
    
    @property
    def size(self):
        return len(self.data)

    @property
    def cart_x(self):
        return self.data['cart_x']
    
    @cart_x.setter
    def cart_x(self, value):
        self.data['cart_x'] = value
    
    @property
    def cart_y(self):
        return self.data['cart_y']
    
    @cart_y.setter
    def cart_y(self, value):
        self.data['cart_y'] = value
    
    @property
    def cart_z(self):
        return self.data['cart_z']
    
    @cart_z.setter
    def cart_z(self, value):
        self.data['cart_z'] = value
    
    @property
    def sph_ra(self):
        return self.data['sph_ra']
    
    @sph_ra.setter
    def sph_ra(self, value):
        self.data['sph_ra'] = value
    
    @property
    def sph_dec(self):
        return self.data['sph_dec']
    
    @sph_dec.setter
    def sph_dec(self, value):
        self.data['sph_dec'] = value
    
    @property
    def sph_z(self):
        return self.data['sph_z']
    
    @sph_z.setter
    def sph_z(self, value):
        self.data['sph_z'] = value
    
    @property
    def weight(self):
        return self.data['weight']
    
    @weight.setter
    def weight(self, value):
        self.data['weight'] = value
    
    @property
    def mass(self):
        return self.data['mass']
    
    @mass.setter
    def mass(self, value):
        self.data['mass'] = value
    
    @property
    def radius(self):
        return self.data['radius']
    
    @radius.setter
    def radius(self, value):
        self.data['radius'] = value
    
    @property
    def e_1(self):
        return self.data['e_1']
    
    @e_1.setter
    def e_1(self, value):
        self.data['e_1'] = value
    
    @property
    def e_2(self):
        return self.data['e_2']
    
    @e_2.setter
    def e_2(self, value):
        self.data['e_2'] = value

    def __dist_prefactor(self, h, units):
        if units == "Mpc_h":
            return 1.0
        elif units == "Mpc":
            return h
        elif units == "kpc":
            return h * 1e-3
        elif units == "kpc_h":
            return 1e-3
        elif units == "Gpc":
            return h * 1e3
        elif units == "Gpc_h":
            return 1e3
        
    def __ang_prefactor(self, units):
        '''
        Transform to `deg`
        '''
        if units == "deg":
            return 1.0
        elif units == "rad":
            return 180.0 / np.pi
        elif units == "arcmin":
            return 1.0 / 60.0
        elif units == "arcsec":
            return 1.0 / 3600.0
        
    def cart_to_sph(self):
        assert self.data['cart_x'].any() or len(self.data['cart_x']) > 0, "Cartesian coordinates are not provided"
        print("Current cosmology:")
        print("Om={:.3f}, Ol={:.3f}, Ok={:.3f}, w0={:.3f}, wa={:.3f}".format(
            self.cosmo_obj.Om0, self.cosmo_obj.Ode0, 1-self.cosmo_obj.Om0-self.cosmo_obj.Ode0, 
            getattr(self.cosmo_obj, 'w0', -1.0), getattr(self.cosmo_obj, 'wa', 0.0)))
        radial_dist = np.sqrt(self.data['cart_x']**2 + self.data['cart_y']**2 + self.data['cart_z']**2)
        radial_dist_mpc = radial_dist / self.cosmo_obj.h * u.Mpc
        self.data['sph_z'] = [
            z_at_value(self.cosmo_obj.comoving_distance, dist)
            for dist in radial_dist_mpc
        ]
        vecs = np.column_stack([self.data['cart_x'], self.data['cart_y'], self.data['cart_z']])
        ra, dec = hp.vec2ang(vecs, lonlat=True)
        self.data['sph_ra'] = ra
        self.data['sph_dec'] = dec
        self.sph_units = "deg"

    def sph_to_cart(self):
        assert self.data['sph_ra'].any() or len(self.data['sph_ra']) > 0, "Spherical coordinates are not provided"
        vecs = hp.ang2vec(self.data['sph_ra'], self.data['sph_dec'], lonlat=True)
        self.data['cart_x'], self.data['cart_y'], self.data['cart_z'] = vecs[:, 0], vecs[:, 1], vecs[:, 2]
        radial_dist = self.cosmo_obj.comoving_distance(self.data['sph_z']).to_value(u.Mpc) * self.cosmo_obj.h
        self.data['cart_x'] *= radial_dist
        self.data['cart_y'] *= radial_dist
        self.data['cart_z'] *= radial_dist
        self.cart_units = "Mpc_h"

    def subset(self, selection):
        subset_cat = self.__class__(self.cosmo_obj)
        subset_cat.data = self.data[selection].copy()
        subset_cat.cart_units = self.cart_units
        subset_cat.sph_units = self.sph_units

        for key, value in self.__dict__.items():
            if key in {"cosmo_obj", "data", "cart_units", "sph_units"}:
                continue
            setattr(subset_cat, key, value)

        return subset_cat

    def match_to_reference(self, ref_cat, nside_x=500, nside_y=300):
        assert isinstance(ref_cat, LSSdata), "Reference catalog must be an LSSdata instance"
        assert self.size > 0, "Spherical coordinates are not provided"
        assert ref_cat.size > 0, "Reference spherical coordinates are not provided"

        ref_xy = np.vstack([ref_cat.sph_ra, ref_cat.sph_dec]).T
        match_xy = np.vstack([self.data['sph_ra'], self.data['sph_dec']]).T

        all_points = np.vstack([ref_xy, match_xy])
        xmin, ymin = np.min(all_points, axis=0)
        xmax, ymax = np.max(all_points, axis=0)

        x_edges = np.linspace(xmin, xmax, nside_x + 1)
        y_edges = np.linspace(ymin, ymax, nside_y + 1)

        def compute_indices(data):
            ix = np.clip(np.digitize(data[:, 0], x_edges, right=False) - 1, 0, nside_x - 1)
            iy = np.clip(np.digitize(data[:, 1], y_edges, right=False) - 1, 0, nside_y - 1)
            return ix, iy

        ref_ix, ref_iy = compute_indices(ref_xy)
        match_ix, match_iy = compute_indices(match_xy)

        def valid_mask(ix, iy, n_x, n_y):
            return (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)

        ref_mask = valid_mask(ref_ix, ref_iy, nside_x, nside_y)
        match_mask = valid_mask(match_ix, match_iy, nside_x, nside_y)

        ref_cells = set(zip(ref_ix[ref_mask], ref_iy[ref_mask]))
        common_cells = ref_cells & set(zip(match_ix[match_mask], match_iy[match_mask]))

        match_selected = np.array(
            [i for i, (ix, iy) in enumerate(zip(match_ix, match_iy)) if (ix, iy) in common_cells],
            dtype=int,
        )

        return match_selected
