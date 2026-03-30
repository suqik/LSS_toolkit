'''
Define a Unified data format for measurements
'''
import astropy.cosmology
import numpy as np
import healpy as hp
import astropy

class LSSdata(object):
    def __init__(
            self, cosmo_obj:astropy.cosmology.FLRW,
            cart_x=None, cart_y=None, cart_z=None, cart_units="Mpc_h",
            sph_ra=None, sph_dec=None, sph_z=None, sph_units ="deg",
            **kwargs
            ):
        
        ### initial cosmology
        self.cosmo_obj = cosmo_obj
        
        ### Cartesian coordinates
        self.cart_x = cart_x
        self.cart_y = cart_y
        self.cart_z = cart_z
        self.cart_units = cart_units

        if self.cart_x is not None:
            assert self.cart_units in ["Mpc_h", "Mpc", "kpc", "kpc_h", "Gpc", "Gpc_h"], "Not supported distance units"
            dist_prefac = self.__dist_prefactor(self.cosmo_obj["h"], self.cart_units)
            self.cart_x *= dist_prefac
            self.cart_y *= dist_prefac
            self.cart_z *= dist_prefac
            self.cart_units = "Mpc_h"

        ### Spherical coordinates
        self.sph_ra = sph_ra
        self.sph_dec = sph_dec
        self.sph_z = sph_z
        self.sph_units = sph_units

        if self.sph_ra is not None:
            assert self.sph_units in ["rad", "deg", "arcmin", "arcsec"], "Not supported angle units"
            ang_prefac = self.__ang_prefactor(self.sph_units)
            self.sph_ra *= ang_prefac
            self.sph_dec *= ang_prefac
            self.sph_units = "rad"

        ### other properties
        for key, value in kwargs.items():
            if key == "weight":
                self.weight = value
            if key == "mass":
                self.mass = value
            if key == "radius":
                self.radius = value
            if key == "e_1":
                self.e_1 = value
            if key == "e_2":
                self.e_2 = value

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
            return 180./np.pi
        elif units == "arcmin":
            return 1./60.
        elif units == "arcsec":
            return 1./3600.
        
    def cart_to_sph(self):
        assert self.cart_x is not None, "Cartesian coordinates are not provided"
        print("Current cosmology:")
        print("Om={:.3f}, Ol={:.3f}, Ok={:.3f}, w0={:.3f}, wa={:.3f}".format(self.cosmo_dict["omega_m"], self.cosmo_dict["omega_l"], self.cosmo_dict["omega_k"], self.cosmo_dict["w0"], self.cosmo_dict["wa"]))
        radial_dist = np.sqrt(self.cart_x**2 + self.cart_y**2 + self.cart_z**2)
        self.sph_z = 1./self.cosmo_obj.scale_factor(radial_dist) - 1
        self.sph_ra, self.sph_dec = hp.vec2ang(np.array([self.cart_x, self.cart_y, self.cart_z]).T, lonlat=True)
        self.sph_units = "deg"

    def sph_to_cart(self):
        assert self.sph_ra is not None, "Spherical coordinates are not provided"
        self.cart_x, self.cart_y, self.cart_z = hp.ang2vec(self.sph_ra, self.sph_dec, lonlat=True)
        radial_dist = self.cosmo_obj.comoving_distance(self.sph_z)
        self.cart_x *= radial_dist
        self.cart_y *= radial_dist
        self.cart_z *= radial_dist
        self.cart_units = "Mpc_h"

    def convert_to_treecorr(self):
        return NotImplementedError
    
    def convert_to_healpix(self):
        return NotImplementedError
    
    def covert_to_dsigma(self):
        return NotImplementedError