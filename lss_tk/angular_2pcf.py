"""
Angular two point correlation function.
"""

import numpy as np
import treecorr

from .database import LSSdata

class Angular_tpcf(object):
    def __init__(self, min_sep, max_sep, nbins, sep_units="arcmin", bin_type='Log', Njk=1, ncpus=1):
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nbins = nbins
        self.sep_units = sep_units
        self.bin_type = bin_type
        self.Njk = Njk
        self.ncpus = ncpus
        
    def prepare_num_den_field(self, cat_d:LSSdata, cat_r:LSSdata, ref_cat:LSSdata=None, nside_x=500, nside_y=300):
        assert len(cat_d.sph_ra) > 0, "data spherical coordinates are not provided"
        assert len(cat_r.sph_ra) > 0, "random spherical coordinates are not provided"
        work_cat_d = cat_d
        work_cat_r = cat_r

        if ref_cat is not None:
            assert len(ref_cat.sph_ra) > 0, "reference spherical coordinates are not provided"
            data_selection = cat_d.match_to_reference(ref_cat, nside_x=nside_x, nside_y=nside_y)
            random_selection = cat_r.match_to_reference(ref_cat, nside_x=nside_x, nside_y=nside_y)
            if len(data_selection) == 0:
                raise ValueError("Reference matching removed all data catalog entries")
            if len(random_selection) == 0:
                raise ValueError("Reference matching removed all random catalog entries")
            work_cat_d = cat_d.subset(data_selection)
            work_cat_r = cat_r.subset(random_selection)

        self.cat_d_prepared = work_cat_d
        self.cat_r_prepared = work_cat_r

        if self.Njk == 1:
            self.tc_cat_d = treecorr.Catalog(
                ra=work_cat_d.sph_ra,
                dec=work_cat_d.sph_dec,
                w=work_cat_d.weight,
                ra_units=work_cat_d.sph_units,
                dec_units=work_cat_d.sph_units,
            )
            self.tc_cat_r = treecorr.Catalog(
                ra=work_cat_r.sph_ra,
                dec=work_cat_r.sph_dec,
                w=work_cat_r.weight,
                ra_units=work_cat_r.sph_units,
                dec_units=work_cat_r.sph_units,
            )
            self.var_method = 'shot'
        else:
            self.tc_cat_d = treecorr.Catalog(
                ra=work_cat_d.sph_ra,
                dec=work_cat_d.sph_dec,
                w=work_cat_d.weight,
                ra_units=work_cat_d.sph_units,
                dec_units=work_cat_d.sph_units,
                npatch=self.Njk
            )
            patch_centers = self.tc_cat_d.patch_centers
            self.tc_cat_r = treecorr.Catalog(
                ra=work_cat_r.sph_ra,
                dec=work_cat_r.sph_dec,
                w=work_cat_r.weight,
                ra_units=work_cat_r.sph_units,
                dec_units=work_cat_r.sph_units,
                patch_centers=patch_centers
            )
            self.var_method = 'jackknife'

    def prepare_shear_field(self, cat_s:LSSdata, flip_g1=False, flip_g2=False):
        assert len(cat_s.sph_ra) > 0, "shear spherical coordinates are not provided"
        g1 = cat_s.e_1.copy()
        g2 = cat_s.e_2.copy()
        if g1.size == 0 or g2.size == 0:
            raise ValueError("Shear components are not provided")
        if not (np.isfinite(g1).all() and np.isfinite(g2).all()):
            raise ValueError("Shear components must be finite")
        if flip_g1:
            g1 *= -1.0
        if flip_g2:
            g2 *= -1.0

        self.cat_s_prepared = cat_s
        self.g1_prepared = g1
        self.g2_prepared = g2

        if self.Njk == 1:
            self.tc_cat_s = treecorr.Catalog(
                ra=cat_s.sph_ra,
                dec=cat_s.sph_dec,
                g1=g1,
                g2=g2,
                w=cat_s.weight,
                ra_units=cat_s.sph_units,
                dec_units=cat_s.sph_units,
            )
            self.var_method = 'shot'
        else:
            self.tc_cat_s = treecorr.Catalog(
                ra=cat_s.sph_ra,
                dec=cat_s.sph_dec,
                g1=g1,
                g2=g2,
                w=cat_s.weight,
                ra_units=cat_s.sph_units,
                dec_units=cat_s.sph_units,
                npatch=self.Njk,
            )
            self.var_method = 'jackknife'
    
    def get_angular_clustering(self):
        if not hasattr(self, "tc_cat_d") or not hasattr(self, "tc_cat_r"):
            raise ValueError("Number-density catalogs are not prepared")

        dd = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.nbins,
            sep_units=self.sep_units,
            bin_type=self.bin_type,
            var_method=self.var_method,
        )
        dr = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.nbins,
            sep_units=self.sep_units,
            bin_type=self.bin_type,
        )
        rr = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.nbins,
            sep_units=self.sep_units,
            bin_type=self.bin_type,
        )

        dd.process(self.tc_cat_d, num_threads=self.ncpus)
        dr.process(self.tc_cat_d, self.tc_cat_r, num_threads=self.ncpus)
        rr.process(self.tc_cat_r, num_threads=self.ncpus)

        wtheta, varwtheta = dd.calculateXi(dr=dr, rr=rr)

        return dd.rnom, wtheta, np.sqrt(varwtheta)

    def get_tangential_shear(self):
        if not hasattr(self, "tc_cat_d") or not hasattr(self, "tc_cat_r"):
            raise ValueError("Lens and random catalogs are not prepared")
        if not hasattr(self, "tc_cat_s"):
            raise ValueError("Shear catalog is not prepared")

        ng = treecorr.NGCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.nbins,
            sep_units=self.sep_units,
            bin_type=self.bin_type,
            var_method=self.var_method,
        )
        rg = treecorr.NGCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.nbins,
            sep_units=self.sep_units,
            bin_type=self.bin_type,
        )
  
        ng.process(self.tc_cat_d, self.tc_cat_s, num_threads=self.ncpus)
        rg.process(self.tc_cat_r, self.tc_cat_s, num_threads=self.ncpus)

        gamma_t, gamma_x, varg = ng.calculateXi(rg=rg)

        return ng.rnom, gamma_t, gamma_x, np.sqrt(varg)

    def get_cosmic_shear(self):
        if not hasattr(self, "tc_cat_s"):
            raise ValueError("Shear catalog is not prepared")

        gg = treecorr.GGCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.nbins,
            sep_units=self.sep_units,
            bin_type=self.bin_type,
            var_method=self.var_method,
        )

        gg.process(self.tc_cat_s, num_threads=self.ncpus)

        return gg.rnom, gg.xip, np.sqrt(gg.varxip), gg.xim, np.sqrt(gg.varxim)
