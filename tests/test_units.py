import unittest

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from lss_tk.angular_2pcf import Angular_tpcf
from lss_tk.database import LSSdata


class TestDegreeNormalization(unittest.TestCase):
    def setUp(self):
        self.cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    def test_spherical_inputs_are_normalized_to_degrees(self):
        deg_cat = LSSdata(self.cosmo, sph_ra=[180.0], sph_dec=[45.0], sph_units="deg")
        rad_cat = LSSdata(self.cosmo, sph_ra=[np.pi], sph_dec=[np.pi / 4.0], sph_units="rad")
        arcmin_cat = LSSdata(self.cosmo, sph_ra=[180.0 * 60.0], sph_dec=[45.0 * 60.0], sph_units="arcmin")
        arcsec_cat = LSSdata(self.cosmo, sph_ra=[180.0 * 3600.0], sph_dec=[45.0 * 3600.0], sph_units="arcsec")

        for cat in (deg_cat, rad_cat, arcmin_cat, arcsec_cat):
            self.assertEqual(cat.sph_units, "deg")
            np.testing.assert_allclose(cat.sph_ra, [180.0], atol=1e-8)
            np.testing.assert_allclose(cat.sph_dec, [45.0], atol=1e-8)

    def test_spherical_cartesian_round_trip_keeps_degrees(self):
        cat = LSSdata(
            self.cosmo,
            sph_ra=[120.0, 240.0],
            sph_dec=[15.0, -10.0],
            sph_z=[0.2, 0.4],
            sph_units="deg",
        )

        expected_ra = cat.sph_ra.copy()
        expected_dec = cat.sph_dec.copy()

        cat.sph_to_cart()
        cat.cart_to_sph()

        self.assertEqual(cat.sph_units, "deg")
        np.testing.assert_allclose(cat.sph_ra, expected_ra, atol=1e-8)
        np.testing.assert_allclose(cat.sph_dec, expected_dec, atol=1e-8)

    def test_angular_tpcf_prepares_treecorr_catalogs_in_degrees(self):
        cat_d = LSSdata(
            self.cosmo,
            sph_ra=[10.0, 20.0, 30.0],
            sph_dec=[0.0, 1.0, -1.0],
            sph_units="deg",
            weight=[1.0, 2.0, 1.5],
        )
        cat_r = LSSdata(
            self.cosmo,
            sph_ra=[11.0, 21.0, 31.0, 41.0],
            sph_dec=[0.5, 1.5, -0.5, 0.0],
            sph_units="deg",
            weight=[1.0, 1.0, 1.0, 1.0],
        )

        corr = Angular_tpcf(min_sep=1.0, max_sep=10.0, nbins=5)
        corr.prepare_num_den_field(cat_d, cat_r)

        self.assertEqual(corr.var_method, "shot")
        np.testing.assert_allclose(corr.tc_cat_d.ra / np.pi * 180.0, cat_d.sph_ra, atol=1e-8)
        np.testing.assert_allclose(corr.tc_cat_d.dec / np.pi * 180.0, cat_d.sph_dec, atol=1e-8)
        np.testing.assert_allclose(corr.tc_cat_d.w, cat_d.weight, atol=1e-8)

    def test_match_to_reference_returns_selected_indices(self):
        ref_cat = LSSdata(
            self.cosmo,
            sph_ra=[10.0, 20.0],
            sph_dec=[0.0, 0.0],
            sph_units="deg",
        )
        cat = LSSdata(
            self.cosmo,
            sph_ra=[10.1, 19.9, 120.0],
            sph_dec=[0.1, -0.1, 30.0],
            sph_units="deg",
        )

        selection = cat.match_to_reference(ref_cat, nside_x=20, nside_y=20)

        np.testing.assert_array_equal(selection, [0, 1])

    def test_angular_tpcf_reference_matching_filters_catalogs(self):
        ref_cat = LSSdata(
            self.cosmo,
            sph_ra=[10.0, 20.0],
            sph_dec=[0.0, 0.0],
            sph_units="deg",
        )
        cat_d = LSSdata(
            self.cosmo,
            sph_ra=[10.1, 20.1, 140.0],
            sph_dec=[0.1, -0.1, 20.0],
            sph_units="deg",
            weight=[1.0, 2.0, 99.0],
        )
        cat_r = LSSdata(
            self.cosmo,
            sph_ra=[10.2, 19.8, 150.0],
            sph_dec=[0.2, -0.2, -20.0],
            sph_units="deg",
            weight=[1.0, 1.0, 55.0],
        )

        corr = Angular_tpcf(min_sep=1.0, max_sep=10.0, nbins=5)
        corr.prepare_num_den_field(cat_d, cat_r, ref_cat=ref_cat, nside_x=20, nside_y=20)

        np.testing.assert_allclose(corr.cat_d_prepared.sph_ra, [10.1, 20.1], atol=1e-8)
        np.testing.assert_allclose(corr.cat_r_prepared.sph_ra, [10.2, 19.8], atol=1e-8)
        np.testing.assert_allclose(corr.tc_cat_d.w, [1.0, 2.0], atol=1e-8)
        np.testing.assert_allclose(corr.tc_cat_r.w, [1.0, 1.0], atol=1e-8)

    def test_angular_tpcf_reference_matching_raises_on_empty_match(self):
        ref_cat = LSSdata(
            self.cosmo,
            sph_ra=[10.0],
            sph_dec=[0.0],
            sph_units="deg",
        )
        cat_d = LSSdata(
            self.cosmo,
            sph_ra=[100.0],
            sph_dec=[30.0],
            sph_units="deg",
        )
        cat_r = LSSdata(
            self.cosmo,
            sph_ra=[110.0],
            sph_dec=[-30.0],
            sph_units="deg",
        )

        corr = Angular_tpcf(min_sep=1.0, max_sep=10.0, nbins=5)

        with self.assertRaises(ValueError):
            corr.prepare_num_den_field(cat_d, cat_r, ref_cat=ref_cat, nside_x=20, nside_y=20)

    def test_prepare_shear_field_uses_degree_coordinates(self):
        cat_s = LSSdata(
            self.cosmo,
            sph_ra=[10.0, 20.0],
            sph_dec=[1.0, -1.0],
            sph_units="deg",
            weight=[1.0, 2.0],
            e_1=[0.1, -0.2],
            e_2=[0.3, -0.4],
        )

        corr = Angular_tpcf(min_sep=1.0, max_sep=10.0, nbins=5)
        corr.prepare_shear_field(cat_s)

        self.assertEqual(corr.var_method, "shot")
        np.testing.assert_allclose(corr.tc_cat_s.ra / np.pi * 180.0, cat_s.sph_ra, atol=1e-8)
        np.testing.assert_allclose(corr.tc_cat_s.dec / np.pi * 180.0, cat_s.sph_dec, atol=1e-8)
        np.testing.assert_allclose(corr.tc_cat_s.g1, cat_s.e_1, atol=1e-8)
        np.testing.assert_allclose(corr.tc_cat_s.g2, cat_s.e_2, atol=1e-8)

    def test_prepare_shear_field_can_flip_individual_components(self):
        cat_s = LSSdata(
            self.cosmo,
            sph_ra=[10.0, 20.0],
            sph_dec=[1.0, -1.0],
            sph_units="deg",
            weight=[1.0, 2.0],
            e_1=[0.1, -0.2],
            e_2=[0.3, -0.4],
        )

        corr = Angular_tpcf(min_sep=1.0, max_sep=10.0, nbins=5)
        corr.prepare_shear_field(cat_s, flip_g1=True, flip_g2=False)
        np.testing.assert_allclose(corr.g1_prepared, [-0.1, 0.2], atol=1e-8)
        np.testing.assert_allclose(corr.g2_prepared, [0.3, -0.4], atol=1e-8)

        corr.prepare_shear_field(cat_s, flip_g1=False, flip_g2=True)
        np.testing.assert_allclose(corr.g1_prepared, [0.1, -0.2], atol=1e-8)
        np.testing.assert_allclose(corr.g2_prepared, [-0.3, 0.4], atol=1e-8)

    def test_prepare_shear_field_does_not_mutate_source_catalog(self):
        cat_s = LSSdata(
            self.cosmo,
            sph_ra=[10.0, 20.0],
            sph_dec=[1.0, -1.0],
            sph_units="deg",
            weight=[1.0, 2.0],
            e_1=[0.1, -0.2],
            e_2=[0.3, -0.4],
        )

        original_e1 = cat_s.e_1.copy()
        original_e2 = cat_s.e_2.copy()

        corr = Angular_tpcf(min_sep=1.0, max_sep=10.0, nbins=5)
        corr.prepare_shear_field(cat_s, flip_g1=True, flip_g2=True)

        np.testing.assert_allclose(cat_s.e_1, original_e1, atol=1e-8)
        np.testing.assert_allclose(cat_s.e_2, original_e2, atol=1e-8)

    def test_get_tangential_shear_runs_after_preparation(self):
        cat_d = LSSdata(
            self.cosmo,
            sph_ra=[10.0, 20.0, 30.0],
            sph_dec=[0.0, 0.5, -0.5],
            sph_units="deg",
            weight=[1.0, 1.0, 1.0],
        )
        cat_r = LSSdata(
            self.cosmo,
            sph_ra=[12.0, 22.0, 32.0, 42.0],
            sph_dec=[0.1, 0.6, -0.6, 0.0],
            sph_units="deg",
            weight=[1.0, 1.0, 1.0, 1.0],
        )
        cat_s = LSSdata(
            self.cosmo,
            sph_ra=[10.5, 20.5, 30.5, 40.5],
            sph_dec=[0.2, 0.7, -0.7, 0.1],
            sph_units="deg",
            weight=[1.0, 1.0, 1.0, 1.0],
            e_1=[0.1, -0.1, 0.05, -0.05],
            e_2=[0.02, -0.02, 0.03, -0.03],
        )

        corr = Angular_tpcf(min_sep=1.0, max_sep=100.0, nbins=5)
        corr.prepare_num_den_field(cat_d, cat_r)
        corr.prepare_shear_field(cat_s)

        rnom, gamma_t, gamma_x, gamma_err = corr.get_tangential_shear()

        self.assertEqual(len(rnom), 5)
        self.assertEqual(len(gamma_t), 5)
        self.assertEqual(len(gamma_x), 5)
        self.assertEqual(len(gamma_err), 5)

    def test_get_cosmic_shear_runs_after_preparation(self):
        cat_s = LSSdata(
            self.cosmo,
            sph_ra=[10.0, 20.0, 30.0, 40.0],
            sph_dec=[1.0, -1.0, 0.5, -0.5],
            sph_units="deg",
            weight=[1.0, 1.0, 1.0, 1.0],
            e_1=[0.1, -0.1, 0.05, -0.05],
            e_2=[0.02, -0.02, 0.03, -0.03],
        )

        corr = Angular_tpcf(min_sep=1.0, max_sep=100.0, nbins=5)
        corr.prepare_shear_field(cat_s)

        rnom, xi_plus, xi_plus_err, xi_minus, xi_minus_err = corr.get_cosmic_shear()

        self.assertEqual(len(rnom), 5)
        self.assertEqual(len(xi_plus), 5)
        self.assertEqual(len(xi_plus_err), 5)
        self.assertEqual(len(xi_minus), 5)
        self.assertEqual(len(xi_minus_err), 5)

    def test_get_tangential_shear_requires_prepared_catalogs(self):
        corr = Angular_tpcf(min_sep=1.0, max_sep=100.0, nbins=5)

        with self.assertRaises(ValueError):
            corr.get_tangential_shear()

    def test_get_cosmic_shear_requires_prepared_shear_catalog(self):
        corr = Angular_tpcf(min_sep=1.0, max_sep=100.0, nbins=5)

        with self.assertRaises(ValueError):
            corr.get_cosmic_shear()


if __name__ == "__main__":
    unittest.main()
