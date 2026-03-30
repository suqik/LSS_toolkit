# LSS-toolkit

Toolkit that incorporates some useful tools for cosmological large-scale structure analysis.

## Data conventions

- Cartesian coordinates are normalized internally to `Mpc_h`.
- Spherical coordinates are normalized internally to degrees.
- Accepted angular input units are `deg`, `rad`, `arcmin`, and `arcsec`.

## Angular Correlations

- `Angular_tpcf.prepare_num_den_field(...)` accepts an optional `ref_cat` argument.
- When `ref_cat` is provided, the data and random catalogs are filtered to the sky cells shared with the reference catalog before TreeCorr catalogs are constructed.
- `Angular_tpcf.prepare_shear_field(...)` builds a TreeCorr shear catalog from `e_1` and `e_2` in an `LSSdata` catalog.
- `prepare_shear_field(...)` supports `flip_g1` and `flip_g2` to change the sign convention without mutating the source catalog.
- `Angular_tpcf.get_tangential_shear()` measures galaxy-galaxy lensing with `NGCorrelation`, using both lens and random catalogs.
- `Angular_tpcf.get_cosmic_shear()` measures cosmic shear with `GGCorrelation` and returns `rnom`, `xi_plus`, `xi_plus_err`, `xi_minus`, and `xi_minus_err`.
