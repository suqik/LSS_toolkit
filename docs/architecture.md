# architecture.md

This documents describes the architecture of the project.

## 1. Dictionary structure

```text
.
├─ lss_tk/                      # Source codes
│  ├─ database.py               # Unified data structures as measurement inputs
│  ├─ utils.py                  # Utilities used in handling data
|  ├─ angular_corr.py           # Main functions of angular correlation functions meaturements
|  ├─ angular_power.py          # Main functions of angular power spectra measurements
|  ├─ spatial_corr.py           # Main functions of spatial correlation functions measurements
|  ├─ spatial_power.py          # Main functions of spatial power spectra measurements
├─ tests/                       # Tests
├─ examples/                    # Examples of using the toolkit. Including jupyter notebooks.
└─ docs/                        # Documents
```

## 2. Module implementation

### 2.1 `database.py`

Data classes should describe the following types of data:

- `LSS_BOX`: saving box-like catalog. 
- `LSS_SKY`: saving survey-like catalog.
- `LSS_FLATMESH`: saving flat meshes.
- `LSS_CURVEMESH`: saving meshes on curved sky.

- All of the data base should support additional features (such as mass, luminosity, radius, et.al.) and meta-data (such as cosmological parameter, boxsize, particle mass, Nmesh, Nside et.al.) specified by user.

#### 2.1.1 `LSS_BOX`
- Must have 3-dimension position coordinates (x,y,z). 
- Units of position: Mpc/h.


#### 2.1.2 `LSS_SKY`
- Must have right ascension (RA) and declination (DEC). Redshift is optional.
- Units of angle: degree.

#### 2.1.3 `LSS_FLATMESH`
- Must have Nmesh as meta-data, describing the total number of meshes.

#### 2.1.4 `LSS_CURVEMESH`
- Use `healpy` to save the mesh
- Must have Nmesh as meta-data, describing the total number of meshes.

### 2.2 Measurement functions

- All of the measurement functions should have unified naming rule. Prefer to use `get_*`, in which star should be replaced by specific measurements.
- Inputs should include data, whose type is specified in following.
- Inputs should include a dictionary saving measurement configuration (such as separation bins)
- Outputs should be a dictionary, saving separation bins and measurement results. 

#### 2.2.1 `angular_corr.py`

Incorporating angular correlation function measurement functions.

- Use `treecorr` to measure the angular correlation functions.
- Support clustering (N-N correlation), tangential shear (N-G correlation), and cosmic shear (G-G correlation).
- Input data should be LSS_SKY class

#### 2.2.2 `angular_power.py`

Incorporating angular power spectrum measurement functions.

- Support full-sky and part-sky measurements.
- Use `healpy.anafast` to implement full-sky measurements.
- Use `pymaster` to implement part-sky measurements.
- Input data should include LSS_CURVEMESH.
- For part-sky, intputs should also include mask.

#### 2.2.3 `spatial_corr.py`

Leave `NotImplementedError`

#### 2.2.4 `spatial_power.py`

Leave `NotImplementedError`