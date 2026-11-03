import xarray as xr

from gtpost.analyze import classifications

# Compression parameters
ENCODINGS = {
    "d50_per_sedclass": {"zlib": True, "complevel": 9},
    # "xcor": {"zlib": True, "complevel": 9},
    # "ycor": {"zlib": True, "complevel": 9},
    "zcor": {"zlib": True, "complevel": 9},
    "subsidence": {"zlib": True, "complevel": 9},
    "preserved_thickness": {"zlib": True, "complevel": 9},
    "diameter": {"zlib": True, "complevel": 9},
    "fraction": {"zlib": True, "complevel": 9},
    "sorting": {"zlib": True, "complevel": 9},
    "porosity": {"zlib": True, "complevel": 9},
    "permeability": {"zlib": True, "complevel": 9},
    "vfractions": {"zlib": True, "complevel": 9},
    "mfractions": {"zlib": True, "complevel": 9},
    "archel": {"zlib": True, "complevel": 9},
    "subenv": {"zlib": True, "complevel": 9},
}


def create_sed_and_obj_dataset(p):
    data_vars = dict(
        d50_per_sedclass=(
            "sed_class",
            p.d50_input,
            dict(
                long_name="D50 values defining each sediment class",
                variable_type="D3D output",
                units="meter",
            ),
        ),
        # xcor=(
        #     ("dimen_x", "dimen_y"),
        #     p.dataset_rasterized.x.values,
        #     dict(
        #         long_name="x coordinates of cells",
        #         variable_type="D3D output",
        #         units="meter",
        #     ),
        # ),
        # ycor=(
        #     ("dimen_x", "dimen_y"),
        #     p.dataset.YCOR.values,
        #     dict(
        #         long_name="y coordinates of cells",
        #         variable_type="D3D output",
        #         units="meter",
        #     ),
        # ),
        zcor=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_zcor"].data,
            dict(
                long_name="z-coordinate of surface",
                variable_type="D3D output",
                units="meter",
            ),
        ),
        subsidence=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_subsidence"].data,
            dict(
                long_name="subsidence in meter per timestep",
                variable_type="D3D output",
                units="meter",
            ),
        ),
        deposit_height=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_depositheight"].data,
            dict(
                long_name="height of deposit",
                variable_type="calculated",
                units="meter",
                notes="only values larger than 0. If erosion took place, deposit height is 0",
            ),
        ),
        preserved_thickness=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_preserved_thickness"].data,
            dict(
                long_name="fraction of deposit height that will have been preserved at the end of simulation",
                variable_type="calculated",
                units="meter",
                notes="only values larger than 0. 0 means that the deposit will have eroded completely by the end of simulation, whereas 1 means that it will be fully preserved",
            ),
        ),
        deposition_age=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_deposition_age"].data,
            dict(
                long_name="timestep at which deposition took place",
                variable_type="calculated",
                units="-",
            ),
        ),
        diameter=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_d50"].data,
            dict(
                long_name="D50 of sediment",
                variable_type="calculated",
                units="meter",
            ),
        ),
        fraction=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_sandfraction"].data,
            dict(
                long_name="fraction of sediment that is sand",
                variable_type="calculated",
                units="-",
            ),
        ),
        sorting=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_sorting"].data,
            dict(
                long_name="sorting value based on Folks 1968",
                variable_type="calculated",
                units="-",
            ),
        ),
        porosity=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_porosity"].data,
            dict(
                long_name="porosity based on Takebayashi and Fujita 2014",
                variable_type="calculated",
                units="-",
            ),
        ),
        permeability=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_permeability"].data,
            dict(
                long_name="permeability based on Panda & Lake 1994",
                variable_type="calculated",
                units="m2",
            ),
        ),
        vfractions=(
            ("time", "sed_class", "x", "y"),
            p.dataset_rasterized["gt_vfraction"].data,
            dict(
                long_name="volume fraction of each grain size class per cell",
                variable_type="D3D output",
                units="fraction",
            ),
        ),
        mfractions=(
            ("time", "sed_class", "x", "y"),
            p.dataset_rasterized["mesh2d_dmsedcum"].data,
            dict(
                long_name="mass fraction of each grain size class per cell",
                variable_type="D3D output",
                units="fraction",
            ),
        ),
        archel=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_architectural_elements"].data,
            dict(
                long_name="architectural element",
                variable_type="calculated",
                units="-",
            ),
        ),
        subenv=(
            ("time", "x", "y"),
            p.dataset_rasterized["gt_subenvironment"].data,
            dict(
                long_name="subenvironment",
                variable_type="calculated",
                units="-",
            ),
        ),
    )
    ds = xr.Dataset(data_vars)
    ds["subenv"].attrs["encoding"] = [i.code for i in classifications.SubEnv.__iter__()]
    ds["subenv"].attrs["descriptions"] = [
        i.description for i in classifications.SubEnv.__iter__()
    ]
    ds["subenv"].attrs["labels"] = [i.label for i in classifications.SubEnv.__iter__()]
    ds["archel"].attrs["encoding"] = [i.code for i in classifications.ArchEl.__iter__()]
    ds["archel"].attrs["descriptions"] = [
        i.description for i in classifications.ArchEl.__iter__()
    ]
    ds["archel"].attrs["labels"] = [i.label for i in classifications.ArchEl.__iter__()]

    return ds
