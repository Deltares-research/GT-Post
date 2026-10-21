import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Self

import dfm_tools as dfmt
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu

from gtpost.analyze import classifications, layering, sediment, statistics, surface
from gtpost.experimental import detect_archels
from gtpost.io import d3d_fm, export
from gtpost.method_validation import run_method

default_settings_file = (
    Path(__file__).parents[1].joinpath("config/default_settings.ini")
)


class ModelResult:
    def __init__(
        self,
        dataset: xr.Dataset,
        sedfile: str | Path,
        modelname: str = "Unnamed Delft3D Geotool modelresult",
        grid_resolution: int = 50,
        settings_file: str | Path = default_settings_file,
    ):
        """
        Delft3D ModelResult class. Used for postprocessing Delft3D GeoTool model
        results.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset of the Delft3D trim file
        sedfile : Pathlike
            Path to the D3D .sed file
        modelname : str, optional
            Name of the model result, by default "Unnamed Delft3D Geotool modelresult"
        grid_resolution : int, optional
            Desired grid resolution of the model, by default 50
        settings_file : Pathlike
            Path to settings .ini file
        """
        self.modelname = modelname
        self.config = ConfigParser()
        # self.config.read(settings_file)
        self.dataset = dataset  # .isel(time=slice(0, 80))  # time slice for testing
        self._dataset_rasterized = None
        self.grid_resolution = grid_resolution

        (
            self.sed_types,
            self.rho_p,
            self.rho_db,
            self.d50_input,
        ) = d3d_fm.read_sedfile(sedfile)

    def __repr__(self):
        return (
            f"< {self.modelname} >\n---------------------------\n"
            + f"Completed timesteps   : {self.timestep}"
        )

    @classmethod
    def from_folder(
        cls,
        delft3d_folder: str | Path,
        settings_file: str | Path = default_settings_file,
        used_data_vars=[
            "mesh2d_waterdepth",
            "mesh2d_uorb",
            "mesh2d_subsupl",
            "mesh2d_hwav",
            "mesh2d_mor_bl",
            "mesh2d_ucx",
            "mesh2d_ucy",
            "mesh2d_dmsedcum",
        ],
    ) -> Self:
        """
        Constructor for ModelResult class from Delft3D i/o folder

        Parameters
        ----------
        delft3d_folder : str | Path
            Location of the D3D i/o folder

        Returns
        -------
        ModelResult
            Instance of ModelResult. Contains the dataset attribute, which contains all
            variables in the trim.nc file.

        Raises
        ------
        TypeError if Delft3D folder is invalid/incomplete
        """
        folder = Path(delft3d_folder)
        map_files, sed_files, sedfile = d3d_fm.get_fm_folders(folder)
        modelname = folder.stem + f" - {sedfile.stem}"

        # Open partitioned datasets for map and sediment files
        ds_map = dfmt.open_partitioned_dataset(map_files)
        ds_sed = dfmt.open_partitioned_dataset(sed_files, decode_times=False)

        # Temporary workaround for decoding the time coordinate in the sed dataset
        time_attrs = ds_sed.time.attrs
        ds_sed = ds_sed.assign_coords(
            time=pd.to_datetime(
                np.hstack([[0], ds_sed.time.values[1:]]),
                unit="s",
                origin="2013-12-01 00:00:00",
            )
        )
        ds_sed.time.assign_attrs(time_attrs)

        # Combine relevant data_vars from map and sediment files
        vars_in_map = [var for var in used_data_vars if var in ds_map.data_vars]
        vars_in_sed = [var for var in used_data_vars if var in ds_sed.data_vars]
        combined_ds = xu.merge([ds_map[vars_in_map], ds_sed[vars_in_sed]])

        return cls(
            dataset=combined_ds,
            sedfile=sedfile,
            modelname=modelname,
            settings_file=settings_file,
        )

    @property
    def timestep(self):
        return len(self.dataset.time)

    @property
    def dataset_rasterized(self):
        #TODO: what if new variables were added to the dataset that are not in the rasterized version?
        if self._dataset_rasterized is None:
            self._dataset_rasterized = dfmt.rasterize_ugrid(
                self.dataset, resolution=self.grid_resolution
            ).transpose(..., "x", "y")
        return self._dataset_rasterized

    @run_method(skip_if_present=["gt_subsidence"], required=["mesh2d_subsupl"])
    def compute_subsidence(self):
        """
        Compute subsidence for postprocessing. Adds computed variable "gt_subsidence" to
        the dataset.

        Adds `gt_subsidence` to the dataset.

        """
        subsidence = np.append(
            self.dataset["mesh2d_subsupl"].values,
            self.dataset["mesh2d_subsupl"].values[-1, :][np.newaxis, ...],
            axis=0,
        )
        subsidence = np.diff(subsidence, axis=0)

        # Add results to dataset
        self.dataset["gt_subsidence"] = (("time", "mesh2d_nFaces"), subsidence)

    @run_method(
        skip_if_present=["gt_depositheight"],
        required=["mesh2d_mor_bl"],
        computed=["compute_subsidence"],
    )
    def compute_deposit_height(self):
        """
        Compute deposit height for postprocessing.

        Adds `gt_depositheight` to the dataset.

        """
        deposit_height = np.zeros_like(self.dataset["mesh2d_waterdepth"])
        deposit_height[1:, :] = (
            self.dataset["mesh2d_mor_bl"].values[1:, :]
            - self.dataset["mesh2d_mor_bl"].values[:-1, :]
        ) + self.dataset["gt_subsidence"].values[1:, :]
        deposit_height[np.abs(deposit_height) < 1e-5] = 0

        # Add results to dataset
        self.dataset["gt_depositheight"] = (("time", "mesh2d_nFaces"), deposit_height)

    @run_method(
        skip_if_present=["gt_preserved_thickness", "gt_deposition_age"],
        required=["mesh2d_mor_bl"],
        computed=["compute_deposit_height"],
    )
    def compute_layering(self):
        """
        Compute preservation potential and deposition age.

        Adds `gt_preserved_thickness` and `gt_deposition_age` to the dataset.

        """
        zcor = -self.dataset["mesh2d_mor_bl"].values
        preserved_thickness, deposition_age = layering.preservation(
            zcor,
            self.dataset["mesh2d_mor_bl"].values,
            self.dataset["gt_depositheight"].values,
        )

        # Add results to dataset
        self.dataset["gt_preserved_thickness"] = (
            ("time", "mesh2d_nFaces"),
            preserved_thickness,
        )
        self.dataset["gt_deposition_age"] = (("time", "mesh2d_nFaces"), deposition_age)

    @run_method(
        skip_if_present=["gt_vfraction"],
        required=["mesh2d_dmsedcum"],
    )
    def compute_volume_fractions(self):
        """
        Compute volume fractions of different sediment types.

        Adds `gt_vfraction` to the dataset.

        """
        # Only the incoming sediment flux determines the composition of potential
        # deposits, so remove fluxes of sediment classes that are negative.
        dmsedcum_influx = self.dataset["mesh2d_dmsedcum"].values.astype(np.float32)
        dmsedcum_influx[dmsedcum_influx < 0] = 0
        vfraction = sediment.calculate_fraction(self.rho_db, dmsedcum_influx)

        # Add results to dataset
        self.dataset["gt_vfraction"] = (("time", "nSedTot", "mesh2d_nFaces"), vfraction)

    @run_method(
        skip_if_present=["gt_sandfraction"],
        computed=["compute_volume_fractions"],
    )
    def compute_sand_fractions(self):
        """
        Compute sand fractions from volume fractions.

        Adds `gt_sandfraction` to the dataset.

        """
        sandfraction = sediment.calculate_sand_fraction(
            self.sed_types, self.dataset["gt_vfraction"].values
        )

        # Add results to dataset
        self.dataset["gt_sandfraction"] = (("time", "mesh2d_nFaces"), sandfraction)

    @run_method(
        skip_if_present=["gt_diameters", "gt_porosity", "gt_permeability"],
        computed=["compute_volume_fractions"],
    )
    def compute_sediment_parameters(self):
        """
        Compute sediment parameters such as mean diameter, porosity and permeability.

        Adds `gt_diameters`, `gt_porosity`, `gt_permeability` and `gt_d50` to the dataset.

        """
        diameters, porosity, permeability = sediment.calculate_diameter(
            np.asarray(self.d50_input, dtype=np.float32),
            np.array([10, 16, 50, 84, 90], dtype=np.float32),
            self.dataset["gt_vfraction"].values,
        )

        # Add results to dataset
        self.dataset["gt_diameters"] = (("time", "mesh2d_nFaces", "dnum"), diameters)
        self.dataset["gt_porosity"] = (("time", "mesh2d_nFaces"), porosity)
        self.dataset["gt_permeability"] = (("time", "mesh2d_nFaces"), permeability)
        self.dataset["gt_d50"] = (("time", "mesh2d_nFaces"), diameters[:, :, 2])

    # def detect_channel_network(self):
    #     """
    #     Detect channels and derived parameters.

    #     Returns
    #     -------
    #     None (attributes channels, channel_skeleton, channel_width and channel_depth
    #     are created)
    #     """
    #     (
    #         self.channels,
    #         self.channel_skeleton,
    #         self.channel_width,
    #         self.channel_depth,
    #     ) = surface.detect_channel_network(
    #         self.dataset,
    #         self.subenvironment,
    #         self.dx,
    #         self.config,
    #     )

    @run_method(
        skip_if_present=[],  # TODO
        required=[
            "mesh2d_uorb",
            "mesh2d_hwav",
            "mesh2d_mor_bl",
            "mesh2d_ucx",
            "mesh2d_ucy",
        ],  # TODO
        computed=[],  # TODO
    )
    def detect_architectural_elements(self):
        """
        Detect architectural elements on the delta from previously determined
        depositional environments, channel data and bed level change data. Attribute
        "architectural_elements" is added and contains the following elements:

        1 - Delta top subaerial (overbank deposits/background sedimentation)
        2 - Delta top subaqeous (submerged overbank deposits/background sedimentation)
        3 - Active channel (bed deposits)
        4 - Mouthbars (mouthbar deposits in vicinity of channels and delta front)
        5 - Delta front (background sedimentation around delta edge)
        6 - Prodelta (marine background sedimentation)

        Returns
        -------
        None (attribute architectural_elements is created)
        """
        self.architectural_elements, self.ae_mapping = detect_archels.detect(self)
        self.subenvironment = np.vectorize(classifications.ae_to_subenvironment.get)(
            self.architectural_elements, 0
        ).astype(np.int32)

    def statistics_summary(self):
        """
        Get statistics from calculated model results
        """
        (
            self.d50_distributions,
            self.d50_distribution_weights,
        ) = statistics.get_diameter_distributions(
            self.architectural_elements,
            self.preserved_thickness,
            self.d50,
            10,
        )

        (
            delta_volume,
            self.archel_volumes,
            archel_d50s,
            archel_fractions,
            archel_sorting,
        ) = statistics.get_stats_per_archel(
            self.architectural_elements,
            self.preserved_thickness,
            self.d50,
            self.sandfraction,
            self.sorting,
            10,
        )

        self.delta_stats = {"delta_volume": delta_volume}
        for key, ae_id in self.ae_mapping.items():
            self.delta_stats[f"ae_{classifications.ArchEl(ae_id).name}_volume"] = (
                self.archel_volumes[key]
            )
            self.delta_stats[f"ae_{classifications.ArchEl(ae_id).name}_d50"] = (
                archel_d50s[key]
            )
            self.delta_stats[
                f"ae_{classifications.ArchEl(ae_id).name}_sandfraction"
            ] = archel_fractions[key]
            self.delta_stats[f"ae_{classifications.ArchEl(ae_id).name}_sorting"] = (
                classifications.sorting_classifier(archel_sorting[key])[0]
            )

    def process(self):
        """
        Run all process methods:

        - Compute sediment parameters
        """
        pass

    def postprocess(self):
        """
        Run all postprocess methods in order:

        - Compute sediment parameters
        - Detect subenvironments
        - Detect architectural elements
        """

        # self.compute_deposit_height()
        # self.compute_layering()
        # self.compute_sediment_parameters()
        # At this point we switch to the rasterized dataset for further processing

        self.detect_architectural_elements()

    def export_sediment_and_object_data(self, out_file: str | Path):
        """
        Export sediment and object data to a NetCDF file.

        This method creates a dataset containing sediment and object data
        and exports it to a specified NetCDF file.

        Parameters
        ----------
        out_file : str or Path
            The path to the output NetCDF file.

        Returns
        -------
        None

        Notes
        -----
        The dataset is created using the `export.create_sed_and_obj_dataset` method
        and saved to a NetCDF file using the `h5netcdf` engine with specified encodings.
        """
        ds = export.create_sed_and_obj_dataset(self)
        ds.to_netcdf(out_file, engine="h5netcdf", encoding=export.ENCODINGS)

    def append_input_ini_file(self, input_ini_file: str | Path, out_file: str | Path):
        """
        Appends sections and settings for postprocessing settings, subenviroment
        definitions and architectural element definitions to the input.ini configuration
        file and writes the updated configuration to an output file.

        Parameters
        ----------
        input_ini_file : str or Path
            The path to the input INI file to be read and modified.
        out_file : str or Path
            The path to the output file where the modified configuration will be written.
        """
        shutil.copyfile(input_ini_file, out_file)
        config = ConfigParser()
        config.read(out_file)

        # Define new sections
        config.add_section("postprocessing_parameters")
        config.add_section("subenvironments")
        config.add_section("architectural_elements")

        # Fill new sections
        for key, value in self.config.items("classification"):
            config.set("postprocessing_parameters", key, value)
        for code, name in zip(
            classifications.subenvironment_codes, classifications.subenvironment_names
        ):
            config.set("subenvironments", str(code), name)
        for code, name in zip(
            classifications.archel_codes, classifications.archel_names
        ):
            config.set("architectural_elements", str(code), name)
        with open(out_file, "w") as configfile:
            config.write(configfile)
