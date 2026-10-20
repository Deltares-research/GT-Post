from pathlib import Path

import numpy as np


def read_sedfile(sedfile):
    # read the sed def file and initialize sed type
    # specific density and dry bed density with an empty list
    sedtype = []
    rho_p = []
    rho_db = []
    d50_input = []
    with open(sedfile) as fobj:
        for line in fobj:
            # read lines until SedType is found
            if "SedTyp" in line:
                sedtype.append(str(line.split()[2]))
            if "RhoSol" in line:
                rho_p.append(float(line.split()[2]))
            if "CDryB" in line:
                rho_db.append(float(line.split()[2]))
            if "SedDia" in line:
                if sedtype[-1] == "sand":
                    d50_input.append(float(line.split()[2]))
            if "WSM" in line:
                if sedtype[-1] == "mud":
                    rhof = 1000
                    mu = 0.001
                    g = 9.81
                    svelocity = float(line.split()[2])
                    d50_mud = np.sqrt(18 * mu * svelocity / g / (rho_p[-1] - rhof))
                    d50_input.append(d50_mud)
    return (
        sedtype,
        np.array(rho_p, dtype=np.float32),
        np.array(rho_db, dtype=np.float32),
        np.array(d50_input, dtype=np.float32),
    )


def get_fm_folders(folder: str | Path):
    """
    Get and validate Delft3D-FM output folder structure

    Parameters
    ----------
    folder : str | Path
        Parent folder containing the D3D-FM output structure

    Returns
    -------
    tuple[list[str], list[str], Path]
        The map files, the sed files, and the sedfile.

    Raises
    ------
    TypeError
        Raised if the folder structure is not a complete Delft3D-FM i/o folder.
    """
    folder = Path(folder)
    folder_dflowfm = folder.joinpath("dflowfm")
    folder_output = folder_dflowfm.joinpath("output")
    map_files = [str(f) for f in folder_output.glob("*_map.nc")]
    sed_files = [str(f) for f in folder_output.glob("*_sed.nc")]
    sedfile = [f for f in folder_dflowfm.glob("*.sed")][0]

    condition = (
        (any(map_files) and any(sed_files))
        & folder.is_dir()
        & folder_dflowfm.is_dir()
        & folder_output.is_dir()
    )

    if not condition:
        raise TypeError("This is not a complete Delft3D-FM i/o folder")
    else:
        return map_files, sed_files, sedfile
