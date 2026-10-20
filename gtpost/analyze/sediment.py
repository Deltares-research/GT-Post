import numba
import numpy as np


def calculate_fraction(rho_db: np.array, dmsedcum_influx: np.array) -> np.array:
    """
    Calculate the volumetric fraction of sediment.
    This function computes the volumetric fraction of sediment by dividing the cumulative
    sediment mass by the dry bed density and normalizing the result.

    Parameters
    ----------
    rho_db : np.array
        Array of dry bed densities.
    dmsedcum_influx : np.array
        Array of cumulative sediment mass.

    Returns
    -------
    np.array
        Array of volumetric fractions of sediment.
    """
    vfraction = np.zeros_like(dmsedcum_influx, dtype=np.float32)

    # derive the volumetric sed flux by dividing by dry bed density
    dvsedcum = dmsedcum_influx / rho_db[np.newaxis, :, np.newaxis]
    sumsedvcum = np.sum(dvsedcum, axis=1)

    vfraction = np.divide(
        dvsedcum,
        sumsedvcum[
            :,
            np.newaxis,
            :,
        ],
        where=sumsedvcum[
            :,
            np.newaxis,
            :,
        ]
        != 0,
    )
    vfraction[np.isnan(vfraction)] = 0
    return vfraction


def calculate_sand_fraction(sed_types: list, vfraction: np.array) -> np.array:
    """
    Calculate the sand fraction from the given sediment types and volume fractions.

    Parameters
    ----------
    sed_types : list
        A list of sediment types.
    vfraction : np.array
        A numpy array representing the volume fractions of different sediment types.
    Returns
    -------
    np.array
        A numpy array containing the summed volume fractions of sand, with mud fractions set to zero.
    """
    sand_indices = [i for i, stype in enumerate(sed_types) if stype == "sand"]
    sand_fraction = np.sum(vfraction[:, sand_indices, :], axis=1).astype(np.float32)
    return sand_fraction


def calculate_sorting(diameters: np.ndarray, percentage2cal: list) -> np.ndarray:
    """
    Calculate the sorting parameter of sediment grain size distribution using Folks
    (1968) method.

    Parameters
    ----------
    diameters : np.ndarray
        A multi-dimensional array of sediment grain diameters.
    percentage2cal : list
        A list of percentage values used to calculate the sorting parameter.
        It should contain the values 10, 16, 84, and 90.

    Returns
    -------
    np.ndarray
        The calculated sorting parameter for the given sediment grain size distribution.

    Notes
    -----
    The sorting parameter is calculated using the formula:
    sorting = (phi_84 - phi_16) / 4 + (phi_90 - phi_10) / 6.6
    where phi_x is the grain size at the xth percentile in grain size phi.
    """
    index10, index16, index84, index90 = (
        percentage2cal.index(p) for p in (10, 16, 84, 90)
    )
    phi = -np.log2(diameters)

    # Compute Folk's (1968) sorting parameter
    return -(
        (phi[..., index84] - phi[..., index16]) / 4
        + (phi[..., index90] - phi[..., index10]) / 6.6
    )


@numba.njit
def calculate_distribution(fraction_data, d50input):
    # Select only relevant xdata
    xdata = d50input[fraction_data > 0]
    xphi = -np.log2(1000 * xdata)
    if len(xdata) > 0:
        xdiff = np.diff(xphi)
        xphi = np.hstack(
            (
                np.array([xphi[0] - 0.5], dtype=np.float32),
                xphi[:-1] + (0.5 * xdiff),
                np.array([xphi[-1] + 0.5], dtype=np.float32),
            )
        )
        # cdf value
        cdf = np.cumsum(fraction_data[fraction_data > 0])
        cdf = np.hstack((np.array([0], dtype=np.float32), cdf))
        # linear fit
        xnew = np.arange(-1, 8, 0.1)
        ynew = np.interp(xnew, xphi, cdf)
        fraction_data = np.diff(ynew)
        xnewd50 = (2 ** (-(xnew[1:]))) / 1000

        if np.abs(np.sum(fraction_data) - 1) < 0.1:
            # weighted distribution
            minval = np.min(fraction_data[(fraction_data > 0.001).nonzero()])
            amount = np.zeros(len(xnewd50), dtype=np.int32)

            for i, d50 in enumerate(xnewd50):
                amount[i] = np.int32(np.round(fraction_data[i] / minval))

            total_samples = np.sum(amount)
            cumsum_samples = np.cumsum(amount)
            d50distributed = np.zeros(total_samples, dtype=np.float32)

            for i, d50 in enumerate(xnewd50[:-2]):
                d50distributed[cumsum_samples[i] : cumsum_samples[i + 1]] = np.full(
                    cumsum_samples[i + 1] - cumsum_samples[i], d50
                )

            phidistributed = -np.log2(1000 * d50distributed)
            phidistributed = phidistributed[np.isfinite(phidistributed)]

            # Standard deviation of the sed distribution
            averaged = np.nanmean(d50distributed)
            averagep = np.nanmean(phidistributed)
            distr_sdev = np.nanstd(phidistributed)
            # Porosity based on empirical fit by Takebayashi & Fujita (2014)
            poros = 0.38 * (
                (3.7632 * (distr_sdev**-0.7552))
                / (1 + (3.7632 * (distr_sdev**-0.7552)))
            )
            # Coefficient of variation of sed distribution
            c_dp = distr_sdev / averagep
            # Skewness of the sediment distribution
            skw = (
                (1 / len(d50distributed)) * np.sum((d50distributed - averaged) ** 3)
            ) / (
                ((1 / len(d50distributed)) * np.sum((d50distributed - averaged) ** 2))
                ** (1.5)
            )
            # Turtuosity based on porosity after Ahmadi et al. (2011)
            turt = np.sqrt(
                ((2 * poros) / (3 * (1 - 1.209 * ((1 - poros) ** (2 / 3))))) + (1 / 3)
            )
            firstel = (averaged**2 * poros**3) / (72 * turt * ((1 - poros) ** 2))
            secel = ((skw * c_dp + 3 * c_dp**2 + 1) ** 2) / ((1 + c_dp**2) ** 2)
            perm = firstel * secel
        else:
            # This occurs if the sum of fractions is not equal to 1 with a tolerance of 0.01
            perm = np.nan
            poros = np.nan
    else:
        perm = np.nan
        poros = np.nan
        xnew = np.arange(-1, 8, 0.1)
        ynew = np.full_like(xnew, np.nan)
    return (ynew, xnew, poros, perm)


@numba.njit(parallel=True)
def calculate_diameter(d50input, percentage2cal, vfraction):
    nt, _, nc = vfraction.shape
    diameters = np.zeros((nt, nc, len(percentage2cal)), dtype=np.float32)
    porosity = np.zeros((nt, nc), dtype=np.float32)
    permeability = np.zeros_like(porosity)
    diameters[0] = np.nan
    porosity[0] = np.nan
    permeability[0] = np.nan
    for it in numba.prange(nt):
        for ic in numba.prange(nc):
            fraction_data = vfraction[it, :, ic]
            # return the phi value needs to be changed back to meters
            (
                cdf,
                xvalue,
                porosity[it, ic],
                permeability[it, ic],
            ) = calculate_distribution(fraction_data, d50input)
            # If poro/perm returns nan, find the last known value from previous time steps
            # This is then still the poros/perm at the surface
            # dmsedcum_final += 1
            if np.isnan(porosity[it, ic]):
                lastknown_idxs = (porosity[:, ic] > 0.0).nonzero()[-1]
                if len(lastknown_idxs) != 0:
                    porosity[it, ic] = porosity[lastknown_idxs[-1], ic]
                    permeability[it, ic] = permeability[lastknown_idxs[-1], ic]
            for ipercen in range(len(percentage2cal)):
                diameters[it, ic, ipercen] = 2 ** (
                    -xvalue[
                        np.argmin(np.abs(cdf - (1 - (0.01 * percentage2cal[ipercen]))))
                    ]
                )
                # set diameter to zero if there is no deposition
                # because in the models, we don't have 2000mm,
                # if it is equals to 2000, then there it assigns to the value
                # below it
                if diameters[it, ic, ipercen] == 2:
                    if it > 0:
                        diameters[it, ic, ipercen] = diameters[it - 1, ic, ipercen]
                    else:
                        diameters[it, ic, ipercen] = 0
                else:
                    pass
    return (diameters, porosity, permeability)
