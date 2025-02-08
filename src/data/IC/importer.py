import sys, os

if __name__ == "__main__":
    os.chdir("../../")
import numpy as np
import pandas as pd
import warnings
import pickle


def get_aeff_df():
    """
    Data from 2012
    https://icecube.wisc.edu/science/data/PS-3years
    Returns a df with cols 'E_min', 'E_max', 'z_min', 'z_max', 'Aeff', 'E_avg', 'z_avg'

    Cuts off energies above 20 000 GeV and z above 0

    """
    file1 = "../src/data/files/IC86-2012-TabulatedAeff.txt"
    colnames = ["E_min", "E_max", "z_min", "z_max", "Aeff"]

    A = pd.read_csv(
        file1,
        header=None,
        skiprows=1,
        names=colnames,
        dtype=np.float64,
        skipinitialspace=True,
        sep=" ",
    )

    # Constant extrapolation of aeff for z edge
    import warnings

    df_edges = A.query("z_min == -1.00")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_edges["z_max"] = np.round(df_edges["z_max"], 2).replace(-0.99, -1.0)
    df_fixed = pd.concat([df_edges, A])

    df_fixed["E_avg"] = (df_fixed.E_min + df_fixed.E_max) / 2
    df_fixed["z_avg"] = (df_fixed.z_min + df_fixed.z_max) / 2
    df_fixed = df_fixed.query("E_max <= 1e6")  # Remove E_max over 1e6 GeV
    df_fixed = df_fixed.query("E_min <= 1e6")  # Remove E_min over 1e6 GeV
    df_fixed = df_fixed.query("z_max <= 0.1")  # Remove z_max above 0.1

    df = df_fixed.reset_index(drop=True)
    extrapolated_df = extrapolate_aeff_edges(df)
    return extrapolated_df


def extrapolate_aeff_edges(df):
    E_averages = df.E_avg.unique()
    for E in E_averages:
        aeff_at_edge = df[df.E_avg == E].query("z_min <= -0.99").reset_index()
        extrapolated_aeff = (
            2 * aeff_at_edge.loc[1, "Aeff"] - aeff_at_edge.loc[2, "Aeff"]
        )
        constant_aeff = aeff_at_edge.loc[1, "Aeff"]
        aeff_at_edge.loc[0, "Aeff"] = extrapolated_aeff
        df.iloc[aeff_at_edge["index"][0]] = aeff_at_edge.iloc[0]

    return df.sort_values(by=["E_avg", "z_avg"])


def get_aeff_df_dc():
    filename = "../../src/data/files/dc_aeff.csv"
    dc_aeff = pd.read_csv(filename, names=["logE", "Aeff"])
    return dc_aeff


def get_flux_factor():
    file1 = "../../src/data/files/event_rate_conv.csv"
    file2 = "../../src/data/files/event_rate_astro.csv"
    file3 = "../../src/data/files/event_rate_prompt.csv"
    file4 = "../../src/data/files/event_rate_z.csv"
    conv = (
        pd.read_csv(
            file1,
            skiprows=0,
            header=None,
            names=["E", "rate"],
            dtype=np.float64,
            skipinitialspace=True,
            sep=",",
        )
        .sort_values(by="E")
        .reset_index()
    )
    astro = (
        pd.read_csv(
            file2,
            skiprows=0,
            header=None,
            names=["E", "rate"],
            dtype=np.float64,
            skipinitialspace=True,
            sep=",",
        )
        .sort_values(by="E")
        .reset_index()
    )
    prompt = (
        pd.read_csv(
            file3,
            skiprows=0,
            header=None,
            names=["E", "rate"],
            dtype=np.float64,
            skipinitialspace=True,
            sep=",",
        )
        .sort_values(by="E")
        .reset_index()
    )
    z_rate = pd.read_csv(
        file4,
        skiprows=2,
        header=None,
        names=["E", "conv", "E1", "astro", "E2", "prompt"],
        dtype=np.float64,
        skipinitialspace=True,
        sep=",",
    ).sort_values(by="E")
    z_rate["factor"] = (z_rate.conv + z_rate.astro + z_rate.prompt) / z_rate.conv
    df = conv
    df["factor"] = (conv.rate + astro.rate + prompt.rate) / conv.rate
    return df[["E", "factor"]], z_rate[["E", "factor"]]


def get_IC_data():
    """
    Returns the observed event data from IC202 companion paper, in 20 zenith bins and 13 E bins
    """
    df = event_data()
    return df


def flux_parametrization(x, K, b, c, a):
    """
    Source: https://arxiv.org/abs/hep-ph/0203272v2
    """
    return K * np.power(x + b * np.exp(-c * np.sqrt(x)), -a)
    # return K * np.power(x,-a)


def event_data():
    """
    Data from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.102.052009 fig2
    Double-checked and confirmed to sum to 305735
    """
    events = np.array(
        [
            [2336, 2096, 1464, 911, 532, 349, 189, 90, 47, 29, 13, 9, 4],
            [2386, 2769, 2102, 1251, 725, 432, 195, 86, 64, 30, 24, 13, 4],
            [1986, 2838, 2406, 1496, 829, 455, 238, 109, 70, 35, 22, 13, 9],
            [1753, 2728, 2520, 1594, 911, 474, 265, 149, 81, 45, 28, 15, 17],
            [1648, 2676, 2737, 1845, 1015, 606, 327, 190, 104, 45, 39, 11, 12],
            [1589, 2725, 2719, 1956, 1107, 616, 321, 180, 103, 64, 32, 17, 12],
            [1613, 2836, 2856, 2075, 1206, 647, 369, 227, 129, 69, 31, 16, 8],
            [1594, 2918, 2976, 2166, 1306, 722, 359, 220, 111, 82, 41, 28, 8],
            [1665, 3100, 3166, 2232, 1309, 771, 412, 212, 136, 76, 46, 21, 11],
            [1660, 3239, 3467, 2423, 1429, 829, 482, 254, 141, 85, 52, 29, 19],
            [1830, 3286, 3599, 2447, 1533, 930, 535, 260, 150, 80, 51, 39, 20],
            [1960, 3668, 3723, 2615, 1667, 978, 567, 297, 176, 109, 52, 43, 19],
            [2050, 3810, 3838, 2815, 1738, 1031, 561, 310, 175, 113, 73, 34, 34],
            [2055, 3834, 4166, 3027, 1842, 1124, 642, 349, 198, 112, 73, 34, 24],
            [2236, 4067, 4296, 3175, 2008, 1206, 686, 385, 211, 130, 88, 42, 32],
            [2191, 4140, 4403, 3237, 2126, 1278, 737, 449, 240, 138, 87, 49, 27],
            [2298, 4319, 4649, 3533, 2246, 1426, 850, 470, 282, 187, 99, 52, 36],
            [2201, 4442, 4807, 3731, 2528, 1576, 939, 555, 309, 186, 146, 76, 46],
            [2197, 4705, 5412, 4253, 2761, 1735, 1045, 611, 345, 228, 150, 90, 42],
            [2021, 4598, 5356, 4259, 2992, 1860, 1164, 718, 423, 273, 163, 98, 76],
        ]
    )
    z_range = np.linspace(-1, -0.05, 20)
    E_buckets = 500 * 10 ** (np.linspace(0.0, 1.2, 13))

    df = pd.DataFrame(events, columns=E_buckets, index=z_range)
    return df


def DC_event_data():
    """
    Data from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.102.052009 fig2
    Double-checked and confirmed to sum to 305735
    """
    events = np.array(
        [
            [32, 85, 143, 186, 208, 179, 119, 67],
            [66, 101, 162, 223, 271, 276, 230, 147],
            [119, 150, 216, 288, 355, 366, 310, 211],
            [179, 193, 266, 356, 431, 440, 401, 295],
            [248, 240, 291, 353, 440, 468, 445, 332],
            [252, 236, 260, 279, 338, 343, 347, 262],
            [218, 192, 211, 206, 220, 215, 218, 186],
            [182, 197, 206, 206, 171, 137, 128, 113],
        ]
    )
    z_range = np.linspace(-1, -0.25, 8)
    E_buckets = np.logspace(0.75, 1.5, 8)

    df = pd.DataFrame(events, columns=E_buckets, index=z_range)
    return df


def get_flux_df():
    """
    Reads the files files/spl-nu-20-01-000.d and files/spl-nu-20-01-n3650.d which contain the solar min and max atm fluxes. Averages these for each zenith angle range and returns the fluxes for zenith between -1.05 to 0.05, extrapolated to 1e5 GeV.

    Files are from http://www.icrr.u-tokyo.ac.jp/~mhonda/nflx2014/index.html section 2.6
    """
    file1 = "../src/data/files/spl-nu-20-01-000.d"
    file2 = "../src/data/files/spl-nu-20-01-n3650.d"
    colnames = ["GeV", "m_flux", "mbar_flux", "e_flux", "ebar_flux"]

    text_rows = np.append(np.arange(0, 2500, 103), (np.arange(1, 2500, 103)))

    df1 = pd.read_csv(
        file1,
        skiprows=text_rows,
        header=None,
        names=colnames,
        dtype=np.float64,
        sep=" ",
        chunksize=101,
    )
    df2 = pd.read_csv(
        file2,
        skiprows=text_rows,
        header=None,
        names=colnames,
        dtype=np.float64,
        sep=" ",
        chunksize=101,
    )

    df_list = []  # List of the dataframes for each zenith angle bin. df_list[i] is the df for angle theta_range[i]
    for left, right in zip(df1, df2):
        left = left.set_index("GeV")
        right = right.set_index("GeV")
        df_concat = pd.concat([left, right])
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        df_means.reset_index(inplace=True)
        df_list.append(df_means)
    df_list.append(
        df_list[-1]
    )  # Put the fluxes for z=-0.9 to -1 at the end and reinterpret them as z=-1 to -1.1 since flux should be symmetric about z=-1.
    binned_df = z_bins(df_list[9:21])
    df = pd.concat(binned_df)
    df = extrapolate_flux(df)  # Extrapolate flux to 1e5 GeV
    return df


def get_flux_df_DC():
    """
    Reads the files files/spl-nu-20-01-000.d and files/spl-nu-20-01-n3650.d which contain the solar min and max atm fluxes. Averages these for each zenith angle range and returns the fluxes for zenith between -1.05 to 0.05, extrapolated to 1e5 GeV.

    Files are from http://www.icrr.u-tokyo.ac.jp/~mhonda/nflx2014/index.html section 2.6
    """
    file1 = "../../src/data/files/spl-nu-20-01-000.d"
    file2 = "../../src/data/files/spl-nu-20-01-n3650.d"
    colnames = ["GeV", "m_flux", "mbar_flux", "e_flux", "ebar_flux"]

    text_rows = np.append(np.arange(0, 2500, 103), (np.arange(1, 2500, 103)))

    df1 = pd.read_csv(
        file1,
        skiprows=text_rows,
        header=None,
        names=colnames,
        dtype=np.float64,
        sep=" ",
        chunksize=101,
    )
    df2 = pd.read_csv(
        file2,
        skiprows=text_rows,
        header=None,
        names=colnames,
        dtype=np.float64,
        sep=" ",
        chunksize=101,
    )

    df_list = []  # List of the dataframes for each zenith angle bin. df_list[i] is the df for angle theta_range[i]
    for left, right in zip(df1, df2):
        left = left.set_index("GeV")
        right = right.set_index("GeV")
        df_concat = pd.concat([left, right])
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        df_means.reset_index(inplace=True)
        df_list.append(df_means)
    df_list.append(df_list[-1])
    binned_df = z_bins_DC(df_list)
    df = pd.concat(binned_df)
    return df


def z_bins_DC(df_list):
    """
    Doubles the number of dataframes, and puts half of the initial flux in each one.
    Also puts the new z-bin limits in columns
    """
    new_theta_range = np.round(
        np.linspace(1, -1.1, 22), 2
    )  # See comment in get_flux_df
    new_df_list = []
    for i, df in enumerate(df_list):
        df["z_min"] = new_theta_range[i + 1]
        df["z_max"] = new_theta_range[i]

        new_df_list.append(df)
    return new_df_list


def z_bins(df_list):
    """
    Doubles the number of dataframes, and puts half of the initial flux in each one.
    Also puts the new z-bin limits in columns
    """
    new_theta_range = np.round(
        np.linspace(0.1, -1.1, 13), 2
    )  # See comment in get_flux_df
    new_df_list = []
    for i, df in enumerate(df_list):
        df["z_min"] = new_theta_range[i + 1]
        df["z_max"] = new_theta_range[i]

        new_df_list.append(df)
    return new_df_list


def extrapolate_flux(flux_df):
    """
    Extrapolates the fluxes to 1e5 GeV
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warning about covariance not estimated
        for zmin in flux_df.z_min.unique():
            res = fit_flux(flux_df, zmin)
            flux_df = flux_df.append(res)
    return flux_df


def fit_flux(flux_df, zmin):
    from sklearn.linear_model import LinearRegression as LR

    """
    Fits all four fluxes for a given z_min 
    """
    df = flux_df[flux_df.z_min == zmin]

    x = df.query("GeV > 5e3").GeV
    y_m = df.query("GeV > 5e3").m_flux
    y_mbar = df.query("GeV > 5e3").mbar_flux
    y_e = df.query("GeV > 5e3").e_flux
    y_ebar = df.query("GeV > 5e3").ebar_flux

    model_m = LR().fit(np.array(np.log10(x)).reshape(-1, 1), np.log10(y_m))
    model_mbar = LR().fit(np.array(np.log10(x)).reshape(-1, 1), np.log10(y_mbar))
    model_e = LR().fit(np.array(np.log10(x)).reshape(-1, 1), np.log10(y_e))
    model_ebar = LR().fit(np.array(np.log10(x)).reshape(-1, 1), np.log10(y_ebar))
    x_new = np.logspace(np.log10(1e4), np.log10(1e6), 100)
    y_new_m = model_m.predict(np.log10(x_new.reshape(-1, 1)))
    y_new_mbar = model_mbar.predict(np.log10(x_new.reshape(-1, 1)))
    y_new_e = model_e.predict(np.log10(x_new.reshape(-1, 1)))
    y_new_ebar = model_ebar.predict(np.log10(x_new.reshape(-1, 1)))

    return pd.DataFrame(
        np.transpose(
            [
                x_new,
                10**y_new_m,
                10**y_new_mbar,
                10**y_new_e,
                10**y_new_ebar,
                zmin * np.ones(100),
                (zmin + 0.1) * np.ones(100),
            ]
        ),
        columns=["GeV", "m_flux", "mbar_flux", "e_flux", "ebar_flux", "z_min", "z_max"],
    )


if __name__ == "__main__":
    get_flux_factor()
