import argparse
import logging
import os
from pathlib import Path
import re
import json
from typing import Any
from pydantic import BaseModel
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
from visualization.plot_experiments_result import read_data_to_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,
                        help='write path with experiments result')
    args = parser.parse_args()

    experiments_path = Path(args.path).expanduser()

    methods = ['overlapschedule', 'randomallocation', 'maxduration', 'dynamicallocation']

    df = read_data_to_df(experiments_path.glob("*.json"),
                         cols={"initial_makespan": -1, "final_makespan": -1, "FAIL": False})
    if df.empty:
        raise FileNotFoundError(f'There is no experiment data in the following path {experiments_path.__str__()}')

    dask_df = dd.from_pandas(df, chunksize=10000)

    # BARPLOT
    df['ms_norm'] = pd.Series(dtype=float)
    fig, ax = plt.subplots()

    selected_methods = ['overlapschedule', 'dynamicallocation']
    data1 = np.empty((0,))
    data2 = np.empty((0,))

    legend: list[str] = []
    for case in [8]:
        makespans_other: list[list[int]] = []
        for dist_seed in np.unique(np.array(df["dist_seed"])):
            df_one_exp = df[df["case_number"] == case][df["dist_seed"] == dist_seed]
            ms_lb = min(df_one_exp["initial_makespan"][df["det_job"] == True][df["method_name"] == "overlapschedule"][
                            df_one_exp["sim_seed"] == df_one_exp["schedule_seed"]])

            df.loc[df_one_exp.index, "ms_norm"] = df_one_exp["final_makespan"] / ms_lb
        data = []
        for method in methods:
            data.append(np.array(list(df["ms_norm"][df["case_number"] == case][df["method_name"] == method])[:]))

    bp = ax.boxplot(data, positions=[4, 3, 2, 1], patch_artist=True, notch=True, vert=False,
                    labels=['CoBOS(ours)', 'RA', 'MD', 'DA'])
    colors = ['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue']
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    for median in bp['medians']:
        median.set(color='royalblue',
                   linewidth=1)
    ax.set_xlabel('Normalised makespan')
    ax.set_xlim(0.98, 1.63)
    plt.grid()
    plt.savefig('boxplot_case_8_grid.png')

