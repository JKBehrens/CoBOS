import argparse
import logging
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dask

dask.config.set({"dataframe.query-planning": True})
import dask.dataframe as dd
from visualization.plot_experiments_result import read_data_to_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=f"%(levelname)-8s: - %(message)s")
    logging.getLogger("mylogger")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the  experiments result")
    parser.add_argument("--case", help="Add case number")

    args = parser.parse_args()

    experiments_path = (
        Path(args.path).expanduser()
        if args.path
        else Path("~/sched_exps/base_sched").expanduser()
    )

    cases = [1, 2, 3, 4, 5, 6, 8]
    if args.case and int(args.case) in cases:
        cases = [int(args.case)]
    else:
        cases = [8]

    methods = [
        "overlapschedule",
        "randomallocation",
        "maxduration",
        "dynamicallocation",
    ]

    df = read_data_to_df(
        experiments_path.glob("*.json"),
        cols={"initial_makespan": -1, "final_makespan": -1, "FAIL": False},
    )
    if df.empty:
        raise FileNotFoundError(
            f"There is no experiment data in the following path {experiments_path.__str__()}"
        )
    else:
        logging.info("The data from the experiments were read successfully.")

    dask_df = dd.from_pandas(df, chunksize=10000)

    # BARPLOT
    df["ms_norm"] = pd.Series(dtype=float)
    fig, ax = plt.subplots()

    selected_methods = ["overlapschedule", "dynamicallocation"]
    data1 = np.empty((0,))
    data2 = np.empty((0,))

    legend: list[str] = []
    data = []
    for case in cases:
        makespans_other: list[list[int]] = []
        for dist_seed in np.unique(np.array(df["dist_seed"])):
            df_one_exp = df[df["case_number"] == case][df["dist_seed"] == dist_seed]
            ms_lb = min(
                df_one_exp["initial_makespan"][df["det_job"] == True][
                    df["method_name"] == "overlapschedule"
                ][df_one_exp["sim_seed"] == df_one_exp["schedule_seed"]]
            )

            df.loc[df_one_exp.index, "ms_norm"] = df_one_exp["final_makespan"] / ms_lb
        for method in methods:
            data.append(
                np.array(
                    list(
                        df["ms_norm"][df["case_number"] == case][
                            df["method_name"] == method
                        ]
                    )[:]
                )
            )

    bp = ax.boxplot(
        data,
        positions=[4, 3, 2, 1],
        patch_artist=True,
        notch=True,
        vert=False,
        labels=["CoBOS(ours)", "RA", "MD", "DA"],
    )
    colors = ["cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue"]
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
    for median in bp["medians"]:
        median.set(color="royalblue", linewidth=1)
    ax.set_xlabel("Normalised makespan")
    ax.set_xlim(0.98, 1.63)
    plt.grid()
    plt.savefig(f"boxplot_case_{cases[0]}_grid.png")
    logging.info(f"Output plot was saved to file boxplot_case_{cases[0]}_grid.png.")
