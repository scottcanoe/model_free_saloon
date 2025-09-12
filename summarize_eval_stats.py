"""
Go down to __main__ and specify the experiment names.

Then run python summarize_eval_stats.py
"""

from pathlib import Path
import pandas as pd

def main(experiments, save_filename):
    summary_df = pd.DataFrame()
    for exp in experiments:
        path = Path(".", "results", exp, "eval_stats.csv")
        df = pd.read_csv(path)
        overall_accuracy = get_accuracy(df)
        rotation_error = get_rotation_error(df)
        avg_num_steps_wo_timeout = get_avg_num_steps_wo_timeout(df)
        num_timed_out = get_num_timed_out_episodes(df)
        num_total_episodes = get_num_total_episodes(df)
        total_run_time = get_total_run_time(df)

        summary_df = pd.concat([summary_df, pd.DataFrame({
            "experiment": [exp],
            "overall_accuracy": [overall_accuracy],
            "rotation_error": [rotation_error],
            "avg_num_steps_wo_timeout": [avg_num_steps_wo_timeout],
            "num_timed_out": [num_timed_out],
            "num_total_episodes": [num_total_episodes],
            "total_run_time": [total_run_time]
        })], ignore_index=True)
    summary_df.to_csv(f"./results/{save_filename}", index=False)

def get_accuracy(df):
    return len(df[df["primary_performance"].isin(["correct", "correct_mlh"])]) / len(df)

def get_rotation_error(df):
    # average rotation error for correct episodes
    subset = df[df["primary_performance"].isin(["correct", "correct_mlh"])]
    return subset["rotation_error"].mean()

def get_num_timed_out_episodes(df):
    return len(df[df["individual_ts_performance"] == "time_out"])

def get_num_total_episodes(df):
    return len(df)

def get_avg_num_steps_wo_timeout(df):
    subset = df[df["individual_ts_performance"] != "time_out"]
    return subset["num_steps"].mean()

def get_total_run_time(df):
    return df["time"].sum()

if __name__ == "__main__":
    experiments = [
        "bio_baseline",
        "bio_uniform_saliency",
        "bio_bio_saliency",
        "standard_baseline",
        "standard_uniform_saliency",
        "standard_bio_saliency",
    ]
    save_filename = "eval_stats_1rot.csv" # saved to results folder
    main(experiments, save_filename)

    exp_6rots = [
        "bio_baseline_6rot",
        "bio_uniform_saliency_6rot",
        "bio_bio_saliency_6rot",
        # "bio_spectral_residual_6rot",
        # "bio_minimum_barrier_6rot",
        "standard_baseline_6rot",
        "standard_uniform_saliency_6rot",
        "standard_bio_saliency_6rot",
        # "standard_spectral_residual_6rot",
        # "standard_minimum_barrier_6rot",
    ]
    save_filename = "summary_eval_stats_6rots.csv"
    main(exp_6rots, save_filename)

    exp_5randrots = [
        "bio_baseline_5randrot",
        "bio_uniform_saliency_5randrot",
        "bio_bio_saliency_5randrot",
        "standard_baseline_5randrot",
        "standard_uniform_saliency_5randrot",
        "standard_bio_saliency_5randrot",
    ]
    save_filename = "summary_eval_stats_5randrots.csv"
    main(exp_5randrots, save_filename)
