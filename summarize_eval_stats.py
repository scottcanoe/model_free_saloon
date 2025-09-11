from pathlib import Path
import pandas as pd

def main(experiments):
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
    summary_df.to_csv("./results/summarize_eval_stats.csv", index=False)

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
    main(experiments)
