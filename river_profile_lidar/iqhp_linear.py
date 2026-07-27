from pathlib import Path
import geopandas
import pickle
import hydra
from omegaconf import DictConfig
import logging

from river_profile_lidar.bathymetry.hab.utils import calculate_da
from habitat_model import HabitatModel  # noqa: F401


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing IQH for: {cfg.river.name}")
    transect_path = Path(cfg.river.linear_transect_path)
    saving_path = Path(cfg.river.linear_iqh_output_path)
    linear_df = geopandas.read_file(transect_path)
    linear_df["Pente"] = linear_df["Pente"].astype(float)

    min_non_zero_slope = linear_df[linear_df["Pente"] > 0]["Pente"].min()
    zero_or_nan_slope_idx = (linear_df["Pente"].isna()) | (linear_df["Pente"] <= 0)
    linear_df.loc[zero_or_nan_slope_idx, "Pente"] = min_non_zero_slope

    linear_df["DA"] = linear_df.apply(
        lambda row: calculate_da(row["Q_IMG"], row["Pente"], row["WIDTH"]),
        axis=1,
    )
    linear_df["Vi"] = linear_df["Q_IMG"] / linear_df["WIDTH"]

    with open("habitat_model.pkl", "rb") as f:
        habitat_model = pickle.load(f)

    linear_df["iqhp_1d"] = linear_df["D84"].apply(
        habitat_model.get_hist_1d_estimation_from_value
    )
    linear_df["iqhp_2d"] = linear_df.apply(
        lambda row: habitat_model.get_hist_2d_estimation_from_value(
            row["DA"], row["Vi"]
        ),
        axis=1,
    )
    linear_df["iqhp"] = linear_df["iqhp_1d"] * linear_df["iqhp_2d"]
    linear_df["surface"] = linear_df["WIDTH"] * 5
    linear_df["UPHP"] = linear_df["iqhp"] * linear_df["surface"]
    schema = geopandas.io.file.infer_schema(linear_df)
    schema["properties"]["FACC_max"] = "float:20.5"
    schema["properties"]["FACC_COR"] = "float:20.5"
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    linear_df.to_file(
        saving_path, driver="ESRI Shapefile", engine="fiona", schema=schema
    )


if __name__ == "__main__":
    run()
