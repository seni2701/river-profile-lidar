from pathlib import Path
import geopandas
import pandas
import logging
import hydra
from omegaconf import DictConfig


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing UPHP Transects for: {cfg.river.name}")
    transects_iqhp_path = cfg.river.transect_iqh_output_path
    linear_iqhp_path = cfg.river.linear_iqh_output_path
    save_path = Path(cfg.river.uphp_table_path)
    transects_iqhp = geopandas.read_file(transects_iqhp_path)
    linear_iqhp = geopandas.read_file(linear_iqhp_path)
    linear_iqhp = linear_iqhp[linear_iqhp["pred"] == 1]
    transects_surface_col_name = (
        "surface" if "surface" in transects_iqhp.columns else "Surface"
    )
    linear_surface_col_name = (
        "surface" if "surface" in linear_iqhp.columns else "Surface"
    )
    UP_df = pandas.DataFrame(columns=["Groupe", "UPHP", "Superficie", "IQHP_Moyen"])
    UP_df["Groupe"] = ["Surfacique", "Lineaire", "Total"]
    UP_df["UPHP"] = [
        transects_iqhp["UPHP"].sum(),
        linear_iqhp["UPHP"].sum(),
        transects_iqhp["UPHP"].sum() + linear_iqhp["UPHP"].sum(),
    ]
    UP_df["Superficie"] = [
        transects_iqhp[transects_surface_col_name].sum(),
        linear_iqhp[linear_surface_col_name].sum(),
        transects_iqhp[transects_surface_col_name].sum()
        + linear_iqhp[linear_surface_col_name].sum(),
    ]
    UP_df["IQHP_Moyen"] = UP_df["UPHP"] / UP_df["Superficie"]
    UP_df["Superficie"] = UP_df["Superficie"].round(0)
    UP_df["UPHP"] = UP_df["UPHP"].round(0)
    UP_df["IQHP_Moyen"] = UP_df["IQHP_Moyen"].round(2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    UP_df.to_csv(save_path)


if __name__ == "__main__":
    run()
