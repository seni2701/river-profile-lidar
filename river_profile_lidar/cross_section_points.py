import geopandas
import hydra
from omegaconf import DictConfig
import logging

from river_profile_lidar.bathymetry.trapezoid.cross_section_points.create_cross_section_points import (
    create_cross_section_points,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing Cross Section Points for: {cfg.river.name}")
    transect_path = cfg.river.transect_path
    saving_folder_path = cfg.river.cross_section_output_path

    data = geopandas.read_file(transect_path)
    for include_middle_points in [True, False]:
        create_cross_section_points(data, saving_folder_path, include_middle_points)
    log.info(f"Cross Section Points for: {cfg.river.name} is Done")


if __name__ == "__main__":
    run()
