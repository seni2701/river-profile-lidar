# river-profile-lidar

1. Create a new conda environnement

```
conda create -n river-profile-lidar python=3.12.4
conda activate river-profile-lidar
```
2. Install requirements
If only running the code
```
pip install -r requirements.txt
```
If working as a dev to commit changes 

```
pip install -r requirements-dev.txt
pre-commit install
```
3. Run the code

Create a river config in the config/river folder. There's is an exemple for Dartmouth.
Make sure that the folder specifiy in the config files exists
```
python -m river_profile_lidar.crop_image river=dartmouth
python -m river_profile_lidar.cross_section_points river=dartmouth
python -m river_profile_lidar.trapezoid_bathymetry river=dartmouth
python -m river_profile_lidar.hab river=dartmouth
python -m river_profile_lidar.water_speed river=dartmouth
python -m river_profile_lidar.d84 river=dartmouth
python -m river_profile_lidar.iqhp_surfacique river=dartmouth
python -m river_profile_lidar.iqhp_linear river=dartmouth
python -m river_profile_lidar.iqhp_transects river=dartmouth
python -m river_profile_lidar.uphp_table river=dartmouth
```
