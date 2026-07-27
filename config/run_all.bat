@echo off
set RIVER=Saint-Jean.yaml

echo.
echo ==========================================================
echo DEBUT DU TRAITEMENT POUR : %RIVER%
echo ==========================================================
echo.

set STEP=cross_section_points
python -m river_profile_lidar.cross_section_points river=%RIVER%
if errorlevel 1 goto error

set STEP=trapezoid_bathymetry
python -m river_profile_lidar.trapezoid_bathymetry river=%RIVER%
if errorlevel 1 goto error

set STEP=hab
python -m river_profile_lidar.hab river=%RIVER%
if errorlevel 1 goto error

set STEP=water_speed
python -m river_profile_lidar.water_speed river=%RIVER%
if errorlevel 1 goto error

set STEP=d84
python -m river_profile_lidar.d84 river=%RIVER%
if errorlevel 1 goto error

set STEP=iqhp_surfacique
python -m river_profile_lidar.iqhp_surfacique river=%RIVER%
if errorlevel 1 goto error

set STEP=iqhp_linear
python -m river_profile_lidar.iqhp_linear river=%RIVER%
if errorlevel 1 goto error

set STEP=iqhp_transects
python -m river_profile_lidar.iqhp_transects river=%RIVER%
if errorlevel 1 goto error

set STEP=uphp_table
python -m river_profile_lidar.uphp_table river=%RIVER%
if errorlevel 1 goto error

echo.
echo ==========================================================
echo SUCCES : Toutes les étapes sont terminées !
echo ==========================================================
pause
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo ERREUR DETECTEE lors de l'etape : %STEP%
echo Le script s'est arrete.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pause
exit /b 1
