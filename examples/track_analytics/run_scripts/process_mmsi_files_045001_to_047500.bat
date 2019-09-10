
REM Activate python virtual environment
CALL ..\..\..\venv\Scripts\activate.bat

REM Define group/block
SET mmsi_group=mmsi_files_045001_to_047500

REM Define MMSI data root folder
SET data_root=C:\mmsi_files

REM update Python Path to allow import of top-level (stonesoup) packages
SET PYTHONPATH=%PYTHONPATH%;..\..\..\;

REM Process through SS
ECHO Processing: %mmsi_group%...
python ..\AisTrackingExactEarthIMM_Single_Bare.py --input_path %data_root%\%mmsi_group% --output_path %data_root%\output_for_%mmsi_group%
pause
