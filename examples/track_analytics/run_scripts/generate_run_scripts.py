import textwrap

group_by = 2_500
mmsi_file_count = 115598

for i in range(0, mmsi_file_count, group_by):
    mmsi_files_handle = f'mmsi_files_{i + 1:06}_to_{i + group_by:06}'
    script_name = f'process_{mmsi_files_handle}.bat'
    with open(script_name, 'w') as s_file:
        s_file.write(textwrap.dedent(fr"""
            REM Activate python virtual environment
            CALL ..\..\..\venv\Scripts\activate.bat
            
            REM Define group/block
            SET mmsi_group={mmsi_files_handle}
            
            REM Define MMSI data root folder
            SET data_root=C:\mmsi_files
            
            REM update Python Path to allow import of top-level (stonesoup) packages
            SET PYTHONPATH=%PYTHONPATH%;..\..\..\;
            
            REM Process through SS
            ECHO Processing: %mmsi_group%...
            python ..\AisTrackingExactEarthIMM_Single_Bare.py --input_path %data_root%\%mmsi_group% --output_path %data_root%\output_for_%mmsi_group%
            pause
"""))

print(f"Remember to manually create script for final {mmsi_file_count % group_by} files.")
