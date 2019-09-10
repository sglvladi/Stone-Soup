import pathlib
import shutil

input_dir = pathlib.Path(r'C:\mmsi_files')

output_dir = pathlib.Path(r'C:\mmsi_files\grouped')

discovered_files = list(input_dir.glob('*.csv'))
discovered_files.sort()
print(f"Discovered {len(discovered_files)} files.")

# Define how files should be split (e.g. how many per folder/archive)
group_every = 2_500
group_files_to_move = []

# Group and archive files
for i, file in enumerate(discovered_files, start=1):
    # Add file to list
    group_files_to_move.append(file)

    # Every group_every files...
    if i >= group_every and i % group_every == 0:

        # Create output dir
        group_dir_name = f'mmsi_files_{i - group_every + 1:06}_to_{i:06}'
        group_output_dir = (output_dir / group_dir_name)
        group_output_dir.mkdir(parents=True, exist_ok=True)

        # Move files into output dir
        print(f"Moving files to: {group_output_dir} ...")
        for f in group_files_to_move:
            f.rename(f'{group_output_dir / f.name}')

        # Create archive
        archive_file = output_dir / group_dir_name
        print(f"Archiving files to: {archive_file} ...")
        shutil.make_archive(
            base_name=archive_file,
            base_dir=group_dir_name,
            root_dir=output_dir,
            format='zip',
        )
        group_files_to_move = []

print(f"{'-' * 80}\nFinished processing {i} of {len(discovered_files)} files.")
print(f"Note: The remaining {len(discovered_files) % group_every} files in '{input_dir}'' will need to be manually moved and archived")
exit(0)
