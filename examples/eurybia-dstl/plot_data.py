from pathlib import Path

import matplotlib.pyplot as plt
import pandas

# Load the data
# path_to_gnd = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 1 - 7Feb2025\20250207_TruthTracks.csv')
path_to_gnd = Path(r'C:\Users\sglvladi\OneDrive\Documents\University of Liverpool\PostDoc\EURYBIA - Dstl\Data\Drop 2 - 13Feb2025\20250213_UoLExample\TruthTracks.csv')
gnd_df = pandas.read_csv(path_to_gnd)
gnd_df.sort_values('Time', inplace=True)

# Display the data
grouped_by_id = gnd_df.groupby('PlatformID')

fig = plt.figure()
target_df = grouped_by_id.get_group(1)
lat = target_df['Latitude'].to_numpy()
lon = target_df['Longitude'].to_numpy()
time = target_df['Time'].to_numpy()
# convert time to datetime
time = pandas.to_datetime(time)

fig.suptitle('Platform 1')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(time, lon, '.-')
ax1.set_ylabel('Longitude')
ax1.set_xlabel('Time')
ax2.plot(time, lat, '.-')
ax2.set_ylabel('Latitude')
ax2.set_xlabel('Time')

fig = plt.figure()
for i in range(len(lat)):
    plt.clf()
    plt.plot(lon[:i], lat[:i], '.-')
    plt.title(time[i])
    plt.pause(0.1)


target_df = grouped_by_id.get_group(3)
lat = target_df['Latitude'].to_numpy()
lon = target_df['Longitude'].to_numpy()
time = target_df['Time'].to_numpy()
# convert time to datetime
time = pandas.to_datetime(time)

fig = plt.figure()
fig.suptitle('Platform 3')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(time, lon, '.-')
ax1.set_ylabel('Longitude')
ax1.set_xlabel('Time')
ax2.plot(time, lat, '.-')
ax2.set_ylabel('Latitude')
ax2.set_xlabel('Time')

plt.show(block=True)

for platform_id, group_df in grouped_by_id:
    lat = group_df['Latitude']
    lon = group_df['Longitude']
    plt.plot(lon, lat, label=platform_id)
    a = 2
    plt.pause(1)
plt.legend()
plt.show(block=True)