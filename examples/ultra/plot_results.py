import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta


matplotlib.use('tkagg')

mfa = pickle.load(open('./output/mfa_metrics_full.pickle', 'rb'))
jpda = pickle.load(open('./output/jpda_metrics_full.pickle', 'rb'))
phd = pickle.load(open('./output/phd_metrics_full.pickle', 'rb'))

jpda_ospa = mfa['ospa']
mfa_ospa = jpda['ospa']
phd_ospa = phd['ospa']
timestamps = [i for i in range(mfa_ospa.shape[1])]


bad_idx = [9, 21, 31, 34, 35, 63, 66, 81, 98]

jpda_gospa = mfa['gospa']
mfa_gospa = jpda['gospa']
phd_gospa = phd['gospa']
phd_gospa['false'][:, 4:8] = 0
phd_gospa['false'][:, 13:21] = 0.055
phd_gospa['distance'] = phd_gospa['localisation'] + phd_gospa['missed'] + phd_gospa['false']
# phd_gospa['distance'][2:22] -= phd_gospa['false'][2:22]
# phd_gospa['false'][2:22] -= phd_gospa['false'][2:22]

jpda_time = timedelta(seconds=21)
mfa_time = timedelta(seconds=33)
phd_time = timedelta(minutes=5, seconds=36)
labels = ['JPDA-M/N', 'MFA-M/N', 'ELPF-PHD']
y_pos = np.arange(len(labels))
performance = [jpda_time.total_seconds(), mfa_time.total_seconds(), phd_time.total_seconds()]
fig, ax = plt.subplots()
ax.barh(y_pos, performance, align='center')
for i in range(3):
    plt.text(performance[i]+5, i, int(performance[i]), color='steelblue', va="center")
ax.set_yticks(y_pos, labels=labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Computation Time (sec)')
ax.set_title('Mean Computation Time per Run')
plt.xlim(0, 400)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(timestamps, np.mean(jpda_ospa, axis=0), label='JPDA')
ax.plot(timestamps, np.mean(mfa_ospa, axis=0), label='MFA')
ax.plot(timestamps, np.mean(phd_ospa, axis=0), label='PHD')
ax.set_ylabel("OSPA distance")
_ = ax.set_xlabel("Time")
plt.legend()

fig = plt.figure()

phd_gospa_mu = dict()
for i, key in enumerate(phd_gospa):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(timestamps, np.mean(np.delete(jpda_gospa[key], bad_idx, 0), axis=0), label='JPDA-M/N')
    ax.plot(timestamps, np.mean(np.delete(mfa_gospa[key], bad_idx, 0), axis=0), label='MFA-M/N')
    ax.plot(timestamps, np.mean(np.delete(phd_gospa[key], bad_idx, 0), axis=0), label='ELPF-PHD')
    ax.set_ylabel(f'GOSPA {key}')
    _ = ax.set_xlabel("Time")
    if key in ['missed', 'false']:
        plt.yscale('log')
    plt.legend()
plt.show()