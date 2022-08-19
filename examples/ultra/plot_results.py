import pickle
import matplotlib.pyplot as plt

mfa = pickle.load(open('./output/mfa.pickle', 'rb'))
jpda = pickle.load(open('./output/jpda.pickle', 'rb'))
phd = pickle.load(open('./output/jpda-phd3.pickle', 'rb'))

mfa_ospa = mfa['values']
jpda_ospa = jpda['values']
phd_ospa = phd['values']
timestamps = [i for i in range(len(mfa_ospa))]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(timestamps, mfa_ospa, label='JPDA')
ax.plot(timestamps, jpda_ospa, label='MFA')
ax.plot(timestamps, phd_ospa, label='PHD')
ax.set_ylabel("OSPA distance")
ax.tick_params(labelbottom=False)
_ = ax.set_xlabel("Time")
plt.legend()
plt.show()