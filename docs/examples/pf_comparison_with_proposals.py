"""
    This is a example comparison between the the implementation of the standard particle filter
    and the case with proposals, so I can see how the two compare
"""

import numpy as np
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity, \
    ConstantAcceleration
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.proposal.simple import PriorAsProposal, KFasProposal, LocalKFasProposal

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.predictor.particle import ParticlePredictorWithProposal, ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater, ParticleUpdaterWithProposal
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor, UnscentedKalmanPredictor, \
    SqrtKalmanPredictor, CubatureKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater
from stonesoup.types.prediction import ParticleStatePrediction

# ideally I use the existing model
np.random.seed(1908)  # fix a random seed
num_steps = 100
start_time = datetime.now().replace(microsecond=0)

# instantiate the transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1.5),
                                                          ConstantVelocity(1.5)])

transition_model2 = CombinedLinearGaussianTransitionModel([ConstantAcceleration(1.5),
                                                          ConstantAcceleration(1.5)])

# initiate the groundtruth
#truth = GroundTruthPath([GroundTruthState([1, 1, 1, 1, 1, 1], timestamp=start_time)])
truth = GroundTruthPath([GroundTruthState([1, 1, 1, 1], timestamp=start_time)])

# iterate over the various timesteps
for k in range(1, num_steps):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True,
                                  time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=1*k)))

# measurement_model = CartesianToBearingRange(  # relative to the first sensor
#     ndim_state=4,
#     mapping=(0, 2),
#     noise_covar=np.diag([np.radians(1), 5]),
#     translation_offset=np.array([[0], [0]]))

measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([200, 200]))


measurements = []

for i, state in enumerate(truth):  # loop over the ground truth detections
    if i >0:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement, timestamp=state.timestamp,
                                       measurement_model=measurement_model))


from stonesoup.plotter import AnimatedPlotterly, Plotterly, Plotter

time_steps = [start_time + timedelta(seconds=2*i) for i in range(num_steps)]

#plotter = AnimatedPlotterly(timesteps=time_steps)
plotter = Plotterly() # Plotter()
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements, [0, 2], #marker=dict(color='blue'),
                          measurements_label='Detections')

predictor = ParticlePredictor(transition_model)
resampler = SystematicResampler()
updater = ParticleUpdater(measurement_model=measurement_model,
                          resampler=resampler)

x = PriorAsProposal(transition_model, measurement_model)
y = KFasProposal(UnscentedKalmanPredictor(transition_model),
                 UnscentedKalmanUpdater(measurement_model))

zz = LocalKFasProposal(UnscentedKalmanPredictor(transition_model),
                 UnscentedKalmanUpdater(measurement_model))

predictor1 = ParticlePredictor(transition_model, proposal=x)
updater1 = ParticleUpdater(resampler=resampler, measurement_model=measurement_model, proposal=x)

predictor2 = ParticlePredictor(transition_model, proposal=y)
updater2 = ParticleUpdater(resampler=resampler, measurement_model=measurement_model, proposal=y)

predictor3 = ParticlePredictor(transition_model, proposal=zz)
updater3 = ParticleUpdater(resampler=resampler, measurement_model=measurement_model, proposal=zz)


# Load the Particle state priors
from stonesoup.types.state import GaussianState, ParticleState, StateVector
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle

prior_state = GaussianState(
    #StateVector([1, 1, 1, 1, 1, 1]),
    #np.diag([10, 1, 1,  10, 1, 1]))
    StateVector([1, 1, 1, 1]),
    np.diag([10, 1,  10, 1]))
number_particles=10
samples = multivariate_normal.rvs(
    np.array(prior_state.state_vector).reshape(-1),
    prior_state.covar,
    size=number_particles)

# create the particles
particles = [Particle(sample.reshape(-1, 1),
                      weight=Probability(1./len(samples)))
             for sample in samples]

particle_prior1 = ParticleState(state_vector=None,
                               particle_list=particles,
                               timestamp=time_steps[0])

particle_prior2 = ParticleState(state_vector=None,
                               particle_list=particles,
                               timestamp=time_steps[0])

particle_prior3 = ParticleState(state_vector=None,
                                          particle_list=particles,
                                          timestamp=time_steps[0])

particle_prior4 = ParticleState(state_vector=None,
                                          particle_list=particles,
                                          timestamp=time_steps[0])

# Load the tracking components
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track1 = Track(particle_prior1)
track2 = Track(particle_prior2)
track3 = Track(particle_prior3)
track4 = Track(particle_prior4)

for detection in measurements:
    prediction = predictor.predict(particle_prior1, timestamp=detection.timestamp)
    hypothesis = SingleHypothesis(prediction, detection)
    post = updater.update(hypothesis)
    track1.append(post)
    particle_prior1 = track1[-1]
   # # # print(np.sum(post.weight), np.sum(np.exp(post.log_weight)), ' pf standard')
   # #
    prediction = predictor1.predict(particle_prior2, timestamp=detection.timestamp)
    hypothesis = SingleHypothesis(prediction, detection)
    post = updater1.update(hypothesis)
    track2.append(post)
    particle_prior2 = track2[-1]
  #  print(np.sum(post.weight), np.sum(np.exp(post.log_weight)), ' Proposal')

    prediction = predictor2.predict(particle_prior3, timestamp=detection.timestamp,
                                    detection=detection)
    hypothesis = SingleHypothesis(prediction, detection)
    post = updater2.update(hypothesis)
    track3.append(post)
    particle_prior3 = track3[-1]

    prediction = predictor3.predict(particle_prior4, timestamp=detection.timestamp,
                                    detection=detection)
    hypothesis = SingleHypothesis(prediction, detection)
    post = updater3.update(hypothesis)
    track4.append(post)
    particle_prior4 = track4[-1]

    # print(np.sum(post.weight), np.sum(np.exp(post.log_weight)), ' RW')
    #sys.exit()

#sys.exit()
plotter.plot_tracks(track1, mapping=[0,2], track_label='standard')
plotter.plot_tracks(track2, mapping=[0,2], track_label='proposal prior')#, line= dict(color='blue'))
plotter.plot_tracks(track3, mapping=[0,2], track_label='proposal kf') #, line= dict(color='green'))
plotter.plot_tracks(track4, mapping=[0,2], track_label='proposal kf local') #, line= dict(color='green'))
plotter.fig.show() #savefig('test_save.png')