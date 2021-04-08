from stonesoup.wrapper.matlab import MatlabWrapper
from stonesoup.reader.elint import ElintDetectionReaderMatlab
from stonesoup.tracker.elint import ElintTracker
# import cProfile as profile
# # In outer section of code
# pr = profile.Profile()
# pr.disable()

matlab_path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\ELINT\LyudmilVersion05Mar2021'

detector = ElintDetectionReaderMatlab(dir_path=matlab_path, num_targets=5)
tracker = ElintTracker(matlab_engine=detector.matlab_engine, detector=detector)

for timestamp, tracks in tracker:
    a = 2