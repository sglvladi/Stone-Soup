from stonesoup.reader.elint import ElintDetectionReaderMatlab
from stonesoup.tracker.elint import ElintTracker

matlab_path = r'C:\Users\sglvladi\Documents\GitHub\StoneSoup-sglvladi\stonesoup\_matlab'

detector = ElintDetectionReaderMatlab(dir_path=matlab_path, num_targets=100)
tracker = ElintTracker(matlab_engine=detector.matlab_engine, detector=detector)

for timestamp, tracks in tracker:
    pass
