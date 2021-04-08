from stonesoup.reader.elint import ElintDetectionReaderMatlab

matlab_path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\ELINT\LyudmilVersion05Mar2021'

detector = ElintDetectionReaderMatlab(dir_path=matlab_path, num_targets=1000.)

for timestamp, detections in detector:
    a = 2