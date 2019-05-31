import pickle
import os
from matplotlib import pyplot as plt


def readData():
    data = pickle.load(open(os.path.join(os.getcwd(), "data/data_augmented2.pickle"), "rb"))
    aisscans = data["aisscans"]
    ownshipdata = data["ownshipdata"]
    radarscans = data["radarscans"]
    return (aisscans, ownshipdata, radarscans)

def readAisData():
    data = pickle.load(open(os.path.join(os.getcwd(), "data/AIS_2017-01-23_2017-01-28.pickle"), "rb"))
    aisscans = data["aisscans"]
    return aisscans

def filterData(scans, box, mintime, maxtime):
    newscans = []
    for scan in scans:
        thistime = scan["time"].replace(tzinfo=None)
        if thistime > mintime and thistime < maxtime:
            newscan = scan
            newscan["detections"] = [
                x for x in scan["detections"] if pointInBox(x, box)]
            if len(newscan["detections"]) > 0:
                newscans.append(newscan)
    return newscans


def pointInBox(detection, box):
    xy = detection["xy"]
    return xy[0] > box[0] and xy[0] < box[1] and xy[1] > box[2] and xy[1] < box[3]


def plotScans(scans, box=None):
    # PLot ais scans
    xx = [s["detections"] for s in scans]
    flat_list = [item for sublist in xx for item in sublist]
    xy = [x["xy"] for x in flat_list]
    plt.plot([x[0] for x in xy],
             [x[1] for x in xy],
             linestyle='', marker='x')
    if box is not None:
        plt.axis(box)
