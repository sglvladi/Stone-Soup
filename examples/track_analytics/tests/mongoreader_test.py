import pymongo
from datetime import timedelta

from stonesoup.reader.mongo import MongoReader, MongoDetectionReader
from stonesoup.feeder.time import TimeSyncFeeder

collections = ["Raw_Static_Reports", "Raw_Position_Reports"]
# reader = MongoReader()
#
# reader.read(host_name="138.253.118.175",
#              host_port=27017,
#              db_name="TA_ExactEarth",
#              collection_names=collections)

host_name = "138.253.118.175"
host_port = 27017
db_name="TA_ExactEarth"

# reader = MongoDetectionReader(host_name="138.253.118.175",
#                               host_port=27017,
#                               db_name="TA_ExactEarth",
#                               collection_name="Raw_Position_Reports",
#                               state_vector_fields=["Longitude", "Latitude"],
#                               time_field="ReceivedTime",
#                               # time_field_format="%Y%m%d_%H%M%S",
#                               timestamp=True
#                               # metadata_fields=['MMSI', 'Time', 'DataType',
#                               #                  'Longitude', 'Latitude',
#                               #                  'COG', 'ROT', 'SOG', 'Heading',
#                               #                  'Maneuver',
#                               #                  'RAIM_flag', 'Vessel_Name',
#                               #                  'Ship_Type',
#                               #                  'Destination', 'IMO',
#                               #                  'Message_ID',
#                               #                  'Repeat_indicator',
#                               #                  'Millisecond',
#                               #                  'Group_code', 'Channel',
#                               #                  'Data_length',
#                               #                  'Navigational_status',
#                               #                  'Accuracy']
#                               )
# detector = TimeSyncFeeder(reader, time_window=timedelta(seconds=1))
# for time, detections in detector.detections_gen():
#     print("Time: {} - Num detections: {}".format(time, len(detections)))

# mongo_client = pymongo.MongoClient(host_name, port=host_port)
# db = mongo_client[db_name]
#
# dynamic_collection = db["Raw_Position_Reports"]
# dynamic_cursor = dynamic_collection.find({})
# static_collection = db["Raw_Static_Reports"]
# static_cursor = dynamic_collection.find({})
# new_collection = db["Raw_Combined_Reports"]
# new_collection.drop()
# docs = []
# for i, row in enumerate(dynamic_cursor.sort([("ReceivedTime",
#                                               pymongo.ASCENDING)]).batch_size(1000)):
#     print(row["ReceivedTimeDate"])
#     docs.append(row)
#     if i%1000 == 0:
#         new_collection.insert_many(docs)
#         docs = []
#     #new_collection.insert(row)
# for row in static_cursor.sort([("ReceivedTime", pymongo.ASCENDING)]):
#     print(row["ReceivedTimeDate"])
#     new_collection.insert(row)
