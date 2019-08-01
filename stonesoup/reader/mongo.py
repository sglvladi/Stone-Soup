
# -*- coding: utf-8 -*-
import pymongo
from datetime import datetime

import numpy as np
from dateutil.parser import parse

from ..base import Property
from ..types.detection import Detection
from .base import Reader, DetectionReader



class MongoReader(Reader):
    """MongoDB Reader"""

    def read(self, host_name, host_port, db_name, collection_names):
        client = pymongo.MongoClient(host_name, port=host_port)
        db = client[db_name]
        static_collection = db[collection_names[0]]
        dynamic_collection = db[collection_names[1]]

        # Pre-process/configure collections
        query = {"InvalidLonLat":{"$ne":"true"}}

        doc = dynamic_collection.find(query)

        for x in doc:
            print(x)


class MongoDetectionReader(DetectionReader):
    """A simple detection reader for MongoDB"""

    host_name = Property(
        str, doc='MongoDB hostname')
    host_port = Property(
        int, doc='MongoDB port')
    db_name = Property(
        str, doc='MongoDB database name')
    collection_name = Property(
        [str], doc='Collection name to read from')
    state_vector_fields = Property(
        [str], doc='List of field names to be used in state vector')
    time_field = Property(
        str, doc='Name of field to be used as time field')
    time_field_format = Property(
        str, default=None, doc='Optional datetime format')
    timestamp = Property(
        bool, default=False, doc='Treat time field as a timestamp from epoch')
    metadata_fields = Property(
        [str], default=None, doc='List of columns to be saved as metadata, '
                                 'default all')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()
        self._mongo_client = pymongo.MongoClient(self.host_name,
                                                 port=self.host_port)
        self._db = self._mongo_client[self.db_name]
        self._collection = self._db[self.collection_name]

    @property
    def detections(self):
        return self._detections.copy()

    def detections_gen(self):

        sort = [(self.time_field, pymongo.ASCENDING)]
        filter = {"InvalidLonLat": {"$ne": "true"}}

        if self.metadata_fields is not None:
            fields = self.state_vector_fields + self.metadata_fields
            fields.append(self.time_field)
            projection = {field: 1 for field in fields}
            cursor = self._collection.find(filter, projection, no_cursor_timeout=True)
        else:
            cursor = self._collection.find(filter, no_cursor_timeout=True)

        for row in cursor.sort(sort):
            if self.time_field_format is not None:
                time_field_value = datetime.strptime(
                    row[self.time_field], self.time_field_format)
            elif self.timestamp is True:
                time_field_value = datetime.fromtimestamp(
                    int(row[self.time_field])/1000)
            else:
                time_field_value = parse(row[self.time_field])

            if self.metadata_fields is None:
                local_metadata = dict(row)
                copy_local_metadata = dict(local_metadata)
                # for (key, value) in copy_local_metadata.items():
                    # if (key == self.time_field) or \
                    #         (key in self.state_vector_fields):
                    #     del local_metadata[key]
            else:
                local_metadata = {field: row[field]
                                  for field in self.metadata_fields
                                  if field in row}


            valid = 0
            for col_name in self.state_vector_fields:
                if row[col_name] != "":
                    valid += 1

            if valid == len(self.state_vector_fields):
                local_metadata["type"] = 1
                detect = Detection(np.array(
                    [[row[col_name]] for col_name in self.state_vector_fields],
                    dtype=np.float32), time_field_value,
                    metadata=local_metadata)
                self._detections = {detect}
            elif valid == 0:
                local_metadata["type"] = 2
                detect = Detection(np.array(
                    [[0]], dtype=np.float32),
                    time_field_value,
                    metadata=local_metadata)
                self._detections = {detect}
            else:
                self._detections = set()
            yield time_field_value, self.detections
        cursor.close()