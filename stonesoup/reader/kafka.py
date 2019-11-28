from datetime import datetime
from kafka import KafkaConsumer

from ..base import Property
from ..buffered_generator import BufferedGenerator
from .base import DetectionReader

class KafkaDetectionReader(DetectionReader):
    """KafkaDetectionReader

    A detection reader that reads detections from a Kafka broker
    """

    topic = Property(str, doc="The Kafka topic on which to listen for messages")
    state_vector_fields = Property(
        [str], doc='List of columns names to be used in state vector', default=None)
    time_field = Property(
        str, doc='Name of column to be used as time field', default=None)
    time_field_format = Property(
        str, default=None, doc='Optional datetime format')
    timestamp = Property(
        bool, default=False, doc='Treat time field as a timestamp from epoch')
    metadata_fields = Property(
        [str], default=None, doc='List of columns to be saved as metadata, '
                                 'default all')
    kafka_options = Property(
        dict, default={},
        doc='Keyword arguments for the underlying csv reader')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._consumer = KafkaConsumer('test')

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for msg in self._consumer:
            timestamp = datetime.fromtimestamp(msg.timestamp/1000.0)
            # TODO: Extract fields from message
            yield timestamp, msg
