# from kafka import KafkaConsumer
#
# consumer = KafkaConsumer('test')
# for msg in consumer:
#     print(msg)

from stonesoup.reader.kafka import KafkaDetectionReader

reader = KafkaDetectionReader('test')

for timestamp, msg in reader:
    print(timestamp)
    print(msg)