class GenericDenbridgeGraph:
    def __init__(self, model_name):
        self.model_name = model_name
        self.path_to_ckpt = '/home/denbridge/Documents/Tensorflow/trained-models/'\
                            + self.model_name + '/frozen_inference_graph.pb'
        self.path_to_labels = '/home/denbridge/Documents/Tensorflow/trained-models/'\
                            + self.model_name + '/label_map.pbtxt'

class GenericHomeGraph:
    def __init__(self, model_name):
        self.model_name = model_name
        self.path_to_ckpt = 'D:/OneDrive/TensorFlow/scripts/camera_control' \
                        '/models/' + self.model_name + '/frozen_inference_graph.pb'
        self.path_to_labels = 'D:/OneDrive/TensorFlow/scripts/camera_control' \
                              '/data/mscoco_complete_label_map.pbtxt'

class OutputInferenceGraph5_Home:
    path_to_ckpt = 'D:/OneDrive/TensorFlow/trained-models' \
               '/output_inference_graph_v5/frozen_inference_graph.pb'
    path_to_labels = 'D:/OneDrive/TensorFlow/trained-models' \
               '/output_inference_graph_v5/label_map.pbtxt'


class OutputInferenceGraph5_Denbridge:
    path_to_ckpt = '/home/denbridge/Documents/Tensorflow/trained-models/' \
                   'output_inference_graph_v5/frozen_inference_graph.pb'
    path_to_labels = '/home/denbridge/Documents/Tensorflow/trained-models' \
                     '/output_inference_graph_v5/label_map.pbtxt'


class OpenDatasetPlusFasterRcnnCoco876754_Home:
    path_to_ckpt = 'D:/OneDrive/TensorFlow/trained-models' \
                   '/open_dataset_plus_faster_rcnn_coco_v876754/frozen_inference_graph.pb'
    path_to_labels = 'D:/OneDrive/TensorFlow/trained-models' \
                   '/open_dataset_plus_faster_rcnn_coco_v876754/label_map.pbtxt'


class OpenDatasetPlusFasterRcnnCoco876754_Denbridge:
    path_to_ckpt = '/home/denbridge/Documents/Tensorflow/trained-models' \
                   '/open_dataset_plus_faster_rcnn_coco_v876754/frozen_inference_graph.pb'
    path_to_labels = '/home/denbridge/Documents/Tensorflow/trained-models' \
                   '/open_dataset_plus_faster_rcnn_coco_v876754/label_map.pbtxt'
