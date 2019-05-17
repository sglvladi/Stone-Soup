# -*- coding: utf-8 -*-
"""Video readers for Stone Soup.

This is a collection of video readers for Stone Soup, allowing quick reading
of video data/streams.
"""

from abc import abstractmethod
from copy import copy
import datetime
import numpy as np
import ffmpeg
import moviepy.editor as mpy
import threading

from ..base import Property
from ..types.sensordata import ImageFrame
from .base import SensorDataReader
from .file import FileReader
from .url import UrlReader


class FrameReader(SensorDataReader):
    """FrameReader base class

    A FrameReader produces :class:`~.SensorData` in the form of
    :class:`~ImageFrame` objects.
    """

    @property
    @abstractmethod
    def frame(self):
        raise NotImplementedError

    @abstractmethod
    def frame_gen(self):
        raise NotImplementedError

    @property
    def sensor_data(self):
        return self.frame

    def sensor_data_gen(self):
        return self.frame_gen()


class VideoClipReader(FileReader, FrameReader):
    """Base class for frame readers """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame = ImageFrame([])
        self.clip = mpy.VideoFileClip(str(self.path))
        def arrange_rgb(image):
            return image[:, :, [2, 1, 0]]

        self.clip = self.clip.fl_image(arrange_rgb)

    @property
    def frame(self):
        return copy(self._frame)

    def frame_gen(self):
        start_time = datetime.datetime.now()
        for timestamp_sec, frame in self.clip.iter_frames(with_times=True):
            timestamp = start_time + datetime.timedelta(seconds=timestamp_sec)
            self._frame = ImageFrame(frame, timestamp)
            yield timestamp, self.frame


class FFmpegVideoStreamReader(UrlReader, FrameReader):
    """Base class for video based readers"""

    buffer_size = Property(int, doc="Size of the frame buffer",
                           default=1)

    def __init__(self, *args, input_args=[], output_args=[], **kwargs):
        UrlReader.__init__(self, *args, **kwargs)

        in_args = input_args[0] \
            if len(input_args) and isinstance(input_args[0], list) \
            else []
        in_kwargs = input_args[1] \
            if len(input_args) == 2 \
            else input_args[0] if len(input_args) and isinstance(input_args[0], dict) \
            else {}
        out_args = output_args[0] \
            if len(output_args) and isinstance(output_args[0], list) \
            else []
        out_kwargs = output_args[1] \
            if len(output_args) == 2 \
            else output_args[0] if len(output_args) and isinstance(output_args[0], dict) \
            else {}

        self.buffer = []
        self._frame = ImageFrame([])
        self.stream = (
            ffmpeg
                .input(self.url.geturl(), *in_args, **in_kwargs)
                .output('pipe:', *out_args, **out_kwargs)
                .run_async(pipe_stdout=True)
        )
        self._stream_info = next(
            s for s in ffmpeg.probe(self.url.geturl())['streams'] if s['codec_type'] == 'video')
        self._stop_thread = False
        self._capture_thread = threading.Thread(target=self.run)
        self._thread_lock = threading.Lock()
        self._read_event = threading.Event()
        self._capture_thread.start()

    @property
    def frame(self):
        return copy(self._frame)

    def frame_gen(self):
        while self._capture_thread.is_alive():
            frame = ImageFrame([])
            timestamp = None
            self._thread_lock.acquire()
            if len(self.buffer):
                frame = self.buffer.pop(0)
                timestamp = frame.timestamp
                self._thread_lock.release()
            else:
                self._read_event.clear()
                self._thread_lock.release()
                self._read_event.wait()
                self._read_event.clear()
            self._frame = frame
            yield timestamp, self.frame

    def run(self):
        while self.stream.poll() is None:
            width = int(self._stream_info['width'])
            height = int(self._stream_info['height'])
            in_bytes = self.stream.stdout.read(width * height * 3)

            if not in_bytes:
                success = False
            else:
                success = True
                frame_np = (
                    np
                        .frombuffer(in_bytes, np.uint8)
                        .reshape([height, width, 3])
                )
            if (success):
                self._thread_lock.acquire()
                if len(self.buffer) == self.buffer_size:
                    self.buffer.pop(0)
                self.buffer.append(ImageFrame(frame_np, datetime.datetime.now()))
                self._thread_lock.release()
                self._read_event.set()
            else:
                self._thread_lock.acquire()
