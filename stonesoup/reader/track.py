import threading
from copy import copy

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..reader.base import Reader
from ..tracker.base import Tracker


class TrackReader(Reader):
    tracker: Tracker = Property(doc='Tracker from which to read tracks')
    run_async: bool = Property(
        doc="If set to ``True``, the reader will read tracks from the tracker asynchronously "
            "and only yield the latest set of tracks when iterated."
            "Defaults to ``False``",
        default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Variables used in async mode
        if self.run_async:
            self._buffer = None
            # Initialise frame capture thread
            self._capture_thread = threading.Thread(target=self._capture)
            self._capture_thread.daemon = True
            self._thread_lock = threading.Lock()
            self._capture_thread.start()

    @property
    def tracks(self):
        return self.current[1]

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        if self.run_async:
            yield from self._tracks_gen_async()
        else:
            yield from self._tracks_gen()

    def _capture(self):
        for timestamp, tracks in self.tracker:
            self._thread_lock.acquire()
            self._buffer = (timestamp, tracks)
            self._thread_lock.release()

    def _tracks_gen(self):
        for timestamp, tracks in self.tracker:
            yield timestamp, tracks

    def _tracks_gen_async(self):
        while self._capture_thread.is_alive():
            if self._buffer is not None:
                self._thread_lock.acquire()
                timestamp, tracks = copy(self._buffer)
                self._buffer = None
                self._thread_lock.release()
                yield timestamp, tracks