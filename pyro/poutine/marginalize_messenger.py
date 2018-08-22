from __future__ import absolute_import, division, print_function

from pyro.ops.einsum import shared_intermediates

from .messenger import Messenger
from .trace_messenger import TraceMessenger


class MarginalizeMessenger(Messenger):
    def __init__(self, trace_messenger):
        assert isinstance(trace_messenger, TraceMessenger)
        self.trace_messenger = trace_messenger
        self.cache = None

    @property
    def trace(self):
        return self.trace_messenger.trace

    def __enter__(self):
        self.cache = {}
        return super(MarginalizeMessenger, self).__enter__()

    def _pyro_sample(self, msg):
        raise NotImplementedError('TODO')
        with shared_intermediates(self.cache):
            # TODO eagerly compute marginalized log_prob
