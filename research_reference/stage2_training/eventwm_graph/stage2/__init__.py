"""Offline trajectory-conditioned prediction of newly observed static cells."""


def __getattr__(name):
    if name in ('Stage2Predictor','apply_graph_delta'):
        from .api import Stage2Predictor,apply_graph_delta
        return {'Stage2Predictor':Stage2Predictor,'apply_graph_delta':apply_graph_delta}[name]
    raise AttributeError(name)
