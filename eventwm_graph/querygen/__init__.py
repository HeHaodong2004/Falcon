"""Query generation, heterogeneous planners/dynamics and privileged Oracle evaluation.

Legal online components (state builder, planners, generator) never import the
simulator; ``native`` and ``oracle`` are the only privileged/simulator modules.
"""
VERSION = 'querygen-oracle-1'
