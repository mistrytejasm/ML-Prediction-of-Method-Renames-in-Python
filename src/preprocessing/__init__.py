"""
Preprocessing (phase 1, not implemented yet).

Responsibility: turn the X returned by src.data.load_master.select_xy_groups()
into model-ready matrices, for example log-transforms of long-tailed counts and
scaling for linear models. Every learned transformation is fitted on training
folds only and then applied to validation/test data. The master dataset is never
modified. See docs/experiment_protocol.md.
"""
