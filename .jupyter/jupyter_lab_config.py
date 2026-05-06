# JupyterLab configuration for SageMaker Studio spaces.
#
# This file lives on the EFS home directory and persists across space restarts.
# Copy to ~/.jupyter/jupyter_lab_config.py inside the SageMaker space.

# Autosave the .ipynb file every 30 seconds (default: 120s).
# Captures any cell outputs that reached the browser before a disconnect.
# Does NOT capture outputs produced while the browser is disconnected —
# use scripts/train.py with file-based logging for long-running cells.
c.ServerApp.autosave_interval = 30  # noqa: F821 — Jupyter injects `c` at runtime

# Never cull idle kernels. Without this, JupyterLab can kill a kernel that
# appears idle even while a background training loop is running in it.
# SageMaker's space-level idle timeout (set in scripts/lcc_setup.sh) handles
# instance shutdown independently of this setting.
c.MappingKernelManager.cull_idle_timeout = 0  # noqa: F821
c.MappingKernelManager.cull_connected = False  # noqa: F821
