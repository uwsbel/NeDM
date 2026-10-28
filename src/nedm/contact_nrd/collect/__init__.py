"""Simulation collectors that write contact NRD training data (system.json + unified_data.npz).

The collectors run MuJoCo on the CPU and need only numpy, mujoco and scipy; they do not import torch. The data contract is in contact_nrd/collection_contract.md. ``export`` writes a data directory,
``validate`` checks one.
"""
