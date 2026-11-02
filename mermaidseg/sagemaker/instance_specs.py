"""SageMaker GPU instance specs + DataLoader worker-sizing guidance.

Used by :meth:`mermaidseg.experiment.Experiment.validate` to warn (never block) when a run's
``num_workers`` / ``persistent_workers`` are risky for the chosen ``instance_type`` — the class of
mistake behind the dinov3-lora-qv-r8 host-RAM OOM (see ``docs/investigating-training-runs.md`` and
``scripts/diagnostics/dataloader_rss_findings.md``).

Keep this table in sync with ``sagemaker/runs/*.yaml``. Values are the AWS published vCPU / host
RAM for each instance (GPU RAM noted for context; the OOM was host RAM, not GPU).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstanceSpec:
    vcpu: int
    ram_gb: int
    gpu: str
    gpu_ram_gb: int


# Only the instances actually referenced by sagemaker/runs/*.yaml. Extend as new ones are used.
INSTANCE_SPECS: dict[str, InstanceSpec] = {
    "ml.g5.xlarge": InstanceSpec(vcpu=4, ram_gb=16, gpu="A10G", gpu_ram_gb=24),
    "ml.g5.2xlarge": InstanceSpec(vcpu=8, ram_gb=32, gpu="A10G", gpu_ram_gb=24),
    "ml.g5.4xlarge": InstanceSpec(vcpu=16, ram_gb=64, gpu="A10G", gpu_ram_gb=24),
    "ml.g6.2xlarge": InstanceSpec(vcpu=8, ram_gb=32, gpu="L4", gpu_ram_gb=24),
    "ml.g6.4xlarge": InstanceSpec(vcpu=16, ram_gb=64, gpu="L4", gpu_ram_gb=24),
    "ml.p3.2xlarge": InstanceSpec(vcpu=8, ram_gb=61, gpu="V100", gpu_ram_gb=16),
}

# Host RAM (GiB) at/below which persistent DataLoader workers are a real OOM risk on a
# large-annotation run. The 32 GiB g6.2xlarge is exactly where the dinov3-lora-qv-r8 run died.
SMALL_RAM_GB = 32


def recommended_num_workers(instance_type: str) -> int | None:
    """Suggested DataLoader ``num_workers`` for an instance: leave one vCPU for the main
    process.

    Returns ``None`` for an unknown instance type.
    """
    spec = INSTANCE_SPECS.get(instance_type)
    if spec is None:
        return None
    return max(1, spec.vcpu - 1)
