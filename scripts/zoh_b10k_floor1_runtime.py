"""User-authorized one-run 7 GiB admission / 1 GiB floor B-10k runtime."""
from src.dual_target.gpu_wait import GpuAdmissionPolicy
import scripts.zoh_b10k_train_runtime as base

POLICY=GpuAdmissionPolicy('zoh_b10k_train_shared7g_floor1_v1',7168,allow_existing_compute=True)
SCOPE='b-10000-train-rollout-floor1-exception'


def main():
    # Reuse the already-audited train-only runtime while replacing only the
    # immutable admission identity. The outer queue separately enforces 1 GiB.
    base.POLICY=POLICY;base.SCOPE=SCOPE;base.main()


if __name__=='__main__':main()
