"""Reuse the audited no-state loader with an isolated official-instruction contract."""
from src.dual_target import zoh_policy_server_no_state as worker
from src.inference.navila_velocity import validate_policy_input

if __name__ == '__main__':
    worker.validate_policy_input = validate_policy_input
    worker.main()
