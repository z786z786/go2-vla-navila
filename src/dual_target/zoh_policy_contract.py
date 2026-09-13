"""New five-action wire contract; finite raw actions, never pre-clamped."""
from .contracts import _finite_vector, ContractValidationError


def validate_action_chunk(actions):
    if not isinstance(actions, (list, tuple)) or len(actions)!=5:
        raise ContractValidationError('ZOH policy requires exactly 5 actions')
    return [_finite_vector(row,3,f'action[{i}]') for i,row in enumerate(actions)]
