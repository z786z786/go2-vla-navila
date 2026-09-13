"""Safety-filter compatibility wrapper for the public action contract."""
from go2_nav.contracts import Action, ActionFilter

class VelocitySafetyFilter:
    def __init__(self, **kwargs): self.filter = ActionFilter(**kwargs)
    def apply(self, action, dt=0.2):
        if not isinstance(action, Action): action = Action(**action)
        return self.filter.apply(action, dt)
