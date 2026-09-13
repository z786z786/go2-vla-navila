# Architecture

The public tree has three boundaries:

1. The existing simulation collectors and navigation modules create observations and expert action chunks.
2. `go2_nav.contracts` defines the versioned observation/action/episode schema and the same action filter used by adapters.
3. `deploy/go2` and `collectors/real_go2` contain the Unitree-facing state, velocity, safety, teleoperation, and RGB-D collection boundary.

`go2_nav.runtime` provides a deterministic two-box visual integration backend and factory-loaded external backends. A model adapter receives instruction, frame history, optional state, and Flow Matching denoising configuration; it returns the same `Action` object consumed by the safety filter and robot adapter.
