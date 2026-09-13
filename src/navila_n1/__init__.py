"""N1 geometry-preview preparation; deliberately no Isaac or policy imports."""

# Keep this package initializer import-free: running ``python -m
# src.navila_n1.candidate_manifest`` must not pre-import the target module.
__all__: list[str] = []
