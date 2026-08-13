"""Model builders.

Deliberately empty: re-exporting `build_dfine` here would make every
`from src.models.ssdlite import ...` drag in transformers and timm as a side
effect, adding several seconds to the startup of a run that never touches
D-FINE. Import from the specific module instead.
"""
