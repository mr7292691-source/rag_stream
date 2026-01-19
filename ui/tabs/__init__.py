# UI Tabs module
from .dashboard_tab import render as render_dashboard_tab
from .extraction_tab import render as render_extraction_tab
from .comparison_tab import render as render_comparison_tab
from .benchmark_tab import render as render_benchmark_tab
from .settings_tab import render as render_settings_tab

__all__ = [
    "render_dashboard_tab",
    "render_extraction_tab",
    "render_comparison_tab",
    "render_benchmark_tab",
    "render_settings_tab",
]
