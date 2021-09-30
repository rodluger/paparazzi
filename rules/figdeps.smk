
from pathlib import Path


# Figures that depend on the scripts in `src/figures/utils`
figures_with_deps = [
    "spot_setup", 
    "spot_infer_y", 
    "spot_infer_yb", 
    "spot_infer_ybs", 
    "spot_infer_ybs_low_snr",
    "snr",
    "inclinations",
    "twospec"
]


def figure_script_with_deps(wildcards):
    """
    If the input to the current rule is one of the scripts in the
    list `figures_with_depths`, returns the path to that script,
    otherwise raises an error. This is used to dynamically
    enable the rule `figure_with_deps` below.

    """
    script = showyourwork.figure_script(wildcards)
    if Path(script).stem in figures_with_deps:
        return script
    else:
        raise NotImplementedError()


# Generate figures w/ extra dependencies
use rule figure from showyourwork as figure_with_deps with:
    input:
        figure_script_with_deps,
        "src/figures/utils/generate.py",
        "src/figures/utils/plot.py",
        "environment.yml"


# Resolve rule ambiguity: always prefer our custom rule
ruleorder: figure_with_deps > figure