from pathlib import Path


# Import the showyourwork module
module showyourwork:
    snakefile:
        "showyourwork/workflow/Snakefile"
    config:
        config


# Use all default rules
use rule * from showyourwork


# Inline figures that need user-defined rules
inline_figures = ["kT00", "Y1m1"]


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


# Generate inline figures
use rule figure from showyourwork as inline_figure with:
    input:
        "src/figures/{figure_name}.py",
        "environment.yml"
    wildcard_constraints:
        figure_name="|".join(inline_figures)
    output:
        report("src/figures/{figure_name}.pdf", category="Figure")


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


# Custom rule to download the Luhman 16B dataset
rule luhman16b_data:
    output:
        report("src/figures/luhman16b.pickle", category="Dataset")
    shell:
        "curl https://zenodo.org/record/5534787/files/luhman16b.pickle --output {output[0]}"


# Custom rule to generate the Luhman 16B figures
use rule figure from showyourwork as luhman16b_figures with:
    input:
        "src/figures/luhman16b.py",
        "src/figures/luhman16b.pickle",
        "environment.yml"
    output:
        report("src/figures/luhman16b_map.pdf", category="Figure")