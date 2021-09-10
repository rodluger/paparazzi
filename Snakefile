from pathlib import Path


# Import the showyourwork module
module showyourwork:
    snakefile:
        "showyourwork/workflow/Snakefile"
    config:
        config


# Use all default rules
use rule * from showyourwork


# Generate inline figure
use rule figure from showyourwork as kT00 with:
    input:
        "src/figures/kT00.py",
        "environment.yml"
    output:
        report("src/figures/kT00.pdf", category="Figure")


# Generate inline figure
use rule figure from showyourwork as Y1m1 with:
    input:
        "src/figures/Y1m1.py",
        "environment.yml"
    output:
        report("src/figures/Y1m1.pdf", category="Figure")


#
scripts_with_deps = ["spot_setup.py", "spot_infer_y.py"]



def figure_script_with_deps():
    script = showyourwork.figure_script
    if Path(script).name in scripts_with_deps:
        return script
    else:
        raise NotImplementedError()


use rule figure from showyourwork as figure_with_deps with:
    input:
        figure_script_with_deps,
        directory("src/figures/utils"),
        "environment.yml"