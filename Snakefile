# User config
configfile: "showyourwork.yml"


# Import the showyourwork module
module showyourwork:
    snakefile:
        "showyourwork/workflow/Snakefile"
    config:
        config


# Use all default rules
use rule * from showyourwork


# Generate inline figures
use rule figure from showyourwork as inline_figure with:
    input:
        "src/figures/{figure_name}.py",
        "environment.yml"
    wildcard_constraints:
        figure_name="kT00|Y1m1"
    output:
        report("src/figures/{figure_name}.pdf", category="Figure")