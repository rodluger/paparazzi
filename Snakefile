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


# Custom rule to download the Luhman 16B dataset
rule luhman16b_data:
    output:
        report("src/figures/luhman16b.pickle", category="Dataset")
    shell:
        "curl https://zenodo.org/record/5534787/files/luhman16b.pickle --output {output[0]}"


# Generate inline figures
use rule figure from showyourwork as inline_figure with:
    input:
        "src/figures/{figure_name}.py",
        "environment.yml"
    wildcard_constraints:
        figure_name="kT00|Y1m1"
    output:
        report("src/figures/{figure_name}.pdf", category="Figure")