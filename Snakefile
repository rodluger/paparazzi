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