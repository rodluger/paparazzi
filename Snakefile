# Import the showyourwork module
module showyourwork:
    snakefile:
        f"{workflow.basedir}/showyourwork/workflow/Snakefile"
    config:
        config


# Use all default rules
use rule * from showyourwork


# Custom rules for this paper
include: "rules/inline.smk"
include: "rules/figdeps.smk"
include: "rules/luhman16b.smk"