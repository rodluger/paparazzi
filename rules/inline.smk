# Inline figures that need user-defined rules
inline_figures = ["kT00", "Y1m1"]


# Generate inline figures
use rule figure from showyourwork as inline_figure with:
    input:
        "src/figures/{figure_name}.py",
        "environment.yml"
    wildcard_constraints:
        figure_name="|".join(inline_figures)
    output:
        report("src/figures/{figure_name}.pdf", category="Figure")