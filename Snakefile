# Generate inline figures
rule inline_figure:
    input:
        "src/scripts/{figure_name}.py"
    wildcard_constraints:
        figure_name="kT00|Y1m1"
    output:
        "src/tex/figures/{figure_name}.pdf"
    script:
        "src/scripts/{wildcards.figure_name}.py"


# Run the two spectrum map optimization
rule twospec_optim:
    input:
        "src/scripts/twospec_model.py"
    output:
        "src/data/twospec_solution.npz"
    cache:
        True
    script:
        "src/scripts/twospec_optim.py"


# Run the Luhman 16b Doppler Imaging optimization
rule luhman16b_optim:
    input:
        "src/data/luhman16b.pickle",
        "src/scripts/luhman16b_model.py"
    output:
        "src/data/luhman16b_solution.npz"
    cache:
        True
    script:
        "src/scripts/luhman16b_optim.py"


# Generate Luhman 16b Doppler Imaging figures
rule luhman16b_plot:
    input:
        "src/data/luhman16b_solution.npz",
        "src/scripts/luhman16b_model.py"
    output:
        "src/tex/figures/luhman16b_data_model.pdf",
        "src/tex/figures/luhman16b_map.pdf",
        "src/tex/figures/luhman16b_spectra.pdf",
        "src/tex/figures/luhman16b_loss.pdf"
    script:
        "src/scripts/luhman16b_plot.py"


# Copy over the static figure from Crossfield et al.
rule luhman16b_static:
    input:
        "src/static/luhman16b_crossfield.png"
    output:
        "src/tex/figures/luhman16b_crossfield.png"
    shell:
        "cp {input} {output}"