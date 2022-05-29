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

# Generate Luhman 16b doppler imaging figures
rule luhman16b:
    input:
        "src/data/luhman16b.pickle"
    output:
        "src/tex/figures/luhman16b_data_model.pdf",
        "src/tex/figures/luhman16b_map.pdf",
        "src/tex/figures/luhman16b_spectra.pdf"
    script:
        "src/scripts/luhman16b.py"

# Copy over the static figure from Crossfield et al.
rule crossfield:
    input:
        "src/static/luhman16b_crossfield.png"
    output:
        "src/tex/figures/luhman16b_crossfield.png"
    shell:
        "cp {input} {output}"