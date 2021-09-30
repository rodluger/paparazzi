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
        report("src/figures/luhman16b_map.pdf", category="Figure"),
        report("src/figures/luhman16b_spectra.pdf", category="Figure"),
        report("src/figures/luhman16b_data_model.pdf", category="Figure")