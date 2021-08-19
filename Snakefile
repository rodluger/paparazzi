rule kT00:
    input:
        "figures/kT00.py"
    output:
        "figures/kT00.pdf"
    conda:
        "environment.yml"
    shell:
        "cd figures && python kT00.py"


rule Y1m1:
    input:
        "figures/Y1m1.py"
    output:
        "figures/Y1m1.pdf"
    conda:
        "environment.yml"
    shell:
        "cd figures && python Y1m1.py"


pdf_extras = ["figures/kT00.pdf", "figures/Y1m1.pdf"]