from __future__ import print_function
import subprocess
import os
import glob
import time
import re

# Generate the github links
hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")[
    :-1
]
slug = os.getenv("TRAVIS_REPO_SLUG", "user/repo")
with open("gitlinks.tex", "w") as f:
    print(
        r"\newcommand{\codelink}[1]{\href{https://github.com/%s/blob/%s/tex/figures/#1.py}{\codeicon}\,\,}"
        % (slug, hash),
        file=f,
    )
    print(
        r"\newcommand{\animlink}[1]{\href{https://github.com/%s/blob/%s/tex/figures/#1.gif}{\animicon}\,\,}"
        % (slug, hash),
        file=f,
    )
    print(
        r"\newcommand{\prooflink}[1]{\href{https://github.com/%s/blob/%s/tex/proofs/#1.ipynb}{\raisebox{-0.1em}{\prooficon}}}"
        % (slug, hash),
        file=f,
    )
    print(
        r"\newcommand{\cilink}[1]{\href{https://dev.azure.com/%s/_build}{#1}}"
        % (slug),
        file=f,
    )

# Tally up the total figure runtime
files = glob.glob("figures/*.figtime")
seconds = 0
for file in files:
    with open(file, "r") as f:
        line = f.readline()
    m, s = re.match("([0-9]*?)m([0-9]*?)s", line).groups()
    seconds += 60 * int(m) + int(s)
with open("figures/allfigures.figtime", "w") as f:
    print(time.strftime("%Mm%Ss", time.gmtime(seconds)), file=f)
