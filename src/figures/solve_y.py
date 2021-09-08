from generate import generate
import matplotlib.pyplot as plt


map, flux0, flux = generate(nc=1)

map.show_components(file="test.pdf")
