import numpy
from matplotlib import pyplot


def random_vec(vec, sigma):
    b = vec + sigma * numpy.random.normal(size=3)
    b = b / numpy.linalg.norm(b)
    return b


def main():
    p = numpy.array([0.0, 0.0, 1.0])

    sigma = 0.01
    dists = [numpy.linalg.norm(p - random_vec(p, sigma)) for _ in range(100000)]

    pyplot.figure()
    pyplot.hist(dists, bins=50)
    pyplot.grid()
    pyplot.title(r"$\sigma = %s$" % sigma)
    pyplot.xlabel(r"$\frac{d}{r}$", fontsize=20)
    pyplot.ylabel("Freq.", fontsize=20)
    pyplot.tight_layout()
    pyplot.savefig("distances_histogram.pdf")
    pyplot.close()

if __name__ == '__main__':
    main()