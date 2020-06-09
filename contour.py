import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import functools

# Init figure
fig, ax = plt.subplots()


def func_3d(x, y):
    sumA = 0
    c = [6, 2, 3, 2, 8, 8]
    a = [-3, 4, -8, -6, 3, -6]
    b = [9, -7, 3, -9, -2, -8]
    for i in range(6):
        sumA += c[i] / (1 + (x - a[i]) ** 2 + (y - b[i]) ** 2)
    return sumA


# Make data.
interval = (-10, 10)
points = np.linspace(interval[0], interval[1], 50)
X, Y = np.meshgrid(points, points)
func3d_vectorized = np.vectorize(func_3d)
Z = func3d_vectorized(X, Y)

# Plot the filled contour.
CS = ax.contourf(X, Y, Z)

# Customize the axis.
ax.xaxis.set_major_locator(MultipleLocator(1.000))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_major_locator(MultipleLocator(1.000))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=10)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

# Add a color bar which maps values to colors.
fig.colorbar(CS, shrink=0.75, aspect=10)

# Helper functions
rng = np.random.default_rng()


def random_uniform_2d(low=-10, high=10, size=2):
    return rng.uniform(low=low, high=high, size=size)


def f(x):
    sumA = 0
    c = [6, 2, 4, 2, 8, 8]
    a = [-3, 4, -8, -6, 3, -6]
    b = [9, -7, 3, -9, -2, -8]
    for i in range(6):
        sumA += c[i] / (1 + (x[0] - a[i]) ** 2 + (x[1] - b[i]) ** 2)
    return sumA


def simple_search(N, show_all=False):
    maximum = np.finfo(np.float64).min
    best_point = None
    for i in range(N):
        x = random_uniform_2d()
        if show_all:
            ax.scatter(x[0], x[1], s=1, c='k')
        y = f(x)
        if y > maximum:
            maximum = y
            best_point = x
            # plot all good random points with white
            if show_all:
                ax.scatter(x[0], x[1], s=1, c='w')
    # plot best found point with magenta
    if show_all:
        ax.scatter(best_point[0], best_point[1], s=1, c='magenta')
    return best_point


def best_take(x, m=5, g=1, N=10):
    maximum = np.finfo(np.float64).min
    # plot start point with blue
    ax.scatter(x[0], x[1], s=1, c='b')
    best_point = x
    x_k = x
    for i in range(N):
        for j in range(m):
            vec2 = random_uniform_2d(-1, 1)
            x_delta = [g * (x_k[0] + vec2[0]), g * (x_k[1] + vec2[1])]
            # plot all random points with white
            ax.scatter(x_delta[0], x_delta[1], s=1, c='k')
            y = f(x_delta)
            if y > maximum:
                maximum = y
                best_point = x_delta
        x_k = best_point
        # plot best found point on current iteration with black
        ax.scatter(x_k[0], x_k[1], s=1, c='w')
    # plot best found point with red
    ax.scatter(best_point[0], best_point[1], s=1, c='r')
    return best_point


def global_search_1(m):
    n = 0
    best_point = None
    maximum = np.finfo(np.float64).min
    while n < m:
        rand_2d = random_uniform_2d()
        result = best_take(rand_2d)
        y = f(result)
        if y > maximum:
            maximum = y
            best_point = result
        else:
            n += 1
    return best_point


def global_search_2(m, simple_n=20 * 20):
    n = 0
    x_start = best_take(random_uniform_2d())
    f_x_start = f(x_start)
    while n < m:
        x = simple_search(simple_n)
        if f(x) > f_x_start:
            ax.scatter(x[0], x[1], s=1, c='magenta')
            x = best_take(x)
            f_x_start = f(x)
            x_start = x
        else:
            n += 1
    return x_start


def is_in_interval(x):
    return interval[0] <= x[0] <= interval[1] and interval[0] <= x[1] <= interval[1]


def global_search_3(m):
    n = 0
    x_prev = random_uniform_2d()
    direction = random_uniform_2d(-1, 1)
    x = best_take(x_prev)
    f_x_prev = f(x_prev)
    while n < m:
        x += direction
        f_x_cur = f(x)
        ax.scatter(x[0], x[1], s=1, c="magenta")
        if f_x_cur >= f_x_prev and is_in_interval(x):
            f_x_prev = f_x_cur
        else:
            x = x_prev
            direction = random_uniform_2d(-1, 1)
        n += 1
    return best_take(x)


def global3(m, q):
    n, p = 0, 0

    x0 = random_uniform_2d()
    x1 = best_take(x0)
    f_x1 = f(x1)

    while n < m:
        x = x1
        direction = random_uniform_2d(-1, 1)
        while True:
            f_x_prev = f(x)
            while not is_in_interval(x + direction):
                direction = random_uniform_2d(-1, 1)
            x += direction
            ax.scatter(x[0], x[1], s=1, c="magenta")
            if f(x) > f_x_prev or p > q:
                break
            else:
                p += 1

        x2 = best_take(x)

        if f_x1 < f(x2):
            x1 = x2
        n += 1

    return x1


def simple_global_search(eps, p):
    v = (interval[1] - interval[0]) * (interval[1] - interval[0])
    v_eps = functools.reduce(lambda a, b: a * b, eps)
    p_eps = v_eps / v
    n_min = np.log(1 - p) / np.log(1 - p_eps)
    n_min = int(np.ceil(n_min))

    print(n_min)
    return simple_search(n_min, True), n_min


def main():
    # Part 1
    # res_simple = simple_search(20 * 20 * 10, True)
    # print(res_simple)

    # Part 2
    # res_best = best_take([0, 0])
    # print(res_best)

    # Part 3. Algorithm 1
    # res = global_search_1(5)
    # print(res)

    # Part 3. Algorithm 2
    # res = global_search_2(5)
    # print(res)

    # Part 4. Algorithm 3
    # res = global3(5, 25)
    # print(res)

    # Part 1.5. Simple Search by eps
    eps = 0.5
    probability = 0.1
    res, n = simple_global_search([eps, eps], probability)
    print(res)
    print(f(res))

    ax.set_title('EPS = {0} P = {1} N = {2}'.format(eps, probability, n), fontsize=15)
    plt.show()


if __name__ == '__main__':
    main()
