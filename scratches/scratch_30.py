import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
np.random.seed(7)


def f(x):
    # this is the physical law governing the system we are learning
    return 0.5 * x + 1 + 2*(x**2) - 0.3 * (x**3)


def obs_f(x):
    #this is function for observations (with noise)
    return f(x) + np.random.randn() * 0.2


def fit_f(x, params):
    # this is normally a neural net
    return sum(c * (x ** i) for i, c in enumerate(reversed(params)))

fit_f = np.vectorize(pyfunc=fit_f, excluded=['params'])


DATA = np.array([(x, obs_f(x)) for x in range(1, 17)])
np.random.shuffle(DATA)
#pprint(DATA)

f1, ax1 = plt.subplots()
ax1.set_xlabel("Samples")
ax1.set_ylabel("Loss function")

f2, ax2 = plt.subplots()
support = np.linspace(0, 16, 10)
ax2.plot(DATA[:, 0], DATA[:, 1], "*", label="data points")
ax2.plot(support, [f(x) for x in support], "-+k", label="optimal function")

for BATCH in [1, 2, 4]:

    assert len(DATA) % BATCH == 0

    LR = 0.1  # learning rate

    coeffs = np.array([1.0, 1.0, 1.0, 1.0])  # trainable parameters


    def loss_fn(y, Y):
        # Mean squared relative error loss
        return np.mean(((y - Y) ** 2) / (Y**2 + 0.01))


    def batches(data):
        x = 0
        while x + BATCH <= len(data):
            yield data[x:x+BATCH]
            x += BATCH


    history = []
    for epoch in range(5):
        for iter, batch in enumerate(batches(DATA)):
            X = batch[:, 0]
            Y = batch[:, 1]
            # infer
            y = fit_f(X, params=coeffs)
            # compute loss
            loss = loss_fn(y, Y)
            print(f"at {iter=} {X=} {Y=}, {coeffs=}, {y=}, {loss=}")
            history.append(loss)
            for j in range(32):
                # mess with parameters according to some heuristic - normally you use ADAM algorithm here
                try_coeffs = coeffs + np.random.randn(len(coeffs)) * LR * loss
                # see if it is "better" with updated coeffs
                try_y = fit_f(X, params=try_coeffs)
                try_loss = loss_fn(try_y,  Y)
                if try_loss < loss:
                    coeffs = try_coeffs
                    break
            else:
                print("FAIL")

    ax1.plot(history, label=f"{BATCH=}, {LR=}")
    ax2.plot(support, fit_f(support, params=coeffs), "-", label=f"learned {BATCH=}, {LR=}")


ax1.legend()
ax2.legend()
plt.show()

