# Модель: Модель росту бактеріальної популяції Monod (5 семестр)
# Автор: Паламарчук Катерина, група АІ-233

from flask import Flask, request, jsonify
import numpy as np
from scipy.integrate import solve_ivp

app = Flask(__name__)

# ---------- Параметри моделі ----------
N0 = 1000
S0 = 1.0
MU_REF = 0.8

Ks = 0.1
Y = 0.5
d = 0.01


# ---------- Модель Monod ----------
def mu_monod(S):
    return MU_REF * S / (Ks + S)


def ode_system(t, y):
    N, S = y

    mu = mu_monod(S)

    dNdt = mu * N - d * N
    dSdt = -(1.0 / Y) * mu * N

    return [dNdt, dSdt]


# ---------- API ----------
@app.route('/calculate', methods=['GET'])
def calculate():

    # параметр часу через URL
    time_hours = float(request.args.get('time', 24))

    t = np.linspace(0, time_hours, 100)

    y0 = [N0, S0]

    sol = solve_ivp(
        ode_system,
        (t[0], t[-1]),
        y0,
        t_eval=t
    )

    N = sol.y[0]

    result = {
        "model": "Monod bacterial growth model",
        "time_hours": time_hours,
        "initial_bacteria": N0,
        "final_bacteria": round(float(N[-1]), 2)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)