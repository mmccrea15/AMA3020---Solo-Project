import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

g = 9.8
l1 = 1
l2 = 1
m1 = 1
m2 = 1

t = np.arange(0, 4 * np.pi, 0.01)
th1 = 15 * np.pi / 180
th2 = 15 * np.sqrt(2) * np.pi / 180
yi = [th1, 0, th2, 0]

def f(y, t):
    ydash = np.zeros_like(y)
    ydash[0] = y[1]
    ydash[1] = (1 / ((m1 + m2) * l1 - m2 * l1 * (np.cos(y[0] - y[2]))**2)) * (
        -m2 * l1 * y[1]**2 * np.sin(y[0] - y[2]) * np.cos(y[0] - y[2]) +
        m2 * g * np.sin(y[2]) * np.cos(y[0] - y[2]) -
        m2 * l2 * y[3]**2 * np.sin(y[0] - y[2]) -
        (m1 + m2) * g * np.sin(y[0])
    )
    ydash[2] = y[3]
    ydash[3] = (1 / (l2 - m2 * l2 * np.cos(y[0] - y[2])**2 / (m1 + m2))) * (
        l1 * y[1]**2 * np.sin(y[0] - y[2]) -
        g * np.sin(y[2]) +
        (m2 * l2 / (m1 + m2)) * y[3]**2 * np.sin(y[0] - y[2]) * np.cos(y[0] - y[2]) +
        g * np.sin(y[0]) * np.cos(y[0] - y[2])
    )
    return ydash

sol = odeint(f, yi, t)

sol_degrees = np.degrees(sol)  

x1 = l1 * np.sin(sol_degrees[:, 0])
y1 = -l1 * np.cos(sol_degrees[:, 0])
x2 = l1 * np.sin(sol_degrees[:, 0]) + l2 * np.sin(sol_degrees[:, 2])
y2 = -l1 * np.cos(sol_degrees[:, 0]) - l2 * np.cos(sol_degrees[:, 2])

plt.plot(t, sol_degrees[:, 0], '-b', label='Pendulum 1')
plt.plot(t, sol_degrees[:, 2], '-r', label='Pendulum 2')
plt.xlabel('time (s)')
plt.ylabel('$\phi_i$ (degrees)')
plt.ylim(-25, 25)
plt.legend()
plt.title('Angular Displacement of Double Pendulum for Small Angles')
plt.savefig('5.png', dpi=300)
plt.close()

#####################################################################


##############################################################################

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

g = 9.8
l1 = 1
l2 = 1
m1 = 1
m2 = 1

t = np.arange(0, 4 * np.pi, 0.01)
th1 = 15 * np.pi / 180
th2 = 0
yi = [th1, 0, th2, 0]

def f(y, t):
    ydash = np.zeros_like(y)
    ydash[0] = y[1]
    ydash[1] = (1 / ((m1 + m2) * l1 - m2 * l1 * (np.cos(y[0] - y[2]))**2)) * (
        -m2 * l1 * y[1]**2 * np.sin(y[0] - y[2]) * np.cos(y[0] - y[2]) +
        m2 * g * np.sin(y[2]) * np.cos(y[0] - y[2]) -
        m2 * l2 * y[3]**2 * np.sin(y[0] - y[2]) -
        (m1 + m2) * g * np.sin(y[0])
    )
    ydash[2] = y[3]
    ydash[3] = (1 / (l2 - m2 * l2 * np.cos(y[0] - y[2])**2 / (m1 + m2))) * (
        l1 * y[1]**2 * np.sin(y[0] - y[2]) -
        g * np.sin(y[2]) +
        (m2 * l2 / (m1 + m2)) * y[3]**2 * np.sin(y[0] - y[2]) * np.cos(y[0] - y[2]) +
        g * np.sin(y[0]) * np.cos(y[0] - y[2])
    )
    return ydash

sol = odeint(f, yi, t)

sol_degrees = np.degrees(sol)  # Convert radians to degrees

x1 = l1 * np.sin(sol_degrees[:, 0])
y1 = -l1 * np.cos(sol_degrees[:, 0])
x2 = l1 * np.sin(sol_degrees[:, 0]) + l2 * np.sin(sol_degrees[:, 2])
y2 = -l1 * np.cos(sol_degrees[:, 0]) - l2 * np.cos(sol_degrees[:, 2])

# Plotting angular displacement of the first pendulum
plt.plot(t, sol_degrees[:, 0], '-b')
#plt.ylim(-0.3, 0.3)
plt.title('Angular Displacement of the First Pendulum for Small Angles')
plt.xlabel('time (s)')
plt.ylabel('$\phi_1$ (degrees)')
plt.savefig('11.png', dpi=300)
plt.close()

# Plotting angular displacement of the second pendulum
plt.plot(t, sol_degrees[:, 2], '-r')
#plt.ylim(-0.4, 0.4)
plt.title('Angular Displacement of the Second Pendulum for Small Angles')
plt.xlabel('time (s)')
plt.ylabel('$\phi_2$ (degrees)')
plt.savefig('12.png', dpi=300)
plt.close()

# Plotting $(\sqrt{2} \phi_1 - \phi_2)$ vs time
plt.plot(t, np.sqrt(2) * sol_degrees[:, 0] - sol_degrees[:, 2], '-')
#plt.ylim(-0.4, 0.4)
plt.xlabel('time (s)')
plt.ylabel('$(\sqrt{2}\phi_1 - \phi_2)$ (degrees)')
plt.title('Angular Displacement of the Linear Combination $(\sqrt{2}\phi_1 - \phi_2)$')
#$(\sqrt{2}\phi_1 - \phi_2)$ vs time
plt.savefig('13.png', dpi=300)
plt.close()

# Plotting $(\sqrt{2} \phi_1 + \phi_2)$ vs time
plt.plot(t, np.sqrt(2) * sol_degrees[:, 0] + sol_degrees[:, 2], '-')
#plt.ylim(-0.4, 0.4)
plt.xlabel('time (s)')
plt.ylabel('$(\sqrt{2}\phi_1 + \phi_2)$ (degrees)')
plt.title('Angular Displacement of the Linear Combination $(\sqrt{2}\phi_1 + \phi_2)$')
#plt.title('$(\sqrt{2}\phi_1 + \phi_2)$ vs time')
plt.savefig('14.png', dpi=300)
plt.close()


###################################################################


# Define parameters
m = 1.0  # mass
l = 1.0  # length of the rod
g = 9.8  # acceleration due to gravity

# Define the Lagrangian function
def lagrangian(phi1, phi2, dphi1, dphi2):
    term1 = 0.5 * m * l**2 * (2*dphi1**2 + dphi2**2 + 2*dphi1*dphi2*np.cos(phi1 - phi2))
    term2 = m * g * l * (2 * np.cos(phi1) + np.cos(phi2))
    return term1 - term2

# Define the derivatives of phi1 and phi2
def derivs(t, y):
    phi1, dphi1, phi2, dphi2 = y
    phi1_dot = dphi1
    dphi1_dot = (-m*l*dphi1**2*np.sin(phi1-phi2) + m*g*np.sin(phi2)*np.cos(phi1-phi2) - m*l*dphi2**2*np.sin(phi1-phi2) - (m+m)*g*np.sin(phi1)) / ((m+m)*l - m*l*np.cos(phi1-phi2)**2)
    phi2_dot = dphi2
    dphi2_dot = (m*l*dphi2**2*np.sin(phi1-phi2) + (m+m)*g*np.sin(phi1)*np.cos(phi1-phi2) + (m+m)*l*dphi1**2*np.sin(phi1-phi2) - (m+m)*g*np.sin(phi2)) / ((m+m)*l - m*l*np.cos(phi1-phi2)**2)
    return [phi1_dot, dphi1_dot, phi2_dot, dphi2_dot]

# Runge-Kutta method
def runge_kutta(y, t, dt):
    k1 = dt * np.array(derivs(t, y))
    k2 = dt * np.array(derivs(t + 0.5*dt, y + 0.5*k1))
    k3 = dt * np.array(derivs(t + 0.5*dt, y + 0.5*k2))
    k4 = dt * np.array(derivs(t + dt, y + k3))
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Initial conditions
phi1_0 = 0.0  # initial angle of the first pendulum
dphi1_0 = 0.0  # initial angular velocity of the first pendulum
phi2_0 = 0.0  # initial angle of the second pendulum
dphi2_0 = 3.0  # initial angular velocity of the second pendulum

# Convert initial conditions to radians
phi1_0 = np.radians(phi1_0)
phi2_0 = np.radians(phi2_0)

# Time parameters
t_start = 0.0
t_end = 10.0
dt = 0.01

# Number of steps
num_steps = int((t_end - t_start) / dt)

# Arrays to store results
time_points = np.linspace(t_start, t_end, num_steps)
phi1_values = np.zeros(num_steps)
phi2_values = np.zeros(num_steps)

# Initial conditions
y = np.array([phi1_0, dphi1_0, phi2_0, dphi2_0])

# Run the Runge-Kutta method
for i in range(num_steps):
    phi1_values[i] = np.degrees(y[0])  # Convert radians to degrees
    phi2_values[i] = np.degrees(y[2])  # Convert radians to degrees
    y = runge_kutta(y, time_points[i], dt)

# Plot the results
plt.plot(time_points, phi1_values, label='Pendulum 1')
plt.plot(time_points, phi2_values, label='Pendulum 2')
plt.xlabel('time')
plt.ylabel('$\phi_i$ (degrees)')  # Update ylabel
plt.legend()
plt.title('Motion of the Double Pendulum using Runge-Kutta Method')
plt.savefig('double_pendulum_motion.png', dpi=300)
plt.show()
