import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Function to define the equations of motion for the double pendulum
def double_pendulum(t, y, L):
    """
    t: time
    y: array containing the angles and angular velocities of the pendulum
    L: length of the pendulum rod
    """
    theta1, theta1_dot, theta2, theta2_dot = y

    # Define constants
    g = 9.81  # acceleration due to gravity

    # Equations of motion
    theta1_ddot = -(g/L) * np.sin(theta1)
    theta2_ddot = -(g/L) * np.sin(theta2)

    return [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]

# Length of the pendulum rod
L = 1.0

# Initial conditions: small angles and zero initial velocities
initial_conditions = [1, 0, 2, 0]

# Time span for simulation
t_span = (0, 10)

# Solve the equations of motion
sol = solve_ivp(lambda t, y: double_pendulum(t, y, L), t_span, initial_conditions)

# Extract the angles
theta1 = sol.y[0]
theta2 = sol.y[2]

# Plot the motion
plt.plot(sol.t, theta1, label='Theta 1')
plt.plot(sol.t, theta2, label='Theta 2')
plt.title('Motion of Double Pendulum (Small Angles)')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.legend()
#plt.grid(True)
plt.savefig('double_pendulum_small_angles.png')

#the motion of a double pendulum for small angles and plot the angles as functions of time. Keep in mind that this is only an approximation, and for larger angles, the motion becomes chaotic and cannot be accurately represented by simple harmonic motion equations.


plt.close()

####################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

g = 9.8
l1 = 1
l2 = 1
m1 = 1
m2 = 1

def f(y, t):
    th1, w1, th2, w2 = y

    ydash = np.zeros_like(y)

    ydash[0] = w1
    ydash[1] = (1 / ((m1 + m2) * l1 - m2 * l1 * np.cos(th1 - th2)**2)) * \
               ((-m2 * l1 * w1 * w1 * np.sin(th1 - th2) * np.cos(th1 - th2)) +
                (m2 * g * np.sin(th2) * np.cos(th1 - th2)) -
                (m2 * l2 * w2 * w2 * np.sin(th1 - th2)) -
                ((m1 + m2) * g * np.sin(th1)))

    ydash[2] = w2
    ydash[3] = (1 / (l2 - (m2 * l2 * np.cos(th1 - th2) * np.cos(th1 - th2) / (m1 + m2)))) * \
               ((l1 * w1 * w1 * np.sin(th1 - th2)) -
                (g * np.sin(th2)) +
                ((m2 * l2 / (m1 + m2)) * w2 * w2 * np.sin(th1 - th2) * np.cos(th1 - th2))) + \
               (g * np.sin(th1) * np.cos(th1 - th2))

    return ydash



#t = np.arange(0, 5*np.pi, 0.02)
t = np.arange(0, 4 * np.pi, 0.01)
#th1 = 90 * np.pi / 180
#th1 = 45
#th1 = 15 * np.pi / 180
#th2 = np.sqrt(2)*15 * np.pi / 180
#th1 = 15
#th2 = np.sqrt(2)*15
th1 = 0
th2 = 0
w1 = 0
w2 = 3
yi = [th1, w1, th2, w2]

sol = odeint(f, yi, t)

x1 = l1 * np.sin(sol[:, 0])
y1 = -l1 * np.cos(sol[:, 0])
x2 = l1 * np.sin(sol[:, 0]) + l2 * np.sin(sol[:, 2])
y2 = -l1 * np.cos(sol[:, 0]) - l2 * np.cos(sol[:, 2])

#plt.figure(figsize=(10, 8))

#plt.subplot(2, 2, 1)
plt.plot(x1, y1, '-b', label='Pendulum 1')
plt.plot(x2, y2, '-r', label= 'Pendulum 2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory of the Double Pendulum')
plt.legend()
plt.savefig('double_pendulum_trajectory1.png',dpi=300)
#blue line is pendulum with mass m1 and red line is pendulum mass m2. Mass m1 is always restricted to some kind of an arc of a circle, but mass m2 is not restricted to an arc, but can take any shape, 

plt.close()

#plt.subplot(2, 2, 2)
plt.plot(t, sol[:, 0], '-b',label='Pendulum 1')
plt.plot(t, sol[:, 2], '-r',label='Pendulum 2')
plt.xlabel('time')
plt.ylabel('$\phi$ (degrees)')
plt.title('Angular Displacement of the Double Pendulum')
plt.legend()
plt.savefig('double_pendulum_trajectory2.png',dpi=300)
# this plot is the angular displacement of mass m1 and the angular displacement of mass m2 with time. Blue line represents the angular displacement of mass m1 as it varies with time, and the red line represents the angular displacement of mass m2 as it varies with time. Angular displacement is theta 1 with respect to the vertical axis and theta 2 with respect to the vertical axis. Angular displacement of 45 degrees initially 

plt.close()

#plt.subplot(2, 2, 3)
plt.plot(t, sol[:, 1], 'b',label='Pendulum 1')
plt.plot(t, sol[:, 3], 'r',label='Pendulum 2')
plt.xlabel('time')
plt.ylabel('velocity (degrees/s)')
plt.title('Velocity of the Double Pendulum')
plt.legend()
plt.savefig('double_pendulum_trajectory3.png', dpi=300)
#Third plot is theta 1 dot and theta 2 dot, that is the angular velocity of mass m1 and the angular velocity of mass m2 with time. The sharp peaks in the velocity graph represents the velocity suddenly changing
plt.close()

#plt.subplot(2, 2, 4)
plt.plot(sol[:, 0], sol[:, 2], '-ok')
plt.xlabel('$\phi_1$ (degrees)')
plt.ylabel('$\phi_2$ (degrees)')
plt.title('$\phi_1$ vs $\phi_2$')
#plt.title('nabla 'r'$\nabla=100$')

#plt.tight_layout()
plt.savefig('double_pendulum_trajectory4.png',dpi=300)
#Is there some kind of correlation between theta 1 and theta 2. Demonstration of the chaos/behaviour of the system. Nature of the motio is very unpredictable and chaotic with large angles. 
plt.close()

#Can start 3 pendulums with initial displacment very similar to each other but bc of the small difference (91,92,93), they end up undergoing vastly different trajectories. This shows how systems are very sensitive to very small differences of initial displacement. 

###################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Function that returns the derivatives of the system
def double_pendulum(t, y, l1, l2, m1, m2, g):
    theta1, omega1, theta2, omega2 = y

    # Equations of motion
    dydt = [omega1,
            (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) -
             2 * np.sin(theta1 - theta2) * m2 * (omega2**2 * l2 + omega1**2 * l1 * np.cos(theta1 - theta2))) /
            (l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))),
            omega2,
            (2 * np.sin(theta1 - theta2) * (omega1**2 * l1 * (m1 + m2) +
             g * (m1 + m2) * np.cos(theta1) + omega2**2 * l2 * m2 * np.cos(theta1 - theta2))) /
            (l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))]

    return dydt

# Set up parameters
l1 = 1.0  # length of pendulum 1
l2 = 1.0  # length of pendulum 2
m1 = 1.0  # mass of pendulum 1
m2 = 1.0  # mass of pendulum 2
g = 9.8   # acceleration due to gravity

# Set up initial conditions
theta1_0 = np.pi / 4  # initial angle of pendulum 1 (45 degrees)
omega1_0 = 0.0       # initial angular velocity of pendulum 1
theta2_0 = np.pi / 2  # initial angle of pendulum 2 (90 degrees)
omega2_0 = 0.0       # initial angular velocity of pendulum 2

y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

# Set up time span
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the ODE using the Runge-Kutta method
sol = solve_ivp(double_pendulum, t_span, y0, args=(l1, l2, m1, m2, g), t_eval=t_eval, method='RK45')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Pendulum 1')
plt.plot(sol.t, sol.y[2], label='Pendulum 2')
plt.title('Double Pendulum Motion')
plt.xlabel('Time (s)')
plt.ylabel('Angle (radians)')
plt.legend()
plt.savefig('eqs_of_motion.png', dpi=300)

plt.close()

######################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

g = 9.8
l1 = 1
l2 = 1
m1 = 1
m2 = 1

def f(y, t):
    th1, w1, th2, w2 = y

    ydash = np.zeros_like(y)

    ydash[0] = w1
    ydash[1] = (1 / ((m1 + m2) * l1 - m2 * l1 * np.cos(th1 - th2)**2)) * \
               ((-m2 * l1 * w1 * w1 * np.sin(th1 - th2) * np.cos(th1 - th2)) +
                (m2 * g * np.sin(th2) * np.cos(th1 - th2)) -
                (m2 * l2 * w2 * w2 * np.sin(th1 - th2)) -
                ((m1 + m2) * g * np.sin(th1)))

    ydash[2] = w2
    ydash[3] = (1 / (l2 - (m2 * l2 * np.cos(th1 - th2) * np.cos(th1 - th2) / (m1 + m2)))) * \
               ((l1 * w1 * w1 * np.sin(th1 - th2)) -
                (g * np.sin(th2)) +
                ((m2 * l2 / (m1 + m2)) * w2 * w2 * np.sin(th1 - th2) * np.cos(th1 - th2))) + \
               (g * np.sin(th1) * np.cos(th1 - th2))

    return ydash



#t = np.arange(0, 5*np.pi, 0.02)
t = np.arange(0, 4 * np.pi, 0.01)
#th1 = 90 * np.pi / 180
#th1 = 45
#th1 = 15 * np.pi / 180
#th2 = np.sqrt(2)*15 * np.pi / 180
#th1 = 15
#th2 = np.sqrt(2)*15
th1 = 0
th2 = 0
w1 = 0
w2 = 10
yi = [th1, w1, th2, w2]

sol = odeint(f, yi, t)

x1 = l1 * np.sin(sol[:, 0])
y1 = -l1 * np.cos(sol[:, 0])
x2 = l1 * np.sin(sol[:, 0]) + l2 * np.sin(sol[:, 2])
y2 = -l1 * np.cos(sol[:, 0]) - l2 * np.cos(sol[:, 2])

#plt.figure(figsize=(10, 8))

#plt.subplot(2, 2, 1)
plt.plot(x1, y1, '-b', label='Pendulum 1')
plt.plot(x2, y2, '-r', label= 'Pendulum 2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory of the Double Pendulum')
plt.legend()
plt.savefig('1.png',dpi=300) 
plt.close()

#plt.subplot(2, 2, 2)
plt.plot(t, sol[:, 0], '-b',label='Pendulum 1')
plt.plot(t, sol[:, 2], '-r',label='Pendulum 2')
plt.xlabel('time')
plt.ylabel('$\phi_i$ (degrees)')
plt.title('Angular Displacement of the Double Pendulum')
plt.legend()
plt.savefig('2.png',dpi=300)
plt.close()

#plt.subplot(2, 2, 3)
plt.plot(t, sol[:, 1], 'b',label='Pendulum 1')
plt.plot(t, sol[:, 3], 'r',label='Pendulum 2')
plt.xlabel('time')
plt.ylabel('velocity (degrees/s)')
plt.title('Velocity of the Double Pendulum')
plt.legend()
plt.savefig('3.png', dpi=300)
plt.close()

#plt.subplot(2, 2, 4)
plt.plot(sol[:, 0], sol[:, 2] * 180 / np.pi, '-ok')
plt.xlabel('$\phi_1$ (degrees)')
plt.ylabel('$\phi_2$ (degrees)')
plt.title('$\phi_1$ vs $\phi_2$')

plt.savefig('4.png',dpi=300)
