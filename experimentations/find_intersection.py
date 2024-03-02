import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def cosine_wave(t, A, f, phi):
    return A * np.cos(2 * np.pi * f * t + phi)

def intersection_eq(t, A1, f1, phi1, A2, f2, phi2):
    return cosine_wave(t, A1, f1, phi1) - cosine_wave(t, A2, f2, phi2)

# Parameters of the cosine waves
A1 = 1.0  # Amplitude of wave 1
f1 = 1.75  # Frequency of wave 1
phi1 = 0  # Phase angle of wave 1

A2 = 1.0  # Amplitude of wave 2
f2 = 1.5  # Frequency of wave 2
phi2 = 0  # Phase angle of wave 2

# Choose an initial guess such that both waves are increasing (or decreasing)
# You can adjust this initial guess based on your specific waves
initial_guess = 0.5

# Find the intersection point
intersection_point = fsolve(intersection_eq, initial_guess, args=(A1, f1, phi1, A2, f2, phi2))
intersection_point = np.abs(intersection_point)

# Calculate the value of y at the intersection point using wave 1 equation
intersection_y = cosine_wave(intersection_point, A1, f1, phi1)

print("Intersection point (t):", intersection_point)
print("Value of y at intersection point:", intersection_y)

# Plot the waves
t_values = np.linspace(0, 2, 1000)
wave1_values = cosine_wave(t_values, A1, f1, phi1)
wave2_values = cosine_wave(t_values, A2, f2, phi2)

plt.plot(t_values, wave1_values, label='Wave 1 (f={})'.format(f1))
plt.plot(t_values, wave2_values, label='Wave 2 (f={})'.format(f2))
plt.scatter(intersection_point, intersection_y, color='red', label='Intersection')

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Cosine Waves Intersection')
plt.legend()
plt.grid(True)
plt.show()
