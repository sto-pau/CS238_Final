import numpy as np
import matplotlib.pyplot as plt

A = 5
total_time = 70
total_cycles = 7
f = 2*np.pi/ (total_time / total_cycles) #1
A_off = 6
B = 1
w = 3*f

# Define the parameter t
t = np.linspace(0, 70, 1000) #(0, 2*np.pi, 100)

# Define the x and y coordinatesES
x = A*np.cos(w*t)
y = B*np.sin(f*t)
z = t

# Plot the curve
plt.plot(x, y)

# Add axis labels and a title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of x = 4*sin(0.4*t) + 6 and y = B*cos(1.4*t)')

# Display the plot
plt.show()

# Create a 3D figure
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot the scatter plot
ax.scatter(x, y, z)

# Add labels and a title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Scatter Plot')

# Show the plot
plt.show()

amplitude = np.sqrt(x**2 + y**2)
direction = 
# Plot the curve
plt.plot(t, amplitude)

# Add axis labels and a title
plt.xlabel('amplitude')
plt.ylabel('t')

# Display the plot
plt.show()