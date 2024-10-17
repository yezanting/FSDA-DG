
import numpy as np
import matplotlib.pyplot as plt

# data
z = np.array([])

# set x and y data
x = np.array([])
y = np.array([])
X, Y = np.meshgrid(x, y)

#  set image and subimage
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# set x and y style
ax.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
ax.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
ax.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))

surf = ax.plot_surface(X, Y, z, cmap='bone', edgecolor='k', linewidth=0.5, antialiased=True)

font_properties = {'fontweight': 'bold', 'fontsize': 12}
ax.set_xlabel('μ', **font_properties)
ax.set_ylabel('γ', **font_properties)
ax.set_zlabel('Dice score (%)', **font_properties)
ax.set_title('SSL-DG with different μ and γ', fontweight='bold', fontsize=14)


ax.set_xticks(x)
ax.set_yticks(y)


cbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)
cbar.set_label('Dice score (%)', **font_properties)


ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)


plt.tight_layout()
plt.show()
