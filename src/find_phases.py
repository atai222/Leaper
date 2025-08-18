import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('triple_jump_analysis.csv')

# Plot the vertical position of both ankles
plt.figure(figsize=(15, 6))
plt.plot(df['frame'], df['LEFT_ANKLE_y'], label='Left Ankle Y')
plt.plot(df['frame'], df['RIGHT_ANKLE_y'], label='Right Ankle Y')

# Invert the y-axis because in image coordinates, 0 is the top
plt.gca().invert_yaxis() 

plt.title('Ankle Vertical Position vs. Frame')
plt.xlabel('Frame Number')
plt.ylabel('Vertical Position (Ankle)')
plt.legend()
plt.grid(True)
plt.show()