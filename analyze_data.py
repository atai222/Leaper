import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your final dataset
df = pd.read_csv('training_data.csv')

# Create a scatter plot to see the relationship between entry velocity and distance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='entry_velocity_mps', y='distance_m', hue='phase_type', s=100)
plt.title('Entry Velocity vs. Phase Distance')
plt.xlabel('Entry Velocity (m/s)')
plt.ylabel('Phase Distance (m)')
plt.grid(True)
plt.savefig('reports/velocity_vs_distance.png')
print("Saved plot to reports/velocity_vs_distance.png")

# Create a plot for contact time vs. distance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='contact_time_s', y='distance_m', hue='phase_type', s=100)
plt.title('Ground Contact Time vs. Phase Distance')
plt.xlabel('Contact Time (seconds)')
plt.ylabel('Phase Distance (m)')
plt.grid(True)
plt.savefig('reports/contact_time_vs_distance.png')
print("Saved plot to reports/contact_time_vs_distance.png")