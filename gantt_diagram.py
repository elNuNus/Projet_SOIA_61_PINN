import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Define the task names and their start and end dates
tasks = {
    "Bibliographic Work": ["2024-09-01", "2024-10-10"],
    "Test 1D Heat Equation (Stationary)": ["2024-10-01", "2025-01-20"],
    "Test 1D Heat Equation (With Time)": ["2025-01-21", "2025-03-02"],
    "Test 2D Heat Equation": ["2025-02-10", "2025-02-25"]
}

# Convert to pandas DataFrame
df = pd.DataFrame(tasks).T
df.columns = ["Start", "End"]
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = pd.to_datetime(df["End"])

# Create the Gantt chart
fig, ax = plt.subplots(figsize=(10, 6))

for idx, (task, row) in enumerate(df.iterrows()):
    ax.barh(task, row["End"] - row["Start"], left=row["Start"], height=0.6, color='skyblue')

# Format the chart
ax.set_xlabel("Timeline")
ax.set_ylabel("Tasks")
ax.set_title("Gantt Chart: PINN Project")
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
