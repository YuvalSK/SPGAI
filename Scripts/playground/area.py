import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

params = {'axes.titlesize': 30,
          'legend.fontsize': 16,
          'figure.figsize': (16, 10),
          'axes.labelsize': 16,
          'axes.titlesize': 12,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'figure.titlesize': 30}

plt.rcParams.update(params)
plt.style.use('seaborn-v0_8-whitegrid')

# Load the data
danenberg = pd.read_csv('Results/Danenberg/raw_area_stats_all_danenberg2022.csv')
jackson = pd.read_csv('Results/Jackson/raw_area_stats_all_jackson2020.csv')

# Add dataset labels
danenberg['Dataset'] = 'Danenberg 2022'
jackson['Dataset'] = 'Jackson 2020'

# Order markers by median for each dataset (high to low)
danenberg_order = danenberg.sort_values('raw_area_median_%', ascending=False)['marker'].tolist()
jackson_order = jackson.sort_values('raw_area_median_%', ascending=False)['marker'].tolist()

# Calculate IQR for each marker
danenberg['IQR_%'] = danenberg['raw_area_q75_%'] - danenberg['raw_area_q25_%']
jackson['IQR_%'] = jackson['raw_area_q75_%'] - jackson['raw_area_q25_%']

# Create figure with two subplots (one per dataset)
fig, axes = plt.subplots(1, 2)

# Plot Danenberg
ax = axes[0]
y_pos = np.arange(len(danenberg_order))

for i, marker in enumerate(danenberg_order):
    row = danenberg[danenberg['marker'] == marker].iloc[0]
    median = row['raw_area_median_%']
    q25 = row['raw_area_q25_%']
    q75 = row['raw_area_q75_%']
    
    # Draw box (Q25 to Q75)
    box_width = q75 - q25
    ax.barh(i, box_width, left=q25, height=0.6, 
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Draw median line
    ax.plot([median, median], [i-0.3, i+0.3], 'k-', linewidth=3)
    
    # Draw whiskers (assuming mean ± std as whiskers, outliers beyond)
    mean = row['raw_area_mean_%']
    std = row['raw_area_std_%']
    lower_whisker = max(0, mean - std)
    upper_whisker = min(100, mean + std)
    
    # Whisker lines
    ax.plot([lower_whisker, q25], [i, i], 'k-', linewidth=1.5)
    ax.plot([q75, upper_whisker], [i, i], 'k-', linewidth=1.5)
    ax.plot([lower_whisker, lower_whisker], [i-0.1, i+0.1], 'k-', linewidth=1.5)
    ax.plot([upper_whisker, upper_whisker], [i-0.1, i+0.1], 'k-', linewidth=1.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(danenberg_order, fontsize=10)
ax.set_xlabel('% Area Coverage', fontsize=12, fontweight='bold')
ax.set_ylabel('Marker', fontsize=12, fontweight='bold')
ax.set_title('Danenberg 2022', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Plot Jackson
ax = axes[1]
y_pos = np.arange(len(jackson_order))

for i, marker in enumerate(jackson_order):
    row = jackson[jackson['marker'] == marker].iloc[0]
    median = row['raw_area_median_%']
    q25 = row['raw_area_q25_%']
    q75 = row['raw_area_q75_%']
    
    # Draw box (Q25 to Q75)
    box_width = q75 - q25
    ax.barh(i, box_width, left=q25, height=0.6, 
            color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Draw median line
    ax.plot([median, median], [i-0.3, i+0.3], 'k-', linewidth=3)
    
    # Draw whiskers
    mean = row['raw_area_mean_%']
    std = row['raw_area_std_%']
    lower_whisker = max(0, mean - std)
    upper_whisker = min(100, mean + std)
    
    # Whisker lines
    ax.plot([lower_whisker, q25], [i, i], 'k-', linewidth=1.5)
    ax.plot([q75, upper_whisker], [i, i], 'k-', linewidth=1.5)
    ax.plot([lower_whisker, lower_whisker], [i-0.1, i+0.1], 'k-', linewidth=1.5)
    ax.plot([upper_whisker, upper_whisker], [i-0.1, i+0.1], 'k-', linewidth=1.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(jackson_order, fontsize=10)
ax.set_xlabel('% Area Coverage', fontsize=12, fontweight='bold')
ax.set_ylabel('Marker', fontsize=12, fontweight='bold')
ax.set_title('Jackson 2020', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('marker_area_distribution_by_dataset.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics table
print("\n" + "="*110)
print("DANENBERG 2022 - SUMMARY STATISTICS (Ordered High to Low)")
print("="*110)
print(f"\n{'Marker':<30} {'Median (%)':<12} {'IQR (%)':<12} {'Q25 (%)':<12} {'Q75 (%)':<12} {'Mean (%)':<12} {'Std (%)':<12}")
print("-"*110)

for marker in danenberg_order:
    row = danenberg[danenberg['marker'] == marker].iloc[0]
    print(f"{marker:<30} {row['raw_area_median_%']:<12.2f} {row['IQR_%']:<12.2f} {row['raw_area_q25_%']:<12.2f} {row['raw_area_q75_%']:<12.2f} {row['raw_area_mean_%']:<12.2f} {row['raw_area_std_%']:<12.2f}")

print("\n" + "="*110)
print("JACKSON 2020 - SUMMARY STATISTICS (Ordered High to Low)")
print("="*110)
print(f"\n{'Marker':<30} {'Median (%)':<12} {'IQR (%)':<12} {'Q25 (%)':<12} {'Q75 (%)':<12} {'Mean (%)':<12} {'Std (%)':<12}")
print("-"*110)

for marker in jackson_order:
    row = jackson[jackson['marker'] == marker].iloc[0]
    print(f"{marker:<30} {row['raw_area_median_%']:<12.2f} {row['IQR_%']:<12.2f} {row['raw_area_q25_%']:<12.2f} {row['raw_area_q75_%']:<12.2f} {row['raw_area_mean_%']:<12.2f} {row['raw_area_std_%']:<12.2f}")

print("\n✓ Analysis complete! Plot saved as 'marker_area_distribution_by_dataset.png'")