#!python
import re
import matplotlib.pyplot as plt
import numpy as np

# Simulating reading the file content
# Replace 'file_content' with: with open('rfc_output_SMOTE.txt', 'r') as f: file_content = f.read()
file_content = """
Metrics on New Data:
  AUC:       0.5000
  Recall:    0.0000
  Precision: 0.0000
  F1-Score:  0.0000
  Accuracy:  0.7419

Metrics on New Data:
  AUC:       0.5000
  Recall:    1.0000
  Precision: 0.2581
  F1-Score:  0.4103
  Accuracy:  0.2581
"""

# Regex to extract metrics blocks
pattern = r"Metrics on New Data:\s+AUC:\s+([\d\.]+)\s+Recall:\s+([\d\.]+)\s+Precision:\s+([\d\.]+)\s+F1-Score:\s+([\d\.]+)\s+Accuracy:\s+([\d\.]+)"
matches = re.findall(pattern, file_content)

# Data structure to hold lists of scores
data = {
    "AUC": [],
    "Recall": [],
    "Precision": [],
    "F1-Score": [],
    "Accuracy": []
}

# Parse matches
for match in matches:
    data["AUC"].append(float(match[0]))
    data["Recall"].append(float(match[1]))
    data["Precision"].append(float(match[2]))
    data["F1-Score"].append(float(match[3]))
    data["Accuracy"].append(float(match[4]))

# Plotting
labels = [f"Exp {i+1}" for i in range(len(matches))]
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Generate bars for each metric
rects1 = ax.bar(x - 2*width, data["F1-Score"], width, label='F1-Score')
rects2 = ax.bar(x - width, data["AUC"], width, label='AUC')
rects3 = ax.bar(x, data["Recall"], width, label='Recall')
rects4 = ax.bar(x + width, data["Precision"], width, label='Precision')
rects5 = ax.bar(x + 2*width, data["Accuracy"], width, label='Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance Metrics from rfc_output_SMOTE.txt')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)  # Set y-axis limit to accommodate 0-1 range
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()

plt.show()
