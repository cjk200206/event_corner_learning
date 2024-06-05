import matplotlib.pyplot as plt
import os


output_path = "/home/cjk2002/code/event_code/event_corner_learning/log"
voxel_path = "/home/cjk2002/code/event_code/event_corner_learning/log/event_superpoint/data.txt"
sae_path = "/home/cjk2002/code/event_code/event_corner_learning/log/event_superpoint_sae/data.txt"

# 示例数据
threshold_list = [2,3,4,5,6,7,8,9]
precisions = []
recalls = []
Repeatabilities = []
LEs = []
NNmAPs = []

precisions_sae = []
recalls_sae = []
Repeatabilities_sae = []
LEs_sae = []
NNmAPs_sae = []

with open(voxel_path,"r+") as file:
    for line in file:
        parts = line.strip().split('\t')
        recalls.append(float(parts[1]))
        precisions.append(float(parts[2]))
        LEs.append(float(parts[3]))
        Repeatabilities.append(float(parts[4]))
        NNmAPs.append(float(parts[5]))

with open(sae_path,"r+") as file:
    for line in file:
        parts = line.strip().split('\t')
        parts_sae = line.strip().split('\t')
        recalls_sae.append(float(parts[1]))
        precisions_sae.append(float(parts[2]))
        LEs_sae.append(float(parts[3]))
        Repeatabilities_sae.append(float(parts[4]))
        NNmAPs_sae.append(float(parts[5]))

plt.figure(figsize=(8, 6))
plt.plot(threshold_list, precisions, marker='o', linestyle='-',label="voxel",color="blue")
plt.plot(threshold_list, precisions_sae, marker='o', linestyle='-',label="sae",color="red")
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.title('Corner Detection Average Precision Curve')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path,"precision.png"))

plt.figure(figsize=(8, 6))
plt.plot(threshold_list, recalls, marker='o', linestyle='-',label="voxel",color="blue")
plt.plot(threshold_list, recalls_sae, marker='o', linestyle='-',label="sae",color="red")
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.title('Corner Detection Recall Curve')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path,"recall.png"))

plt.figure(figsize=(8, 6))
plt.plot(threshold_list, Repeatabilities, marker='o', linestyle='-',label="voxel",color="blue")
plt.plot(threshold_list, Repeatabilities_sae, marker='o', linestyle='-',label="sae",color="red")
plt.xlabel('Threshold')
plt.ylabel('Repeatability')
plt.title('Repeatability Curve')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path,"Repeatability.png"))

plt.figure(figsize=(8, 6))
plt.plot(threshold_list, LEs, marker='o', linestyle='-',label="voxel",color="blue")
plt.plot(threshold_list, LEs_sae, marker='o', linestyle='-',label="sae",color="red")
plt.xlabel('Threshold')
plt.ylabel('Localization Error')
plt.title('Localization Error Curve')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path,"Localization_Error.png"))

plt.figure(figsize=(8, 6))
plt.plot(threshold_list, NNmAPs, marker='o', linestyle='-',label="voxel",color="blue")
plt.plot(threshold_list, NNmAPs_sae, marker='o', linestyle='-',label="sae",color="red")
plt.xlabel('Threshold')
plt.ylabel('NNmAP')
plt.title('NNmAP Curve')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path,"NNmAP.png"))
