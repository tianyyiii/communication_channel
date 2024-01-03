import diffcloth_py as diffcloth
from pySim.pySim import pySim
from pySim.functional import SimFunction
import common

import numpy as np
import torch
import trimesh
import random
import time
import json
import os 

def step(x, v, a, simModule):
    x1, v1 = simModule(x, v, a)
    return x1, v1

def sample_gaussian_noise(mean, std):
    target_pose = np.random.normal(mean, std, [3])
    
    
def set_sim(example):
    sim = diffcloth.makeSim(example)
    sim.resetSystem()
    stateInfo = sim.getStateInfo()
    x0 = stateInfo.x
    v0 = stateInfo.v
    x0 = torch.tensor(x0, requires_grad=True)
    v0 = torch.tensor(v0, requires_grad=True)
    return sim, x0, v0


def calculate_jacobian(x0, v0, a0, keypoints):
    total_forward_time = 0
    forward_iteration = 0
    total_backward_time = 0
    backward_iteration = 0
    jacobian = torch.zeros((20 * 3, a0.shape[0]))
    for i, keypoint in enumerate(keypoints):
        for axis in range(3):
            a00 = a0.clone().detach()
            a00.requires_grad = True
            time_start = time.time()
            x1, v1 = step(x0.clone().detach(), v0.clone().detach(), a00, pysim)
            time_end = time.time()
            total_forward_time += (time_end - time_start)
            forward_iteration += 1
            loss = x1[keypoint * 3 + axis]
            time_start = time.time()
            loss.backward()
            time_end = time.time()
            total_backward_time += (time_end - time_start)
            backward_iteration += 1
            jacobian[i * 3 + axis, :] = a00.grad
    print("forward_time", total_forward_time/forward_iteration)
    print("backward_time", total_backward_time/backward_iteration)
    return jacobian


def process_mesh(clothes_path, save_path):
    for cloth in os.listdir(clothes_path):
        cloth_mesh = clothes_path + "/" + cloth + "/" + cloth + ".obj"
        scene = trimesh.load_mesh(cloth_mesh)
        try: 
            mesh = scene.dump()[1]
        except:
            mesh = scene
        connected_components = mesh.split(only_watertight=False)
        print(len(connected_components))
        if len(connected_components) == 1:
            print(cloth)
            break
    vertices = mesh.vertices
    vertices[:, [1,2]] = vertices[:, [2, 1]]
    mesh.vertices = vertices
    min_bound, max_bound = mesh.bounding_box.bounds
    scale = max(max_bound - min_bound)
    vertices = vertices/scale * 4 
    mesh.vertices = vertices 
    min_bound, max_bound = mesh.bounding_box.bounds
    average = (min_bound + max_bound) / 2 
    for i in range(3):
        vertices[:, i] -= average[i]
    mesh.vertices = vertices
    min_bound, max_bound = mesh.bounding_box.bounds
    vertices[:, 1] -= (min_bound[1]-(-5.65))
    mesh.vertices = vertices
    #if len(mesh.vertices) > 2048:
    #    mesh = mesh.simplify_quadratic_decimation(2048)
    mesh.export(save_path)

if __name__ == "__main__":
    #process_mesh("../../../ClothesNetData/ClothesNetM/Tops/NoCollar_Lsleeve_FrontClose", "top.obj")
    torch.set_printoptions(precision=8)
    common.setRandomSeed(1349)
    example = "wear_hat"
    experiment_index = 0
    sim, x0, v0 = set_sim(example)
    position = sim.getStateInfo().x_fixedpoints
    helper = diffcloth.makeOptimizeHelper(example)
    pysim = pySim(sim, helper, True)
    
    keypoints = random.sample(range(579), 20)
    
    x, v = x0, v0
    
    a = sim.getStateInfo().x_fixedpoints
    a = torch.tensor(a)
    calculate_jacobian(x, v, a, keypoints)
    
    time_total = 0
    for i in range(300):
        a = sim.getStateInfo().x_fixedpoints
        a = torch.tensor(a)
        time_start = time.time()
        x, v = step(x, v, a, pysim)
        time_end = time.time()
        time_total += (time_end - time_start)
        y_mean = sum(x[1:-1:3])/len(x[1:-1:3])
        print(y_mean)
    sim.exportCurrentSimulation("hat")
    print("time_forward_with_collision", time_total/300)
        
    # a = x[392*3: 392*3+3]
    # for i in range(5):
    #     x, v = step(x, v, a, pysim)
    # sim.exportCurrentSimulation("fold")
    x_start = x.clone()
    
    data = []
    for index in range(1):
        sequence = dict()
        x_now = x_start.detach().numpy().tolist()
        points = [x_now[i:i + 3] for i in range(0, len(x_now), 3)]
        sequence["init_state"] = points
        sequence["target_state"] = points
        attached_points = sim.getStateInfo().x_fixedpoints.tolist()
        attached_points = [attached_points[i:i + 3] for i in range(0, len(attached_points), 3)]
        sequence["attached_point"] = attached_points
        a = a.detach().numpy().tolist()
        a = [a[i:i + 3] for i in range(0, len(a), 3)]
        sequence["attached_point_target"] = a
        sequence["response_matrix"] = np.zeros((len(x_start), 3 * len(keypoints))).tolist()
        for i in sequence:
            print(np.array(sequence[i]).shape)

        data.append(sequence)
    with open('data_demo.json', 'w') as json_file:
        json.dump(data, json_file)

    
    
    
    