import sys
import os
import argparse
from unittest import result

#sys.path.insert(0, "/root/autodl-tmp/DiffCloth_manimodel/pylib")
sys.path.insert(0, os.path.abspath("./pylib"))

import contextlib
import io
import json
import time
import random
import open3d as o3d
import trimesh
import torch
import numpy as np
import tqdm
import common
from pySim.functional import SimFunction
from pySim.pySim import pySim, pySimF
import diffcloth_py as diffcloth
import trajectory
import jacobian
import gc
import math

from renderer import WireframeRenderer
import pywavefront




CONFIG = {
    'fabric': {
        "clothDimX": 6,
        "clothDimY": 6,
        "k_stiff_stretching": 550,
        "k_stiff_bending":  0.01,
        "gridNumX": 40,
        "gridNumY": 80,
        "density": 1, #1
        "keepOriginalScalePoint": False,
        'isModel': True,
        "custominitPos": False,
        "fabricIdx": 2,  # Enum Value
        "color": (0.3, 0.9, 0.3),
        "name":  "remeshed/top.obj",
    },
    'scene': {
        "orientation": 1, #1 # Enum Value
        "attachmentPoints": 2,  # CUSTOM_ARRAY
        "customAttachmentVertexIdx": [(0., [])],
        "trajectory": 0,  # Enum Value
        "primitiveConfig": 3,  # Enum Value
        'windConfig': 0,  # Enum Value
        'camPos':  (-10.38, 4.243, 12.72),
        "camFocusPos": (0, -4, 0),
        'camFocusPointType': 3,  # Enum Value
        "sceneBbox":  {"min": (-7, -7, -7), "max": (7, 7, 7)},
        "timeStep": 1.0 / 30.0, #1/90
        "stepNum": 250,
        "forwardConvergenceThresh": 1e-8,
        'backwardConvergenceThresh': 5e-4,
        'name': "wind_tshirt"
    }
}


def set_sim_from_config(config):
    sim = diffcloth.makeSimFromConfig(config)
    sim.resetSystem()
    stateInfo = sim.getStateInfo()
    x0 = stateInfo.x
    v0 = stateInfo.v
    x0 = torch.tensor(x0, requires_grad=True)
    v0 = torch.tensor(v0, requires_grad=True)
    return sim, x0, v0


def step(x, v, a, simModule):
    x1, v1 = simModule(x, v, a)
    return x1, v1

def stepF(x, v, a, f, simModule):
    x1, v1 = simModule(x, v, a, f)
    return x1, v1


def mnormal(mesh, x_pos):
    shape = trimesh.load(mesh)
    try:
        vertices = shape.vertices
    except:
        shape = mesh.dump()[1]
        vertices = shape.vertices
    shape.vertices = x_pos
    return shape.vertex_normals


def sample_gaussian(variance=0.25):
    mean = [0, 0, 0]  
    covariance = [[variance, 0, 0],  
                [0, variance, 0],
                [0, 0, variance]]
    sample = np.random.multivariate_normal(mean, covariance)
    return sample


def read_mesh_ignore_vtvn(mesh_file):
    pos_vec = []  # 存储顶点位置
    tri_vec = []  # 存储三角形面

    with open(mesh_file, "r") as file:
        for line in file:
            tokens = line.split()
            if not tokens or tokens[0].startswith("#"):
                continue

            if tokens[0] == "v":  # 顶点
                x, y, z = map(float, tokens[1:4])
                pos_vec.append((x, y, z))

            elif tokens[0] == "f":  # 面
                # 仅处理每个面的顶点索引，忽略可能的纹理和法线索引
                vertex_indices = [
                    int(face.partition("/")[0]) - 1 for face in tokens[1:4]
                ]
                tri_vec.append(vertex_indices)

    print("load mesh: ", mesh_file, " with ", len(pos_vec),
          " vertices and ", len(tri_vec), " faces")
    return np.array(pos_vec), np.array(tri_vec)


def get_keypoints(mesh_file, kp_file):
    # mesh = o3d.io.read_triangle_mesh(mesh_file)
    # mesh_vertices = np.asarray(mesh.vertices)
    mesh_vertices, _ = read_mesh_ignore_vtvn(mesh_file)

    pcd = o3d.io.read_point_cloud(kp_file)
    pcd_points = np.asarray(pcd.points)

    indices = []
    for point in pcd_points:
        distances = np.sqrt(np.sum((mesh_vertices - point) ** 2, axis=1))
        nearest_vertex_index = np.argmin(distances)
        indices.append(nearest_vertex_index)

    return indices


def get_coord_by_idx(x, idx):
    return x[idx*3:(idx+1)*3]


def cubic_bezier(p0, p1, p2, p3, t):
    """Calculate a point on a cubic Bezier curve with given control points at parameter t."""
    return (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3


def create_bent_curve(p0, p3, bend_factor=0.5, num_points=100):
    """Create a curve that bends towards the z-axis and passes through p0 and p3."""
    # Calculate control points for bending
    p1 = p0 + np.array([0, bend_factor, 0])
    p2 = p3 + np.array([0, bend_factor, 0])

    # Generate points along the curve
    t_values = np.linspace(0, 1, num_points)
    curve_points = np.array([cubic_bezier(p0, p1, p2, p3, t)
                            for t in t_values])

    return curve_points


def render_record(sim, kp_idx=None, curves=None):
    renderer = WireframeRenderer(backend="pyglet")

    forwardRecords = sim.forwardRecords

    mesh_vertices = forwardRecords[0].x.reshape(-1, 3)
    mesh_faces = np.array(diffcloth.getSimMesh(sim))
    x_records = [forwardRecords[i].x.reshape(-1, 3)
                 for i in range(len(forwardRecords))]

    renderer.add_mesh(mesh_vertices, mesh_faces, x_records)
    if kp_idx is not None:
        renderer.add_kp(mesh_vertices, kp_idx)

    if curves is not None:
        for c in curves:
            renderer.add_curve(c)

    renderer.show()
    renderer.run()
    



def main(mesh_path="../../python_code/DLG_Dress032_1.obj", result_path="/home/ubuntu/result"):
    contact_map_path = result_path + "/" + "contact_heatmap.npy"
    contact_force_path = result_path + "/" + "contact_force.npy"
    state_path = result_path + "/" + "DLG_Dress032_1.npz"
    contact_map = np.load(contact_map_path)
    contact_force = np.load(contact_force_path)
    state = np.load(state_path, allow_pickle=True)
    kp_idx = state["kp_idx"].tolist()
    state_data = state["data"]
    contact_map = contact_map[:, :, :, :2]
    contact_force = contact_force[:, :, :2, :]
    contact_point = np.argmax(contact_map, axis=2)
    config = CONFIG.copy()
    config['fabric']['name'] = mesh_path
    score = []
    start_time = time.time()
    for t in range(10):
        for trial in range(contact_force.shape[1]):
            attached_points = [contact_point[t, trial, 0], contact_point[t, trial, 1]]
            config['scene']['customAttachmentVertexIdx'] = [(0.0, [contact_point[t, trial, 0], contact_point[t, trial, 1]])]
            sim_init, x, v = set_sim_from_config(config)
            v = v * 0
            init_step = math.floor((t % 440)/20)
            init_state = state_data[init_step]["init_state"].flatten()
            init_kp_state = state_data[init_step]["init_state"][kp_idx].flatten()
            x = torch.tensor(init_state)
            helper_init = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim_init)
            pysim_init = pySim(sim_init, helper_init, True)
            target_state = state_data[init_step]["target_state"][(t % 440) % 20, :, :].flatten()
            
            min_loss = np.inf
            for i in range(30):
                a_delta = 0.01 * contact_force[t, trial, :, :].flatten() / np.linalg.norm(contact_force[t, trial, :, :].flatten())
                a_delta = torch.tensor(a_delta)
                a_0 = (x.view(-1, 3))[attached_points].flatten()
                a = a_0 + a_delta
                x, v = step(x, v, a, pysim_init)
                x_pos = (x.clone().detach().view(-1, 3)[kp_idx]).numpy()
                loss = np.linalg.norm(target_state - x_pos.flatten())
                if loss < min_loss:
                    min_loss = loss
            print("----------------")
            
            gt_point = state_data[init_step]["attached_point"][(t % 440) % 20, :]
            gt_target = state_data[init_step]["attached_point_target"][(t % 440) % 20, :, :]
            
            attached_points = [gt_point[0], gt_point[1]]
            config['scene']['customAttachmentVertexIdx'] = [(0.0, [gt_point[0], gt_point[1]])]
            sim_gt, x_gt, v_gt = set_sim_from_config(config)
            v_gt = v_gt * 0
            x_gt = torch.tensor(init_state)
            helper_gt = diffcloth.makeOptimizeHelperWithSim("wear_hat", sim_gt)
            pysim_gt = pySim(sim_gt, helper_gt, True)
            

            a_0 = (x_gt.clone().view(-1, 3))[attached_points].flatten()
            a_delta_gt = gt_target.flatten() - a_0.clone().detach().numpy()
            a_delta_gt = 0.01 * a_delta_gt / np.linalg.norm(a_delta_gt)
            a_delta_gt = torch.tensor(a_delta_gt)
            min_loss_gt = np.inf
            x_gt, v_gt = x_gt.clone(), v_gt.clone()
            for i in range(20):
                a_0 = (x_gt.view(-1, 3))[attached_points].flatten()
                a_gt = a_0 + a_delta_gt
                x_gt, v_gt = step(x_gt, v_gt, a_gt, pysim_gt)
                x_pos_gt = (x_gt.clone().detach().view(-1, 3)[kp_idx]).numpy()
                loss_gt = np.linalg.norm(target_state - x_pos_gt.flatten())
                if loss_gt < min_loss_gt:
                    min_loss_gt = loss_gt
            print("----------------")
            print(min_loss)
            print(min_loss_gt, "gt")
            print(np.linalg.norm(target_state - init_kp_state))
            #score.append(min_loss/np.linalg.norm(target_state - init_kp_state))
            score.append(min_loss)
    print(sum(score)/len(score))
    print((time.time()-start_time)/150)
            
                    
            #render_record(sim_init)

        






if __name__ == '__main__':
    main()