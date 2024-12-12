import argparse
import cv2
import sys
import time
from scipy.optimize import minimize
from structs import *


def compute_specific_center_point(
    mesh: Mesh, inside_points, num_nearest_neighbour: int, vertex_index: int
):
    vertex = mesh.vertices[vertex_index]
    dist = np.linalg.norm(inside_points - vertex, axis=1)
    idx = np.argsort(dist)[:num_nearest_neighbour]
    center = np.mean(inside_points[idx], axis=0)
    return center


def compute_center_points(mesh: Mesh, inside_points, num_nearest_neighbour: int):
    center_points = []
    for l in range(mesh.vertex_num()):
        center_points.append(
            compute_specific_center_point(mesh, inside_points, num_nearest_neighbour, l)
        )
    return np.array(center_points)


def find_nearest(mesh: Mesh, point):
    dist = np.linalg.norm(mesh.vertices - point, axis=1)
    parent = np.argmin(dist)
    distance = dist[parent]
    return (parent, distance)


def compute_center_points_unique(
    mesh: Mesh, inside_points, num_nearest_neighbours: int
):
    center_points = []
    included_points = [[] for _ in range(mesh.vertex_num())]

    for pt in inside_points:
        res = find_nearest(mesh, pt)
        included_points[res[0]].append(PointDist(pt, res[1]))

    for i in range(mesh.vertex_num()):
        sorted_inc_points = np.sort(included_points[i])[:num_nearest_neighbours]
        if len(sorted_inc_points) > 0:
            points = np.array([pt.point for pt in sorted_inc_points])
            center = np.sum(points, axis=0)
            center = center / float(num_nearest_neighbours)
            center_points.append(center)
        else:
            center_points.append(np.array([0.0, 0.0, 0.0]))
    return np.array(center_points)


def compute_outside_points(mesh: Mesh, points):
    triangles = mesh.construct_triangles()
    is_inside = mesh.is_inside_batch(triangles, points)
    inside_points = np.array([points[i] for i in range(len(points)) if is_inside[i]])
    outside_points = np.array(
        [points[i] for i in range(len(points)) if not is_inside[i]]
    )
    return inside_points, outside_points


def outside_points_distance(mesh: Mesh, outside_points) -> float:
    nearest_points = np.array(
        [
            nearest_point(outside_points[i], mesh.faces, mesh.vertices)
            for i in range(len(outside_points))
        ]
    )
    distances = np.linalg.norm(nearest_points - outside_points, axis=1)
    return np.sum(distances)


def compute_loss_function_base(
    mesh: Mesh, outside_points, num_total_points, center_nn_points, lambda_fac
):
    reconstruct_error = outside_points_distance(mesh, outside_points) / num_total_points
    represent_error = np.sum(np.linalg.norm(mesh.vertices - center_nn_points, axis=1))
    print(mesh.vertices - center_nn_points)
    print(np.linalg.norm(mesh.vertices - center_nn_points, axis=1))
    represent_error /= mesh.vertex_num()
    total_error = reconstruct_error * lambda_fac + represent_error
    return Result(lambda_fac, reconstruct_error, represent_error, total_error)


def compute_loss_function(
    mesh: Mesh,
    points,
    lambda_fac: float,
    num_nearest_neighbours: int,
    option: int,
    unique: int,
):
    inside_points, outside_points = compute_outside_points(mesh, points)

    if unique == 1:
        if option == 1:
            n = min(num_nearest_neighbours, len(inside_points))
            center_points = compute_center_points_unique(mesh, inside_points, n)
        else:
            center_points = compute_center_points_unique(
                mesh, points, num_nearest_neighbours
            )
    else:
        if option == 1:
            n = min(num_nearest_neighbours, len(inside_points))
            center_points = compute_center_points(mesh, inside_points, n)
        else:
            center_points = compute_center_points(mesh, points, num_nearest_neighbours)

    return compute_loss_function_base(
        mesh, outside_points, len(points), center_points, lambda_fac
    )


def cost_function(x, data: Data) -> float:
    target_dist = x[0]
    target_point = (1 - target_dist) * data.center_point[
        data.index
    ] + target_dist * data.mesh.vertices[data.index]

    new_mesh = data.mesh
    new_mesh.vertices[data.index] = target_point

    inside_points, outside_points = compute_outside_points(new_mesh, data.points)
    num_nearest_neighbours = data.num_nearest_neighbours
    center_points = data.center_point
    
    print("before")
    print(center_points[data.index])
    print("after")

    if data.unique == 1:
        if data.center_point_option == 1:
            n = min(num_nearest_neighbours, len(inside_points))
            print(n)
            center_points[data.index] = compute_center_points_unique(
                new_mesh, inside_points, n
            )[data.index]
            print(center_points[data.index])
        else:
            center_points[data.index] = compute_center_points_unique(
                new_mesh, data.points, num_nearest_neighbours
            )[data.index]
            print(center_points[data.index])
    else:
        if data.center_point_option == 1:
            n = min(num_nearest_neighbours, len(inside_points))
            print(n)
            center_points[data.index] = compute_specific_center_point(
                new_mesh, inside_points, n, data.index
            )
            print(center_points[data.index])
        else:
            center_points[data.index] = compute_specific_center_point(
                new_mesh, data.points, num_nearest_neighbours, data.index
            )
            print(center_points[data.index])

    result = compute_loss_function_base(
        new_mesh,
        outside_points,
        len(data.points),
        center_points,
        data.lambda_fac,
    )
    return result.total_error


if __name__ == "__main__":
    mesh = Mesh()
    mesh.load_from_file("../../test_mesh.obj")
    filename = "../../test_vertices.obj"
    with open(filename, "r") as infile:
        lines = infile.readlines()
    lines = [line.split() for line in lines]
    points = np.array((0, 3), dtype=np.float32)
    points = np.empty((0, 3), dtype=np.float32)
    for l in lines:
        if l[0] == "v":
            points = np.vstack([points, np.array(l[1:]).astype(float)])
    num_nearest_neighbours = 40

    for unique in range(2):
        for option in range(2):
            if unique == 1:
                centerpoints = compute_center_points_unique(
                    mesh, points, num_nearest_neighbours
                )
            else:
                centerpoints = compute_center_points(
                    mesh, points, num_nearest_neighbours
                )
            data = Data(
                centerpoints,
                0,
                mesh,
                points,
                20,
                num_nearest_neighbours,
                option,
                unique,
            )
            cost = cost_function([0.5], data)
            print("unique: ", unique)
            print("option: ", option)
            print("cost: ", cost)