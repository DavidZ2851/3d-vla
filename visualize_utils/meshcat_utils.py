import numpy as np
import trimesh.transformations as tra


import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf


"""
Some code borrowed from https://github.com/google-research/ravens
under Apache license
"""


def isRotationMatrix(M, tol=1e-4):
    tag = False
    I = np.identity(M.shape[0])

    if (np.linalg.norm((np.matmul(M, M.T) - I)) < tol) and (
        np.abs(np.linalg.det(M) - 1) < tol
    ):
        tag = True

    if tag is False:
        print("M @ M.T:\n", np.matmul(M, M.T))
        print("det:", np.linalg.det(M))

    return tag


def trimesh_to_meshcat_geometry(mesh):
    """
    Args:
        mesh: trimesh.TriMesh object
    """

    return meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces)


def visualize_mesh(vis, name, mesh, color=None, transform=None):
    """Visualize a mesh in meshcat"""

    if color is None:
        color = np.random.randint(low=0, high=256, size=3)

    mesh_vis = trimesh_to_meshcat_geometry(mesh)
    color_hex = rgb2hex(tuple(color))
    material = meshcat.geometry.MeshPhongMaterial(color=color_hex)
    vis[name].set_object(mesh_vis, material)

    if transform is not None:
        vis[name].set_transform(transform)


def rgb2hex(rgb):
    """
    Converts rgb color to hex

    Args:
        rgb: color in rgb, e.g. (255,0,0)
    """
    return "0x%02x%02x%02x" % (rgb)


def visualize_scene(vis, object_dict, randomize_color=True, visualize_transforms=False):

    for name, data in object_dict.items():

        # try assigning a random color
        if randomize_color:
            if "color" in data:
                color = data["color"]

                # if it's not an integer, convert it to [0,255]
                if not np.issubdtype(color.dtype, np.int):
                    color = (color * 255).astype(np.int32)
            else:
                color = np.random.randint(low=0, high=256, size=3)
                data["color"] = color
        else:
            color = [0, 255, 0]

        # mesh_vis = trimesh_to_meshcat_geometry(data['mesh_transformed'])
        mesh_vis = trimesh_to_meshcat_geometry(data["mesh"])
        color_hex = rgb2hex(tuple(color))
        material = meshcat.geometry.MeshPhongMaterial(color=color_hex)

        mesh_name = f"{name}/mesh"
        vis[mesh_name].set_object(mesh_vis, material)
        vis[mesh_name].set_transform(data["T_world_object"])

        if visualize_transforms:
            frame_name = f"{name}/transform"
            make_frame(vis, frame_name, T=data["T_world_object"])


def create_visualizer(clear=True):
    print(
        "Waiting for meshcat server... have you started a server? Run `meshcat-server` to start a server"
    )
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6002").open()
    print(vis.url())
    if clear:
        vis.delete()
    return vis


def visualize_triad(vis, name, h=0.15, radius=0.001, o=1.0, T=None):
    """Add a red-green-blue triad to the Meschat visualizer.
    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]["x"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8, opacity=o),
    )
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]["x"].set_transform(rotate_x)

    vis[name]["y"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8, opacity=o),
    )
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]["y"].set_transform(rotate_y)

    vis[name]["z"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000FF, reflectivity=0.8, opacity=o),
    )
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]["z"].set_transform(rotate_z)

    if T is not None:
        is_valid = isRotationMatrix(T[:3, :3])

        if not is_valid:
            raise ValueError("meshcat_utils:attempted to visualize invalid transform T")

        vis[name].set_transform(T.astype(np.float64))


def draw_grasp(
    vis, line_name, transform, h=0.15, radius=0.001, o=1.0, color=[255, 0, 0]
):
    """Draws line to the Meshcat visualizer.
    Args:
      vis (Meshcat Visualizer): the visualizer
      line_name (string): name for the line associated with the grasp.
      transform (numpy array): 4x4 specifying transformation of grasps.
      radius (float): radius of frame visualization
      o (float): opacity
      color (list): color of the line.
    """
    vis[line_name].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=rgb2hex(tuple(color)), reflectivity=0.8, opacity=o),
    )
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[line_name].set_transform(transform @ rotate_z)


def visualize_bbox(vis, name, dims, T=None, color=[255, 0, 0]):
    """Visualize a bounding box using a wireframe.

    Args:
        vis (MeshCat Visualizer): the visualizer
        name (string): name for this frame (should be unique)
        dims (array-like): shape (3,), dimensions of the bounding box
        T (4x4 numpy.array): (optional) transform to apply to this geometry

    """
    color_hex = rgb2hex(tuple(color))
    material = meshcat.geometry.MeshBasicMaterial(wireframe=True, color=color_hex)
    bbox = meshcat.geometry.Box(dims)
    vis[name].set_object(bbox, material)

    if T is not None:
        vis[name].set_transform(T)


def visualize_pointcloud(vis, name, pc, color=None, transform=None, **kwargs):
    """
    Args:
        vis: meshcat visualizer object
        name: str
        pc: Nx3 or HxWx3
        color: (optional) same shape as pc[0 - 255] scale or just rgb tuple
        transform: (optional) 4x4 homogeneous transform
    """
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)
        # Resize the color np array if needed.
        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])
        if color.ndim == 1:
            color = np.ones_like(pc) * np.array(color)

        # Divide it by 255 to make sure the range is between 0 and 1,
        color = color.astype(np.float32) / 255
    else:
        color = np.ones_like(pc)

    vis[name].set_object(
        meshcat.geometry.PointCloud(position=pc.T, color=color.T, **kwargs)
    )

    if transform is not None:
        vis[name].set_transform(transform)


def compute_pointcloud_pangolin(pc, color, transform=None):
    """Draw a pointcloud in pangolin visualizer"""
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)
        if color.ndim == 3:
            # flatten it to be [H*W, 3]
            color = color.reshape(-1, color.shape[-1]) / 255.0
        if color.ndim == 1:
            color = np.ones_like(pc) * np.array(color) / 255.0
    else:
        color = np.ones_like(pc)

    if transform is not None:
        pc = tra.transform_points(pc, transform)

    return pc, color


def visualize_pointclouds_pangolin(vis, pc_dict, **kwargs):
    """Draw a pointcloud in pangolin visualizer"""
    pc_list = []
    color_list = []

    for key, val in pc_dict.items():
        pc_list.append(val["pc"])
        color_list.append(val["color"])

    pc = np.concatenate(pc_list, axis=0)
    color = np.concatenate(color_list, axis=0)

    vis.draw_scene(pc, color, draw_axis=False, **kwargs)


def visualize_robot(vis, robot, name="robot", q=None, color=None):
    if q is not None:
        robot.set_joint_cfg(q)
    robot_link_poses = {
        linkname: robot.link_poses[linkmesh][0].cpu().numpy()
        for linkname, linkmesh in robot.link_map.items()
    }
    if color is not None and isinstance(color, np.ndarray) and len(color.shape) == 2:
        assert color.shape[0] == len(robot.physical_link_map)
    link_id = -1
    for link_name in robot.physical_link_map:
        link_id += 1
        coll_mesh = robot.link_map[link_name].collision_mesh
        assert coll_mesh is not None
        link_color = None
        if color is not None and not isinstance(color, np.ndarray):
            color = np.asarray(color)
        if color.ndim == 1:
            link_color = color
        else:
            link_color = color[link_id]
        if coll_mesh is not None:
            visualize_mesh(
                vis[name],
                f"{link_name}_{robot}",
                coll_mesh,
                color=link_color,
                transform=robot_link_poses[link_name].astype(np.float),
            )


def get_color_from_score(labels, use_255_scale=False):
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in [np.float32, float]:
        labels = scale * labels
        return [1 - labels, labels, 0]
    else:
        scale = 255.0 if use_255_scale else 1.0
        return scale * np.stack(
            [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
            axis=1,
        )


# def load_grasp_points():
#     control_points = np.load("data/panda.npy", allow_pickle=True)
#     control_points = [
#         [0, 0, 0, 1],
#         [0, 0, 0, 1],
#         list(control_points[0, :]),
#         list(control_points[1, :]),
#         list(control_points[-2, :]),
#         list(control_points[-1, :]),
#     ]

#     grasp_pc = np.asarray(control_points, np.float32)
#     grasp_pc[2, 2] = 0.059
#     grasp_pc[3, 2] = 0.059
#     mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])

#     modified_grasp_pc = []
#     modified_grasp_pc.append([0, 0, 0, 1])
#     modified_grasp_pc.append(mid_point)
#     modified_grasp_pc.append(grasp_pc[2])
#     modified_grasp_pc.append(grasp_pc[4])
#     modified_grasp_pc.append(grasp_pc[2])
#     modified_grasp_pc.append(grasp_pc[3])
#     modified_grasp_pc.append(grasp_pc[5])

#     return np.asarray(modified_grasp_pc, dtype=np.float32).T


# GRASP_VERTICES = load_grasp_points()


def visualize_grasp(vis, name, transform, color=[255, 0, 0]):
    vis[name].set_object(
        g.Line(
            g.PointsGeometry(GRASP_VERTICES),
            g.MeshBasicMaterial(color=rgb2hex(tuple(color))),
        )
    )
    vis[name].set_transform(transform.astype(np.float64))


def draw_table(table_range, vis, name, color):
    xs = [table_range[0][0], table_range[1][0]]
    ys = [table_range[0][1], table_range[1][1]]
    zs = [table_range[0][2], table_range[1][2]]

    coord_indexes = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
        (0, 0, 0),
    ]

    for dim in range(2):
        vertices = []
        for cindex in coord_indexes:
            vertices.append(
                [
                    xs[cindex[0]],
                    ys[cindex[1]],
                    zs[dim],
                ]
            )
            side = "bottom" if dim == 0 else "top"
            vis[f"{name}/{side}"].set_object(
                g.Line(
                    g.PointsGeometry(np.asarray(vertices).T),
                    g.MeshBasicMaterial(color=rgb2hex(tuple(color))),
                )
            )