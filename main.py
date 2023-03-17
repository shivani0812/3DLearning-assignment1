"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio
import numpy as np
import mcubes

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh, get_points_renderer, unproject_depth_image
from starter.camera_transforms import render_cow
from starter.dolly_zoom import dolly_zoom
from starter.render_generic import load_rgbd_data


def render_gif(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None, output_path='cow_gif.gif'
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0, -3]], device=device)

    # Render 360 degrees around the cow.
    images = []

    for i in range(0, 360, 10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3.0,
                                                                 azim=i,
                                                                 device=device)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255).astype(np.uint8)
        # The .cpu moves the tensor to GPU (if needed).

        images.append(rend)
    imageio.mimsave(output_path, images, fps=15)


def render_gif_from_mesh(vertices, faces, output_path, textures=None, distance=3, fov=60, image_size=256, color=[0.7, 0.7, 1], device=None):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    # if textures is None:
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    # elif textures.shape[0] != vertices.shape[0]:
    #     textures = textures.unsqueeze(0)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0, -3]], device=device)

    # Render 360 degrees around the cow.
    images = []

    for i in range(0, 360, 10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
                                                                 azim=i,
                                                                 device=device)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=fov, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255).astype(np.uint8)
        # The .cpu moves the tensor to GPU (if needed).

        images.append(rend)
    imageio.mimsave(output_path, images, fps=15)


    
def render_gif_from_pointcloud(verts, rgb, output_path, image_size=256, background_color=(1, 1, 1), distance=10, fov=60, device=None):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    point_cloud = pytorch3d.structures.Pointclouds(points=verts,
                                                   features=rgb).to(device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)
    
    images = []

    for i in range(0, 360, 10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
                                                                 azim=i,
                                                                 device=device)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=fov, device=device
        )

        rend = renderer(point_cloud, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        rend = (rend * 255).astype(np.uint8)

        images.append(rend)
    imageio.mimsave(output_path, images, fps=15)


def construct_tetrahedron(image_size=256, color=[0.7, 0.7, 1], device=None, output_path='output/tetrahedron_gif.gif'):
    

    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    tetrahedron_base = 4
    tetrahedron_height = 4
    # vertices = torch.tensor([[-tetrahedron_base/2, -tetrahedron_height/2, -tetrahedron_base/2],
    #  [tetrahedron_base/2, -tetrahedron_height/2, -tetrahedron_base/2],
    #  [0, -tetrahedron_height/2, tetrahedron_base/2],
    #  [0, tetrahedron_height/2, 0]], dtype=torch.float32)

    vertices = torch.tensor([
        [-1.0, -1.0, -1.0],
        [1, -1, -1],
        [0, -1, 0],
        [0, 1, 0]])

    faces = torch.tensor([
        [0, 3, 1],
        [1, 3, 2],
        [2, 3, 0],
        [0, 1, 2],
    ])


#     vertices = torch.tensor([
#     [-1.0, -1.0, -1.0],
#     [-21.0, -1.0, -1.0],
#     [1.0, 1.0, 1.0],
#     [-1.0, 1.0, -1.0]
# ], dtype=torch.float32)
    # faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.int64)

    # vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    # faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    render_gif_from_mesh(vertices, faces, output_path=output_path,
                         distance=5, image_size=image_size, color=color, device=None)


def construct_cube(image_size=256, color=[0.7, 0.7, 1], device=None, output_path='output/cube_gif.gif'):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    vertices = torch.tensor([[-1, -1, -1],
                             [1, -1, -1],
                             [-1, 1, -1],
                             [1, 1, -1],
                             [-1, -1, 1],
                             [1, -1, 1],
                             [-1, 1, 1],
                             [1, 1, 1]], dtype=torch.float32)

    faces = torch.tensor([[0, 1, 2],
                          [1, 2, 3],
                          [4, 5, 6],
                          [5, 6, 7],
                          [0, 2, 4],
                          [2, 4, 6],
                          [1, 3, 5],
                          [3, 5, 7],
                          [0, 1, 4],
                          [1, 4, 5],
                          [2, 3, 6],
                          [3, 6, 7]], dtype=torch.int64)

    # vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    # faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    render_gif_from_mesh(vertices, faces, output_path=output_path,
                         distance=5, image_size=image_size, color=color, device=None)


def retextured_mesh(vertices, faces, image_size=256, color1=[1, 0.2, 0], color2=[0.2, 1, 0.8], output_path='output/retextured_mesh.gif', device=None):
    if device is None:
        device = get_device()

    z = vertices[:, 2]
    alpha = (z - z.min())/(z.max() - z.min())
    alpha = alpha.repeat(3, 1).T
    color1 = torch.tensor(color1)
    color2 = torch.tensor(color2)
    color = alpha * color2 + (1-alpha) * color1

    textures = torch.ones_like(vertices)
    textures = textures * (color)
    torch.unsqueeze(textures, 0)


    render_gif_from_mesh(vertices, faces, textures=textures, output_path=output_path,
                         distance=5, image_size=image_size, color=color, device=None)


def camera_transformations(output_path='output/camera_transformations.jpg', device=None):
    # 90 degree counter-clockwise rotation around the z-axis
    R1 = [[0, -1,  0],
          [1,  0,  0],
          [0,  0,  1]]

    R2 = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]

    R3 = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]

    # 90 degree counter-clockwise rotation around the y-axis
    R4 = [[0., 0., -1],
            [0, 1, 0],
            [1., 0, 0]]

            

    T1 = [0, 0, 0]
    T2 = [0, 0, 2]
    T3 = [1, -1, 0]
    T4 = [3, 0, 3]

    R_relative = [R1, R2, R3, R4]
    T_relative = [T1, T2, T3, T4]

    i = 0

    for (R_relative, T_relative) in zip(R_relative, T_relative):
        plt.imsave('.'.join([output_path.split('.')[0] + '-{0}'.format(i),
                             output_path.split('.')[-1]]),
                   render_cow(R_relative=R_relative,
                              T_relative=T_relative))
        i += 1

def render_point_cloud_from_rgbd(output_path = 'output/poin_cloud_plant.gif', device = None):
    if device is None:
        device = get_device()

    # Load the RGB-D data
    data = load_rgbd_data(path="data/rgbd_data.pkl")

    # get the points and colors of image 1
    points1, colors1 = unproject_depth_image(torch.from_numpy(data['rgb1']),
                                       torch.from_numpy(data['mask1']),
                                       torch.from_numpy(data['depth1']),
                                       data['cameras1'])

    # get the points and colors of image 2
    points2, colors2 = unproject_depth_image(torch.from_numpy(data['rgb2']),
                                       torch.from_numpy(data['mask2']),
                                       torch.from_numpy(data['depth2']),
                                       data['cameras2'])



    # currently the pointcloud is rotated, so we rotate it back
    R = torch.tensor([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]], dtype=torch.float32)
    points1 = torch.matmul(points1, R)
    points2 = torch.matmul(points2, R)

    


    points3 = torch.cat((points1, points2), dim=0)
    colors3 = torch.cat((colors1, colors2), dim=0)

    points1.unsqueeze_(0)
    colors1.unsqueeze_(0)
    points2.unsqueeze_(0)
    colors2.unsqueeze_(0)
    points3.unsqueeze_(0)
    colors3.unsqueeze_(0)    

    render_gif_from_pointcloud(points1, colors1, output_path='output/plant1.gif', device=device)
    render_gif_from_pointcloud(points2, colors2, output_path='output/plant2.gif', device=device)
    render_gif_from_pointcloud(points3, colors3, output_path='output/plant3.gif', device=device)


def render_torus_parametric(image_size=256, num_samples=200,output_path='output/torus_parametric.gif', device=None):
    if device is None:
        device = get_device()

    R = 4
    r = 2

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    render_gif_from_pointcloud(points.unsqueeze(0), color.unsqueeze(0), distance = 15,output_path=output_path, device=device)

def render_torus_implicit(image_size=256, num_samples=200, voxel_size = 64,output_path='output/torus_implicit.gif', device=None):
    if device is None:
        device = get_device()

    R = 4
    r = 2

    min_value = -R - r
    max_value = R + r

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # calculate number of voxels for a torus
    voxels = (X ** 2 + Y ** 2 + Z ** 2 + R ** 2 - r ** 2) ** 2 - 4 * R ** 2 * (X ** 2 + Y ** 2)

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    render_gif_from_mesh(vertices, faces, textures=textures, output_path=output_path,
                         distance=15, device=None)

def render_spider(image_size=512, fov= 60,color=[0.2, 0.5, 0.6], device=None, output_path='output/spider_gif.gif'):
    if device is None:
        device = get_device()
    
    mesh = pytorch3d.io.load_objs_as_meshes(["data/obj/Only_Spider_with_Animations_Export.obj"])
    # mesh = mesh.to(device)
    vertices, faces = mesh.get_mesh_verts_faces(0)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    steps = range(120, -120 -5)
    images = []

    R = [[1,0,0],
         [0,1,0],
         [0,0,1]]

    distance = 200

    for i in range(100, -100, -10):
        T = torch.tensor([[0, i, distance]], device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        rend = (rend * 255).astype(np.uint8)
        images.append(rend)
    imageio.mimsave(output_path, images, fps=15)
    
    
def extra_credit(path,distance,output_path,num_samples = 100, color = [0.7,0.7,0.7], device = None):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes([path])
    vertices, faces = mesh.get_mesh_verts_faces(0)
    areas = mesh.faces_areas_packed()
    
    render_gif_from_mesh(vertices, faces, textures=None, output_path='output/joint_mesh.gif', device=device, distance=distance)
    # areas = areas / areas.sum()
    sampled_faces = torch.multinomial(areas, num_samples, replacement=True)
    # sampled_faces = np.random.choice(np.arange(len(faces)), size=num_samples, p=areas)

    sampled_vertices = faces[sampled_faces]

    alpha1 = 1. - torch.rand(num_samples)
    alpha2 = (1. - alpha1) * torch.rand(num_samples)
    alpha3 = (1. - alpha1) * (1. - alpha2)

    alpha1 = alpha1.reshape(-1, 1)
    alpha2 = alpha2.reshape(-1, 1)
    alpha3 = alpha3.reshape(-1, 1)

    x = vertices[sampled_vertices[:, 0]]
    y = vertices[sampled_vertices[:, 1]]
    z = vertices[sampled_vertices[:, 2]]
    points = alpha1 * x + alpha2 * y + alpha3 * z
    
    # vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    # faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    colors = torch.ones_like(points) * torch.tensor(color)

    points.unsqueeze_(0)
    colors.unsqueeze_(0)
    output_path = '.'.join([output_path.split('.')[0] + '-{0}'.format(num_samples),
                            output_path.split('.')[-1]])
    
    render_gif_from_pointcloud(points, colors, distance = distance,output_path=output_path, device=device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-q',
                        '--question',
                        default='1.1')
    args = parser.parse_args()
    question = args.question

    if question == '1.1':
        vertices, faces = load_cow_mesh("data/cow.obj")
        output_path = 'output/cow_gif.gif'
        render_gif_from_mesh(vertices, faces, output_path)

    if question == '1.2':
        dolly_zoom()
    if question == '2.1':
        construct_tetrahedron()

    if question == '2.2':
        construct_cube()

    if question == '3':
        vertices, faces = load_cow_mesh("data/cow.obj")
        retextured_mesh(vertices, faces)

    if question == '4':
        camera_transformations()
    
    if question == '5.1':
        render_point_cloud_from_rgbd()

    if question == '5.2':
        render_torus_parametric()
    
    if question == '5.3':
        render_torus_implicit()
    
    if question == '6':
        render_spider()
    if question == '7':
        for i in [10, 100, 1000, 10000]:
            extra_credit(path = "data/cow.obj",distance = 5,output_path = 'output/extra_credit_cow_gif.gif',num_samples=i)
            extra_credit(path = "data/joint_mesh.obj",distance = 15,output_path = 'output/extra_credit_joint_gif.gif',num_samples=i)
        
