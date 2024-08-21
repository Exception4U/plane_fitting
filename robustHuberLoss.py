import torch

# Set random seed for reproducibility
torch.manual_seed(42)

def generate_noisy_planes(num_points=1000, noise_std=0.1):
    # Plane 1: x-y plane (z = 0)
    x1 = torch.rand(num_points) * 10 - 5  # x between -5 and 5
    y1 = torch.rand(num_points) * 10 - 5  # y between -5 and 5
    z1 = torch.zeros(num_points)  # z = 0 for the x-y plane
    plane1 = torch.stack([x1, y1, z1], dim=1)

    # Plane 2: y-z plane (x = 0)
    y2 = torch.rand(num_points) * 10 - 5  # y between -5 and 5
    z2 = torch.rand(num_points) * 10 - 5  # z between -5 and 5
    x2 = torch.zeros(num_points)  # x = 0 for the y-z plane
    plane2 = torch.stack([x2, y2, z2], dim=1)

    # Add Gaussian noise
    noise1 = torch.randn_like(plane1) * noise_std
    noise2 = torch.randn_like(plane2) * noise_std

    plane1_noisy = plane1 + noise1
    plane2_noisy = plane2 + noise2

    return plane1_noisy, plane2_noisy

# Generate the noisy planes
plane1_noisy, plane2_noisy = generate_noisy_planes()

def plane_mse(plane_params, points):
    a, b, c, d = plane_params
    residuals = points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d
    return torch.mean(residuals**2)

def least_squares_plane_fitting(points, max_iters=1000, lr=1e-3):
    plane_params = torch.randn(4, requires_grad=True)  # Initialize plane parameters

    optimizer = torch.optim.Adam([plane_params], lr=lr)

    initial_mse = plane_mse(plane_params, points).item()

    for _ in range(max_iters):
        optimizer.zero_grad()
        loss = plane_mse(plane_params, points)
        loss.backward()
        optimizer.step()

    final_mse = plane_mse(plane_params, points).item()

    return plane_params.detach().numpy(), initial_mse, final_mse

# Fit plane 1 (x-y plane) using least squares
params_plane1, initial_mse1, final_mse1 = least_squares_plane_fitting(plane1_noisy)

# Fit plane 2 (y-z plane) using least squares
params_plane2, initial_mse2, final_mse2 = least_squares_plane_fitting(plane2_noisy)

print(f"Plane 1 (Least Squares): Initial MSE = {initial_mse1:.6f}, Final MSE = {final_mse1:.6f}")
print(f"Plane 2 (Least Squares): Initial MSE = {initial_mse2:.6f}, Final MSE = {final_mse2:.6f}")


def add_outliers(points, percentage=0.1, magnitude=50):
    num_outliers = int(len(points) * percentage)
    outliers = torch.randn(num_outliers, 3) * magnitude
    return torch.cat([points, outliers], dim=0)

# Add outliers to both planes
plane1_noisy_outliers = add_outliers(plane1_noisy)
plane2_noisy_outliers = add_outliers(plane2_noisy)

# Fit plane 1 with outliers using least squares
params_plane1_outliers, initial_mse1_outliers, final_mse1_outliers = least_squares_plane_fitting(plane1_noisy_outliers)

# Fit plane 2 with outliers using least squares
params_plane2_outliers, initial_mse2_outliers, final_mse2_outliers = least_squares_plane_fitting(plane2_noisy_outliers)

print(f"Plane 1 with Outliers (Least Squares): Initial MSE = {initial_mse1_outliers:.6f}, Final MSE = {final_mse1_outliers:.6f}")
print(f"Plane 2 with Outliers (Least Squares): Initial MSE = {initial_mse2_outliers:.6f}, Final MSE = {final_mse2_outliers:.6f}")


def custom_huber_loss(residuals, delta=1.0):
    abs_residuals = torch.abs(residuals)
    quadratic = torch.min(abs_residuals, torch.tensor(delta))
    linear = abs_residuals - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def robust_plane_fitting(points, max_iters=1000, lr=1e-3, delta=1.0):
    plane_params = torch.randn(4, requires_grad=True)  # Initialize plane parameters

    optimizer = torch.optim.Adam([plane_params], lr=lr)

    residuals = points[:, 0] * plane_params[0] + points[:, 1] * plane_params[1] + points[:, 2] * plane_params[2] + plane_params[3]
    initial_mse = torch.mean(residuals**2).item()

    for _ in range(max_iters):
        optimizer.zero_grad()
        residuals = points[:, 0] * plane_params[0] + points[:, 1] * plane_params[1] + points[:, 2] * plane_params[2] + plane_params[3]
        loss = custom_huber_loss(residuals, delta).mean()
        loss.backward()
        optimizer.step()

    residuals = points[:, 0] * plane_params[0] + points[:, 1] * plane_params[1] + points[:, 2] * plane_params[2] + plane_params[3]
    final_mse = torch.mean(residuals**2).item()

    return plane_params.detach().numpy(), initial_mse, final_mse

# Fit plane 1 with outliers using Huber loss
params_plane1_huber, initial_mse1_huber, final_mse1_huber = robust_plane_fitting(plane1_noisy_outliers)

# Fit plane 2 with outliers using Huber loss
params_plane2_huber, initial_mse2_huber, final_mse2_huber = robust_plane_fitting(plane2_noisy_outliers)

print(f"Plane 1 with Outliers (Huber Loss): Initial MSE = {initial_mse1_huber:.6f}, Final MSE = {final_mse1_huber:.6f}")
print(f"Plane 2 with Outliers (Huber Loss): Initial MSE = {initial_mse2_huber:.6f}, Final MSE = {final_mse2_huber:.6f}")
