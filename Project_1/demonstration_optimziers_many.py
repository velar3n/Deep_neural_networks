import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Rosenbrock function + grad
# ----------------------------
def rosenbrock(x, a=1.0, b=100.0):
    X, Y = x[0], x[1]
    return (a - X)**2 + b*(Y - X**2)**2

def grad_rosenbrock(x, a=1.0, b=100.0):
    X, Y = x[0], x[1]
    dX = -2*(a - X) - 4*b*X*(Y - X**2)
    dY =  2*b*(Y - X**2)
    return np.array([dX, dY], dtype=float)

# ----------------------------
# Optimizers
# ----------------------------
def optimize(optimizer, x0, steps=2000, lr=1e-3, seed=42,
             noise_std=0.0,
             beta=0.9,
             beta1=0.9, beta2=0.999,
             rms_beta=0.99,
             eps=1e-8,
             weight_decay=0.0):
    
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    
    # State variables
    v_mom = np.zeros_like(x)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    s = np.zeros_like(x)
    v_muon = np.zeros_like(x)

    # HISTORY: We must store the path!
    history = [x.copy()]

    for t in range(1, steps + 1):
        # 1. Gradient Calculation
        if optimizer == "Nesterov":
            # Nesterov lookahead
            g = grad_rosenbrock(x - lr * beta * v_mom)
        else:
            g = grad_rosenbrock(x)

        # 2. Add Noise (Simulate SGD)
        if noise_std > 0:
            g = g + rng.normal(0.0, noise_std, size=2)

        # 3. Update Rules
        if optimizer == "GD":
            x = x - lr * g

        elif optimizer == "Momentum":
            # Classic Polyak Momentum
            v_mom = beta * v_mom + g 
            x = x - lr * v_mom

        elif optimizer == "Nesterov":
            # Sutskever formulation equivalent
            v_mom = beta * v_mom + g
            x = x - lr * v_mom

        elif optimizer == "RMSProp":
            s = rms_beta * s + (1 - rms_beta) * (g**2)
            x = x - lr * g / (np.sqrt(s) + eps)

        elif optimizer == "Adam":
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

        elif optimizer == "AdamW":
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            # Weight decay is applied to X directly, decoupled from gradient adaptation
            x = x - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * x)

        elif optimizer == "Muon":
            # Simplified 1D Muon (Normalized Momentum)
            v_muon = beta * v_muon + (1 - beta) * g # EMA style for Muon usually
            
            # "Newton-Schulz" on a vector is just Normalization
            norm = np.linalg.norm(v_muon)
            if norm > 1e-12:
                direction = v_muon / norm
            else:
                direction = v_muon
            
            x = x - lr * direction - lr * weight_decay * x

        # Save history
        history.append(x.copy())

        # Stop if diverged
        if not np.isfinite(x).all():
            break

    return np.array(history)

# ----------------------------
# Visualization
# ----------------------------
def demo_rosenbrock_trajectories(
    x0=(-2.0, 2.0), # Harder starting point
    steps=1000
):
    # Setup Grid
    xs = np.linspace(-2.5, 2.5, 300)
    ys = np.linspace(-1.5, 3.5, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = rosenbrock([X, Y])
    Z_plot = np.clip(Z, 0, 400) # Clip for visibility

    # Run Optimizers
    # Note: LRs are tuned for visual clarity
    opt_configs = [
        ("GD",        {"lr": 0.0005, "steps": steps*2}), # GD is slow
        ("Momentum",  {"lr": 0.0005, "beta": 0.9, "steps": steps}),
        ("Nesterov",  {"lr": 0.0005, "beta": 0.9, "steps": steps}),
        ("RMSProp",   {"lr": 0.01,   "steps": steps}),
        ("Adam",      {"lr": 0.05,   "steps": steps}),
        ("Muon",      {"lr": 0.02,   "steps": steps, "beta": 0.95})
    ]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot Surface
    ax.plot_surface(X, Y, Z_plot, cmap='gray', alpha=0.2, rstride=5, cstride=5)
    
    # Global Min
    ax.scatter(1.0, 1.0, 0, s=200, c='yellow', marker='*', label='Global Min', zorder=10)

    # Plot paths
    colors = plt.cm.jet(np.linspace(0, 1, len(opt_configs)))
    
    for (name, params), color in zip(opt_configs, colors):
        path = optimize(name, x0, **params)
        
        # Calculate Z for the path
        z_path = [rosenbrock(p) for p in path]
        
        # Plot line
        ax.plot(path[:,0], path[:,1], z_path, label=name, color=color, linewidth=2)
        # Plot Endpoint
        ax.scatter(path[-1,0], path[-1,1], z_path[-1], s=50, c=[color], marker='o')
        
        print(f"{name:10s}: Steps={len(path)} End=({path[-1,0]:.2f}, {path[-1,1]:.2f})")

    ax.set_title("Comparison of Optimizer Trajectories")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 3.5)
    ax.set_zlim(0, 400)
    ax.view_init(elev=60, azim=-45)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_rosenbrock_trajectories()
