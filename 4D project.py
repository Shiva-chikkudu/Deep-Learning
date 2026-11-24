import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ------------------------------------------------------------
# 1. 2D "4D object": refractive index changes over time
# ------------------------------------------------------------

GRID = 64
TIME_STEPS = 40
NUM_RAYS = 25
HISTORY = 8


def make_index_field(t):
    """
    Create a time-varying refractive-index field.

    This represents a simplified 4D object:
    x, y are spatial dimensions and t is the fourth dimension.
    """
    x = np.linspace(-1, 1, GRID)
    y = np.linspace(-1, 1, GRID)
    X, Y = np.meshgrid(x, y)

    # A moving internal optical region
    center_x = 0.35 * np.sin(2 * np.pi * t / TIME_STEPS)
    center_y = 0.25 * np.cos(2 * np.pi * t / TIME_STEPS)

    radius = 0.30
    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # Background index = 1.0, internal region = 1.45
    n = np.ones((GRID, GRID))
    n[distance < radius] = 1.45

    return n


def trace_rays(n, num_rays=NUM_RAYS):
    """
    Very simplified 2D ray tracing through a refractive-index field.

    Rays enter from the left and bend toward higher refractive index.
    This is not a full physically exact ray tracer, but it demonstrates
    the project pipeline: light -> refraction -> internal sensors -> AI.
    """
    y_start = np.linspace(-0.8, 0.8, num_rays)
    rays = np.zeros((num_rays, GRID, 2))  # [ray, x-step, (y, angle)]

    rays[:, 0, 0] = y_start
    rays[:, 0, 1] = 0.0  # initial angle

    dx = 2.0 / GRID

    for step in range(1, GRID):
        prev_y = rays[:, step - 1, 0]
        prev_angle = rays[:, step - 1, 1]

        # Find nearest grid row for each ray
        y_index = np.clip(
            ((prev_y + 1) / 2 * (GRID - 1)).astype(int),
            0,
            GRID - 1
        )

        # Approximate gradient of refractive index
        grad_y = np.gradient(n, axis=0)

        # Simple refraction-like steering:
        # rays bend toward regions of higher refractive index
        bend = 0.035 * grad_y[y_index, step]

        new_angle = prev_angle + bend
        new_y = prev_y + dx * np.tan(new_angle)

        rays[:, step, 0] = np.clip(new_y, -1, 1)
        rays[:, step, 1] = new_angle

    return rays


def measure_sensors(rays):
    """
    Internal light sensors:
    measure ray density at several vertical sensor planes.
    """
    sensor_positions = [16, 32, 48]
    readings = []

    for pos in sensor_positions:
        ys = rays[:, pos, 0]
        hist, _ = np.histogram(
            ys,
            bins=16,
            range=(-1, 1)
        )
        readings.append(hist / (hist.sum() + 1e-8))

    return np.concatenate(readings)


# ------------------------------------------------------------
# 2. Generate simulation dataset
# ------------------------------------------------------------

def generate_dataset():
    sequences = []
    targets = []

    for episode in range(30):
        history = []

        for t in range(TIME_STEPS):
            n = make_index_field(t)
            rays = trace_rays(n)
            sensors = measure_sensors(rays)

            history.append(sensors)

            if len(history) > HISTORY:
                sequences.append(np.array(history[-HISTORY - 1:-1]))
                targets.append(sensors)

    return (
        torch.tensor(np.array(sequences), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32)
    )


# ------------------------------------------------------------
# 3. AI mechanism: LSTM light-pattern predictor
# ------------------------------------------------------------

class LightPredictor(nn.Module):
    def __init__(self, input_size=48, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def train_model(X, Y, epochs=30):
    model = LightPredictor(input_size=X.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(X)
        loss = loss_function(prediction, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d} | Loss: {loss.item():.6f}")

    return model


# ------------------------------------------------------------
# 4. Run simulation and visualize
# ------------------------------------------------------------

def visualize(n, rays, title):
    plt.figure(figsize=(8, 5))
    plt.imshow(
        n,
        extent=[-1, 1, -1, 1],
        origin="lower",
        cmap="viridis",
        alpha=0.75
    )

    for ray in rays:
        plt.plot(
            np.linspace(-1, 1, GRID),
            ray[:, 0],
            color="red",
            alpha=0.45,
            linewidth=0.8
        )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Refractive index")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Generating optical simulation dataset...")

    X, Y = generate_dataset()
    print("Dataset shape:", X.shape, Y.shape)

    print("Training AI light-prediction model...")
    model = train_model(X, Y)

    # Demonstrate prediction on a new time-varying object
    history = []

    for t in range(12):
        n = make_index_field(t)
        rays = trace_rays(n)
        sensors = measure_sensors(rays)
        history.append(sensors)

        if t == 5:
            visualize(
                n,
                rays,
                title=f"Light inside 4D object at time {t}"
            )

    recent_history = torch.tensor(
        np.array(history[-HISTORY:])[None, ...],
        dtype=torch.float32
    )

    predicted_next = model(recent_history).detach().numpy()[0]

    print("\nPredicted next internal sensor pattern:")
    print(np.round(predicted_next, 3))