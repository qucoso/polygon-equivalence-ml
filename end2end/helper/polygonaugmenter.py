import torch
import math
import random
from typing import Tuple

def biased_random_uniform(min_val, max_val, bias_power=2.0):
    u = random.random()  # gleichverteilte Zufallszahl [0, 1]
    biased_u = u ** bias_power  # bias: stärker Richtung min_val
    return min_val + (max_val - min_val) * biased_u

class PolygonAugmenter:
    def __init__(
            self,
            scale_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.5, 0.8), (1.2, 2.0)),
            rotation_range: Tuple[float, float] = (15.0, 345.0),
            translate_range: Tuple[float, float] = (0.2, 3.0),
            bias_power: float = 2.0
    ):
        self.scale_ranges = scale_ranges
        self.rotation_range = rotation_range
        self.translate_range = translate_range
        self.bias_power = bias_power

    def __call__(self, polygon_tensor: torch.Tensor, second_polygon: torch.Tensor = None) -> torch.Tensor:
        result = polygon_tensor.clone()
        device = polygon_tensor.device

        min_xy, _ = polygon_tensor.min(dim=0)
        max_xy, _ = polygon_tensor.max(dim=0)
        center = (min_xy + max_xy) / 2

        # Special mode: Align second polygon to the first one's center
        if second_polygon is not None:
            second_min_xy, _ = second_polygon.min(dim=0)
            second_max_xy, _ = second_polygon.max(dim=0)
            second_center = (second_min_xy + second_max_xy) / 2
            offset = center - second_center
            return second_polygon + offset

        # --- Standard Augmentation ---
        extent = (max_xy - min_xy).mean()

        # Shuffle augmentation operations
        # ops = ['rotate', 'scale', 'translate']
        # ops_to_apply = torch.randperm(len(ops)).tolist()
        ops = (["scale"] * 2) + (["rotate"] * 1) + (["translate"] * 7)
        # ops = (["scale"] * 1) + (["rotate"] * 1) + (["translate"] * 1)
        op = random.choice(ops)

        # applied = False
        # for op_idx in ops_to_apply:
        #     op = ops[op_idx]
            # Ensure at least one operation is applied
            # if torch.rand(1).item() > 0.5 or not applied:

        if op == 'rotate':
            angle_deg = biased_random_uniform(*self.rotation_range, self.bias_power)
            angle_rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=result.dtype, device=device)

            result = (result - center) @ rot_matrix.T + center
            # applied = True

        elif op == 'scale':
            # Choose from one of the two scale ranges
            range_idx = 0 if torch.rand(1).item() > 0.5 else 1
            scale = biased_random_uniform(*self.scale_ranges[range_idx], self.bias_power)

            result = (result - center) * scale + center
            # applied = True

        elif op == 'translate':
            direction = torch.randn(2, device=device)
            direction = direction / (direction.norm() + 1e-6)  # Normalize
            distance = biased_random_uniform(*self.translate_range, self.bias_power) * extent

            result = result + (direction * distance)
            # applied = True

        return result