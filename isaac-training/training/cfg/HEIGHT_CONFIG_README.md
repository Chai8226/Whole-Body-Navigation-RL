# Height Constraint Configuration Guide

This document explains the configurable parameters for height constraints in the navigation environment.

## Configuration Parameters

All parameters are set in `train.yaml` under the `env` section:

```yaml
env:
  # Flight height constraints
  min_flight_height: 0.5        # Minimum allowed flight height (m)
  max_flight_height: 4.0        # Maximum allowed flight height (m)
  height_penalty_deadzone: 0.2  # Buffer zone before penalty applies (m)
  height_penalty_weight: 8.0    # Weight of height penalty in reward function
```

## Parameter Details

### `min_flight_height` (default: 0.5 m)
- **Type**: float
- **Description**: The minimum allowed flight height for the drone
- **Effect**: Drones flying below `min_flight_height - height_penalty_deadzone` will receive penalties
- **Typical values**: 
  - Indoor/low-altitude: 0.3 - 0.5 m
  - Outdoor/standard: 0.5 - 1.0 m

### `max_flight_height` (default: 4.0 m)
- **Type**: float
- **Description**: The maximum allowed flight height for the drone
- **Effect**: Drones flying above `max_flight_height + height_penalty_deadzone` will receive penalties
- **Typical values**:
  - Low-ceiling: 2.5 - 3.5 m
  - Standard: 3.5 - 4.5 m
  - High-altitude: 5.0 - 6.0 m

### `height_penalty_deadzone` (default: 0.2 m)
- **Type**: float
- **Description**: Buffer zone (tolerance) before penalties are applied
- **Effect**: 
  - Safe zone: `[min_flight_height - deadzone, max_flight_height + deadzone]`
  - Example with defaults: [0.3m, 4.2m] is penalty-free
- **Typical values**:
  - Strict control: 0.1 m
  - Standard: 0.2 m
  - Lenient: 0.3 - 0.5 m

### `height_penalty_weight` (default: 8.0)
- **Type**: float
- **Description**: Multiplier for height penalty in the reward function
- **Effect**: Higher values = stronger penalty for violating height constraints
- **Formula**: `reward -= penalty_height * height_penalty_weight`
- **Typical values**:
  - Light penalty: 2.0 - 5.0
  - Standard: 5.0 - 10.0
  - Heavy penalty: 10.0 - 20.0

## Penalty Calculation

The height penalty is calculated as follows:

```python
# No penalty in safe zone
if min_flight_height - deadzone <= z <= max_flight_height + deadzone:
    penalty = 0

# Penalty for flying too low
if z < min_flight_height - deadzone:
    penalty = ((min_flight_height - deadzone) - z)²

# Penalty for flying too high
if z > max_flight_height + deadzone:
    penalty = (z - (max_flight_height + deadzone))²
```

The squared penalty means that larger violations are punished exponentially more.

## Example Configurations

### Indoor Low-Ceiling Environment
```yaml
env:
  min_flight_height: 0.3
  max_flight_height: 2.5
  height_penalty_deadzone: 0.15
  height_penalty_weight: 10.0
```
Safe flying zone: [0.15m, 2.65m]

### Outdoor Standard Environment
```yaml
env:
  min_flight_height: 0.5
  max_flight_height: 4.0
  height_penalty_deadzone: 0.2
  height_penalty_weight: 8.0
```
Safe flying zone: [0.3m, 4.2m] (current default)

### High-Altitude Environment
```yaml
env:
  min_flight_height: 1.0
  max_flight_height: 6.0
  height_penalty_deadzone: 0.3
  height_penalty_weight: 6.0
```
Safe flying zone: [0.7m, 6.3m]

### Lenient Training (Early Stage)
```yaml
env:
  min_flight_height: 0.5
  max_flight_height: 4.5
  height_penalty_deadzone: 0.5
  height_penalty_weight: 5.0
```
Safe flying zone: [0.0m, 5.0m] - Large tolerance for exploration

## Tuning Guidelines

1. **Start with lenient settings** during early training to allow exploration
2. **Gradually increase `height_penalty_weight`** as training progresses
3. **Reduce `height_penalty_deadzone`** for more precise height control
4. **Match height range to your environment**:
   - Consider obstacle heights (beam_height_range)
   - Leave clearance above tallest obstacles
   - Ensure minimum clearance from ground

## Related Parameters

These height constraints work together with:
- `beam_height_range`: Height of static beam obstacles [0.5, 3.5] m
- Collision threshold: 0.1 m (surface clearance)
- Termination conditions: `below_bound` (z < 0.2m), `above_bound` (z > 4.0m)

## Notes

- The penalty uses **squared error**, so violations increase exponentially
- All height values are in **meters** (SI units)
- Height is measured from the ground plane (z = 0)
- The termination bounds (0.2m, 4.0m) should be compatible with your height range
- If you change `max_flight_height` significantly, also update the termination `above_bound` in code
