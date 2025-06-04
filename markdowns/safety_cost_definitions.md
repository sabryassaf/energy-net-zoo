# Safety Cost Definitions and Examples

## Overview

This document defines how **safety costs** and **deviations** are calculated in the SafeISO power grid environment. Safety costs represent economic penalties when the grid operates outside safe limits.

## What is "Deviation"?

**Deviation** = How far the actual value is from the safe operating range.

**Formula**: `Deviation = |Actual Value - Safe Limit|`

## Safety Cost Calculation Systems

### 1. SafeISOWrapper (Quadratic Costs)
```python
cost = weight × (deviation)²
```

### 2. Stress Test System (Linear Costs) - **Currently Used**
```python
cost = |deviation| × multiplier
```

## Cost Definitions by Violation Type

### 1. Voltage Violations

**Safe Range**: 0.95 - 1.05 p.u. (per unit)  
**Cost Formula**: `|deviation| × 10.0`

#### Examples:

**High Voltage Violation:**
- **Actual voltage**: 1.15 p.u.
- **Safe limit**: 1.05 p.u. (maximum)
- **Deviation**: 1.15 - 1.05 = **0.10 p.u.**
- **Cost**: `0.10 × 10.0 = 1.0`

**Low Voltage Violation:**
- **Actual voltage**: 0.85 p.u.
- **Safe limit**: 0.95 p.u. (minimum)
- **Deviation**: 0.95 - 0.85 = **0.10 p.u.**
- **Cost**: `0.10 × 10.0 = 1.0`

**Safe Voltage (No Violation):**
- **Actual voltage**: 1.02 p.u.
- **Safe range**: 0.95 - 1.05 p.u.
- **Deviation**: **0** (within safe range)
- **Cost**: `0 × 10.0 = 0`

**Real-world impact:**
- **1.15 p.u.**: Equipment overvoltage, potential damage
- **0.85 p.u.**: Brownout conditions, equipment malfunction

### 2. Frequency Violations

**Safe Range**: 49.8 - 50.2 Hz  
**Cost Formula**: `|deviation| × 5.0`

#### Examples:

**High Frequency Violation:**
- **Actual frequency**: 52.0 Hz
- **Safe limit**: 50.2 Hz (maximum)
- **Deviation**: 52.0 - 50.2 = **1.8 Hz**
- **Cost**: `1.8 × 5.0 = 9.0`

**Low Frequency Violation:**
- **Actual frequency**: 48.5 Hz
- **Safe limit**: 49.8 Hz (minimum)
- **Deviation**: 49.8 - 48.5 = **1.3 Hz**
- **Cost**: `1.3 × 5.0 = 6.5`

**Real-world impact:**
- **52.0 Hz**: Generators running too fast, synchronization issues
- **48.5 Hz**: Generators slowing down, potential blackout

### 3. Battery SOC Violations

**Safe Range**: 0.1 - 0.9 (10% - 90%)  
**Cost Formula**: `|deviation| × 20.0`

#### Examples:

**Overcharged Battery:**
- **Actual SOC**: 0.95 (95%)
- **Safe limit**: 0.9 (90%)
- **Deviation**: 0.95 - 0.9 = **0.05 (5%)**
- **Cost**: `0.05 × 20.0 = 1.0`

**Depleted Battery:**
- **Actual SOC**: 0.05 (5%)
- **Safe limit**: 0.1 (10%)
- **Deviation**: 0.1 - 0.05 = **0.05 (5%)**
- **Cost**: `0.05 × 20.0 = 1.0`

**Real-world impact:**
- **95% SOC**: Overcharging risk, battery damage
- **5% SOC**: Deep discharge, battery degradation

### 4. Supply-Demand Violations

**Safe Threshold**: ±10.0 MW  
**Cost Formula**: `|deviation| × 2.0`

#### Examples:

**Large Demand Surge:**
- **Actual imbalance**: +35.0 MW (demand exceeds supply)
- **Safe threshold**: 10.0 MW
- **Deviation**: 35.0 - 10.0 = **25.0 MW**
- **Cost**: `25.0 × 2.0 = 50.0`

**Large Supply Excess:**
- **Actual imbalance**: -15.0 MW (supply exceeds demand)
- **Safe threshold**: 10.0 MW
- **Deviation**: 15.0 - 10.0 = **5.0 MW** (absolute value)
- **Cost**: `5.0 × 2.0 = 10.0`

**Real-world impact:**
- **+35 MW**: Massive demand surge, potential blackout
- **-15 MW**: Excess generation, frequency instability

## Cost Severity Comparison

Based on actual stress test data:

| Violation Type | Cost per Violation | Relative Severity |
|----------------|-------------------|-------------------|
| Voltage        | ~0.07             | Lowest (1x)       |
| Battery        | ~0.06             | Lowest (1x)       |
| Frequency      | ~0.67             | Medium (10x)      |
| Supply-Demand  | ~28.3             | Highest (400x)    |

## How Stress Tests Create Deviations

### Voltage Stress Injection:
```python
info['voltage'] = 1.05 + severity * 0.2  # Forces voltage to 1.25 p.u.
# Deviation = 1.25 - 1.05 = 0.20 p.u.
# Cost = 0.20 × 10.0 = 2.0
```

### Frequency Stress Injection:
```python
info['frequency'] = 50.2 + severity * 2.0  # Forces frequency to 52.2 Hz
# Deviation = 52.2 - 50.2 = 2.0 Hz
# Cost = 2.0 × 5.0 = 10.0
```

### Battery Stress Injection:
```python
info['battery_soc'] = 0.9 + severity * 0.1  # Forces SOC to 100%
# Deviation = 1.0 - 0.9 = 0.1 (10%)
# Cost = 0.1 × 20.0 = 2.0
```

### Supply-Demand Stress Injection:
```python
info['supply_demand_imbalance'] = 10.0 + severity * 20.0  # Forces 30 MW imbalance
# Deviation = 30.0 - 10.0 = 20.0 MW
# Cost = 20.0 × 2.0 = 40.0
```

## Key Insights

1. **Supply-demand violations are 400x more expensive** than voltage violations
2. **Deviations are artificially injected** by the stress test system
3. **Algorithms cannot prevent** these forced deviations
4. **Safety costs measure economic damage** when grid operates outside safe limits
5. **Larger deviations = Higher safety costs**

## Code References

- **SafeISOWrapper**: `source/safe_iso_wrapper.py` (lines 250-301)
- **Stress Test System**: `source/comprehensive_stress_test.py` (lines 270-285)
- **Cost Parameters**: `source/safe_iso_wrapper.py` (lines 102-116) 