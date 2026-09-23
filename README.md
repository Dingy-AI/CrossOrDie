# CrossOrDie
Neural Network Learns How to Cross a Bridge


Initial Splendor Agent and Environment

Linux: source .venv/bin/activate

Windows: .venv/Scripts/activate.bat

Activate virtual environment: source .venv/Scripts/activate

# evolve flat ground demo
python tools/evolve_flat_ground.py

# continue flat ground demo
python tools/evolve_flat_ground.py --generations 25

# continue existing population
python tools/evolve_flat_ground.py --resume checkpoints/flat_ground/latest_population.npz --generations 100




# checkpoint inspector
python tools/inspect_checkpoint.py checkpoints/flat_ground/best_ever.npz



# Runs pytest 
pytest -s




# Runs the sandbox
python tools/creature_sandbox.py

# Runs the best creature 
 python tools/creature_sandbox.py --genome checkpoints/flat_ground/best_ever.npz




CREATURE GENOME

BODY
├── body_length
├── body_height
├── body_mass
├── body_density
└── body_balance

FRONT LIMB
├── attachment_position
├── upper_length
├── lower_length
├── thickness
├── hip_strength
├── knee_strength
└── joint_range

REAR LIMB
├── attachment_position
├── upper_length
├── lower_length
├── thickness
├── hip_strength
├── knee_strength
└── joint_range

BEHAVIOR
├── motor_speed
├── motor_force
└── neural_controller_weights


                   TORSO
          ┌──────────────────┐
          │                  │
          └──────────────────┘
             │            │
          rear hip     front hip
             │            │
          upper          upper
             │            │
           knee          knee
             │            │
          lower          lower
            ───          ───
           foot          foot

BRAIN INPUT — 17 values

Torso
├── sin(angle)                  1
├── cos(angle)                  1
├── angular velocity            1
├── velocity X                  1
├── velocity Y                  1
├── vertical displacement       1
└── distance to goal            1

Joints
├── front hip angle             1
├── front hip velocity          1
├── front knee angle            1
├── front knee velocity         1
├── rear hip angle              1
├── rear hip velocity           1
├── rear knee angle             1
└── rear knee velocity          1

Feet
├── front foot contact          1
└── rear foot contact           1
                                  ──
                                  17

Input → Hidden weights     17 × 8 = 136
Hidden biases                       8
Hidden → Output weights     8 × 4 = 32
Output biases                        4
                                 ─────
                                   180

GENOME

Morphology
│
├── Body             5
├── Front leg        7
├── Rear leg         7
└── Joints           4
                    ──
                    23

Brain
│
├── W1             136
├── b1               8
├── W2              32
└── b2               4
                    ───
                    180

TOTAL               203 genes