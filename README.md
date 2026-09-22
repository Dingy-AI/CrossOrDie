# CrossOrDie
Neural Network Learns How to Cross a Bridge


Initial Splendor Agent and Environment

Linux: source .venv/bin/activate

Windows: .venv/Scripts/activate.bat

Activate virtual environment: source .venv/Scripts/activate

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

# Runs pytest 
pytest -s


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