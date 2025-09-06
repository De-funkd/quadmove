#!/usr/bin/env bash
set -e

# Top-level README.md
cat > README.md << 'EOF'
# Quad Move

Code Authors: **Advait Desai** and **Gargi Gupta**  
Project Mentors: **Ansh Semwal** and **Prajwal Awhad**  
This project is part of SRA VJTI's Eklavya 2025 Program.

Train a PPO-based gait policy in MuJoCo and deploy it on a low-cost, tortoise-style quadruped.

## 📁 Subprojects

1. Brax Training Viewer       → `1_brax_training_viewer/README.md`  
2. Mujoco Menagerie          → `2_mujoco_menagerie/README.md`  
3. Monte Carlo               → `3_monte_carlo/README.md`  
4. Q-Learning                → `4_Q_learning/README.md`  
5. Number Classifier         → `5_number_classifier/README.md`  
6. DQN                       → `6_DQN/README.md`  
7. DDQN                      → `7_DDQN/README.md`  
8. PPO – Bipedal             → `8_PPO_bipedal/README.md`  
9. PPO – Go2                 → `9_PPO_go2/README.md`

Run `./generate_readmes.sh` to regenerate all subproject READMEs automatically!
EOF

# Array of subproject directories and their readme contents
declare -A projects=(
  ["1_brax_training_viewer"]=$'![Multi-ant Viewer](pre_gifs/multi_ants.gif)\n\n``````'
  ["2_mujoco_menagerie"]=$'![Mujoco Flybody](pre_gifs/flybody.gif)\n\n``````'
  ["3_monte_carlo"]=$'![Frozen Lake](pre_gifs/frozen_lake.gif) ![Blackjack](pre_gifs/blackjack.gif)\n\n``````'
  ["4_Q_learning"]=$'![Cart Pole](pre_gifs/cart_pole.gif) ![Mountain Car](pre_gifs/mountain_car.gif)\n\n``````'
  ["5_number_classifier"]=$'**Number Classifier Demo**\n\n``````'
  ["6_DQN"]=$'![Lunar Lander DQN](pre_gifs/lunar_lander.gif)\n\n``````'
  ["7_DDQN"]=$'![Lunar Lander DDQN](pre_gifs/lunar_lander.gif)\n\n``````'
  ["8_PPO_bipedal"]=$'![Bipedal Walker](pre_gifs/bipedal_walker.gif)\n\n``````'
  ["9_PPO_go2"]=$'**Walk Demo**\n\n``````'
)

# Loop through each project and write its README.md
for dir in "${!projects[@]}"; do
  mkdir -p "$dir"
  cat > "$dir/README.md" << EOF
# $(echo $dir | sed 's/_/ /g')

${projects[$dir]}
EOF
done

echo "All README.md files generated successfully!"
