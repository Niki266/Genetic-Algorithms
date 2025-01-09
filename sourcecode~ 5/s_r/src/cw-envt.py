import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import random
import creature
import genome
import population
import matplotlib.pyplot as plt
import pandas as pd

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Connect to the physics server
if p.connect(p.GUI) < 0:
    raise Exception("Failed to connect to physics server")

p.setAdditionalSearchPath(pybullet_data.getDataPath())

def make_arena(arena_size=10, wall_height=1):
    wall_thickness = 0.5
    floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness])
    floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness])

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])

    # Create four walls
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size/2, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size/2, wall_height/2])

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])

    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[arena_size/2, 0, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[-arena_size/2, 0, wall_height/2])

def evaluate_fitness(creature_id):
    pos, _ = p.getBasePositionAndOrientation(creature_id)
    # Adding debug text to visualize the robot's position
    p.addUserDebugText(f"Fitness: {pos[2]:.2f}", pos, textColorRGB=[1, 0, 0], lifeTime=1)
    return pos[2]  # Height achieved as the fitness score

def run_simulation(creature, duration=2400):
    try:
        with open('test.urdf', 'w') as f:
            f.write(creature.to_xml())
        print("Robot URDF created.")
        
        start_position = [5, -4, 1]
        creature_id = p.loadURDF(os.path.join(current_dir, 'test.urdf'), start_position)
        if creature_id < 0:
            raise Exception("Failed to load URDF.")
        print(f"Robot loaded with ID: {creature_id}")
        
        num_joints = p.getNumJoints(creature_id)
        print(f"Number of joints in the robot: {num_joints}")

        max_height = 0

        for _ in range(duration):
            p.stepSimulation()
            apply_motor_signals(creature_id, creature, num_joints)
            time.sleep(1./240.)

            # Track the maximum height achieved
            current_height = p.getBasePositionAndOrientation(creature_id)[0][2]
            if current_height > max_height:
                max_height = current_height

        fitness = evaluate_fitness(creature_id)
        position = p.getBasePositionAndOrientation(creature_id)[0]
        print(f"Robot position: {position}")

        # Color the line based on the height
        line_color = [0, 1, 0] if position[2] >= 2.0 else [1, 0, 0]  # Green if height >= 2.0, else red
        # Draw a line from the robot to the origin for tracking
        p.addUserDebugLine(position, [0, 0, 0], line_color, 2, duration)
        p.removeBody(creature_id)
        return max_height
    except Exception as e:
        print(f"Error during simulation: {e}")
        return float('-inf')

def apply_motor_signals(creature_id, creature, num_joints):
    motors = creature.get_motors()
    for joint_index, motor in enumerate(motors):
        if joint_index < num_joints:  # Ensure the joint index is within the range
            try:
                target_position = motor.get_output()
                p.setJointMotorControl2(creature_id, joint_index, p.POSITION_CONTROL, targetPosition=target_position)
            except Exception as e:
                print(f"Error applying motor signal to joint {joint_index}: {e}")

def genetic_algorithm_experiment(pop_size=10, gene_count=3, generations=3, mutation_rate=0.1, crossover_rate=0.7):
    pop = population.Population(pop_size=pop_size, gene_count=gene_count)
    for generation in range(generations):
        print(f"Generation {generation}")
        fitnesses = []
        for cr in pop.creatures:
            fitness = run_simulation(cr)
            fitnesses.append(fitness)
            print(f"Fitness of creature: {fitness}")
        best_fitness = max(fitnesses)
        print(f"Best fitness: {best_fitness}")
        fit_map = population.Population.get_fitness_map(fitnesses)
        new_creatures = []
        for _ in range(len(pop.creatures)):
            p1_ind = population.Population.select_parent(fit_map)
            p2_ind = population.Population.select_parent(fit_map)
            if p1_ind is None or p2_ind is None:
                continue
            p1 = pop.creatures[p1_ind]
            p2 = pop.creatures[p2_ind]
            dna = genome.Genome.crossover(p1.dna, p2.dna)
            dna = genome.Genome.point_mutate(dna, rate=mutation_rate, amount=0.25)
            dna = genome.Genome.shrink_mutate(dna, rate=0.25)
            dna = genome.Genome.grow_mutate(dna, rate=0.1)
            cr = creature.Creature(gene_count)
            cr.update_dna(dna)
            new_creatures.append(cr)
        for cr in pop.creatures:
            if run_simulation(cr) == best_fitness:
                new_cr = creature.Creature(gene_count)
                new_cr.update_dna(cr.dna)
                new_creatures[0] = new_cr
                filename = f"elite_{generation}.csv"
                genome.Genome.to_csv(cr.dna, filename)
                break
        pop.creatures = new_creatures

    return best_fitness

# Experiment settings
experiments = [
    {'pop_size': 10, 'mutation_rate': 0.1, 'crossover_rate': 0.7},
    {'pop_size': 20, 'mutation_rate': 0.2, 'crossover_rate': 0.6},
    {'pop_size': 30, 'mutation_rate': 0.3, 'crossover_rate': 0.5},
    {'pop_size': 40, 'mutation_rate': 0.4, 'crossover_rate': 0.4},
]

results = []

p.setGravity(0, 0, -10)
arena_size = 20
make_arena(arena_size=arena_size)

# Load mountain
mountain_position = (0, 0, 0)
mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
mountain_urdf_path = os.path.join(current_dir, "shapes", "gaussian_pyramid.urdf")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
mountain = p.loadURDF(mountain_urdf_path, mountain_position, mountain_orientation, useFixedBase=1)
print("Mountain loaded.")

# Set the camera position to ensure visibility of the entire arena
p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[5, 0, 1])

for experiment in experiments:
    best_fitness = genetic_algorithm_experiment(
        pop_size=experiment['pop_size'],
        gene_count=3,
        generations=2,  # Reduced to 3 generations for faster execution
        mutation_rate=experiment['mutation_rate'],
        crossover_rate=experiment['crossover_rate']
    )
    results.append({
        'pop_size': experiment['pop_size'],
        'mutation_rate': experiment['mutation_rate'],
        'crossover_rate': experiment['crossover_rate'],
        'best_fitness': best_fitness
    })

# Set real-time simulation to allow visual inspection
p.setRealTimeSimulation(1)

while True:
    time.sleep(1)

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Create graphs
plt.figure(figsize=(10, 5))

# Graph 1: Best Fitness vs Population Size
plt.subplot(1, 2, 1)
for mutation_rate in df['mutation_rate'].unique():
    subset = df[df['mutation_rate'] == mutation_rate]
    plt.plot(subset['pop_size'], subset['best_fitness'], label=f'Mutation Rate {mutation_rate}')
plt.xlabel('Population Size')
plt.ylabel('Best Fitness')
plt.title('Best Fitness vs Population Size')
plt.legend()

# Graph 2: Best Fitness vs Crossover Rate
plt.subplot(1, 2, 2)
for pop_size in df['pop_size'].unique():
    subset = df[df['pop_size'] == pop_size]
    plt.plot(subset['crossover_rate'], subset['best_fitness'], label=f'Population Size {pop_size}')
plt.xlabel('Crossover Rate')
plt.ylabel('Best Fitness')
plt.title('Best Fitness vs Crossover Rate')
plt.legend()

plt.tight_layout()
plt.show()

# Create tables
table1 = df.pivot(index='mutation_rate', columns='pop_size', values='best_fitness')
table2 = df.pivot(index='crossover_rate', columns='pop_size', values='best_fitness')

print("Table 1: Best Fitness for Different Population Sizes and Mutation Rates")
print(table1)
print("\nTable 2: Best Fitness for Different Population Sizes and Crossover Rates")
print(table2)