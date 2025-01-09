import pybullet as p
import pybullet_data
import time

# Connect to the physics server
p.connect(p.GUI)

# Set the search path to find URDFs and other assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a simple plane URDF
planeId = p.loadURDF("plane.urdf")

# Load a simple robot URDF
robotId = p.loadURDF("r2d2.urdf", [0, 0, 1])

# Set gravity
p.setGravity(0, 0, -10)

# Run the simulation for a few seconds
for i in range(240):
    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect from the physics server
p.disconnect()
