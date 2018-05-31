import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, target_angles=None,
        target_rotor_speeds=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 1
        self.action_high = 800
        self.action_size = 4

        # Goal
        self.target_rotor_speeds = target_rotor_speeds if target_rotor_speeds is not None else np.array(
            [(self.action_high - self.action_low)/2 for i in range(4)])
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.target_angles = target_angles if target_angles is not None else np.array([0., 0., 0.]) 

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""
        survivor_reward = 100
        pos_punishment = np.sqrt((abs(self.sim.pose[:3] - self.target_pos)**2).sum())/10000
        angle_punishment = (abs(self.sim.pose[3:] - self.target_angles)).sum()/10000
        rotor_equality_measure = np.array([(rotor_rpm - np.median(rotor_speeds))**2 for rotor_rpm in rotor_speeds])
        rotor_equality_reward = 0#1000. - rotor_equality_measure.sum()
        dont_use_rotor_extrema_punishment = 0#(abs(rotor_speeds - self.target_rotor_speeds)**4).sum()/30000000
        #print("dont_use_rotor_extrema_punishment", dont_use_rotor_extrema_punishment, "rotor_speeds", rotor_speeds)
        # return reward + angle_reward + rotor_equality_reward
        return survivor_reward - pos_punishment + rotor_equality_reward - angle_punishment - dont_use_rotor_extrema_punishment

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds = [rotor_speeds for i in range(4)] if len(rotor_speeds) == 1 else rotor_speeds
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state