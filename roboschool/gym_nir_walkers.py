from roboschool.gym_mujoco_walkers import *

class RoboschoolMutant(RoboschoolForwardWalkerMujocoXML):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot', 'right_mid_foot', 'left_mid_foot']
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "mutant.xml", "torso", action_dim=12, obs_dim=38, power=2.5)
    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground
