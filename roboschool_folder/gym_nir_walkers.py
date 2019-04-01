from roboschool.gym_mujoco_walkers import *
import pandas as pd
import csv

class RoboschoolMutant(RoboschoolForwardWalkerMujocoXML):
    foot_list = ['front_left_foot', 'front_right_foot', 'mid_left_foot', 'mid_right_foot', 'back_left_foot', 'back_right_foot']
    foot_colors = ['#000000', '#cc7f4d', '#00667f', '#ff667f', '#7f7f7f', '#00ff7f']
    # todo: убрать в параметры скрипта всё что ниже
    gait_name = "triple"
    gait_cycle_len = 30
    bad_contact_reward = -0.1
    log_rewards = False;

    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "mutant.xml", "torso", action_dim=12, obs_dim=38, power=2.5)
        self.feet_graph = FeetGraph(self.foot_list, self.foot_colors, 500)
        gdf = pd.read_csv('./walk_analyse/gaits.csv') #todo: убрать путь в параметры скрипта
        self.gaits = gdf.set_index('gait_name').T.to_dict()
        self.desired_contacts = generate_points(self.gaits[self.gait_name], self.gait_cycle_len)
        self.main_leg_last_contact = False;
        self.gait_step = 0;

        if self.log_rewards:
            self.f = open('./walk_analyse/reward_log.csv', 'w')  #todo: убрать путь в параметры скрипта
            fieldnames = ['alive', 'progress', 'electricity_cost', 'joints_at_limit_cost', 'feet_collision_cost',
                               'gait_reward']
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def init_cameras(self):
        self.cam1 = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "top_camera")
        self.cam2 = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "side_camera")

    def _render(self, mode, **kwargs):
        if kwargs.get("stop") is True:
            return
        if mode=="human":
            self.scene.human_render_detected = True
            return self.scene.cpp_world.test_window()
        elif mode=="rgb_array":
            self.camera_follow_top()
            rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
            rendered_rgb1 = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )

            self.camera_follow_side()
            rgb, _, _, _, _ = self.camera.render(False, False, False)  # render_depth, render_labeling, print_timing)
            rendered_rgb2 = np.fromstring(rgb, dtype=np.uint8).reshape((self.VIDEO_H, self.VIDEO_W, 3))

            rendered_rgb3 = self.feet_graph.draw_contacts(self.feet_contact)

            rendered_rgb = np.concatenate((rendered_rgb1, rendered_rgb2),axis=1)
            rendered_rgb = np.concatenate((rendered_rgb, rendered_rgb3),axis=0)
            return rendered_rgb
        else:
            assert(0)


    def HUD(self, s, a, done):
        active = self.scene.actor_is_active(self)
        if active and self.done<=2:
            self.scene.cpp_world.test_window_history_advance()
            self.scene.cpp_world.test_window_observations(s.tolist())
            self.scene.cpp_world.test_window_actions(a.tolist())
            self.scene.cpp_world.test_window_rewards(self.rewards)
        if self.done<=1: # Only post score on first time done flag is seen, keep this score if user continues to use env
            info_str = "step: %04i  reward: %07.1f %s" % (self.frame, self.reward, "DONE" if self.done else "")
            if active:
                self.scene.cpp_world.test_window_score(info_str)
            self.camera.test_window_score(info_str)  # will appear on video ("rgb_array"), but not on cameras istalled on the robot (because that would be different camera)


    def camera_follow_top(self):
        x, y, z = self.body_xyz
        self.camera.move_and_look_at(x, y, 4, x, y, 1.0)

    def camera_follow_side(self):
        x, y, z = self.body_xyz
        self.camera.move_and_look_at(x, y-3.0, z, x, y, z)

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.alive_bonus(state[0]+self.initial_z, self.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
            if contact_names - self.foot_ground_object_names:
                feet_collision_cost += self.foot_collision_cost

        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        ###############
        gait_reward = 0
        contacts = state[32:37]
        if (self.main_leg_last_contact is False and contacts[0] is True):
            self.gait_step = 0
        if self.gait_step >= self.gait_cycle_len:
            self.gait_step = 0
        self.main_leg_last_contact = contacts[0]
        for i in range(len(contacts)):
            if contacts[i] != self.desired_contacts[self.gait_step][i]:
                gait_reward += self.bad_contact_reward
        self.gait_step += 1
        ###############

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost,
            gait_reward
            ]

        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.episode_over(self.frame)
        self.done   += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)

        if self.log_rewards is True:
            row = {}
            row['alive'] = alive
            row['progress'] = progress
            row['electricity_cost'] = electricity_cost
            row['joints_at_limit_cost'] = joints_at_limit_cost
            row['feet_collision_cost'] = feet_collision_cost
            row['gait_reward'] = gait_reward
            self.writer.writerow(row)
            self.f.flush()

        return state, sum(self.rewards), bool(done), {}

    def __del__(self):
        self.f.close()



import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
class FeetGraph:
    def __init__(self, foot_list, foot_colors, length, delta=0.2):
        self.delta = delta
        self.foot_list = foot_list
        self.foot_colors = foot_colors
        self.length = length

        yspan = len(foot_list)
        self.yplaces = [.5 + i for i in range(yspan)]
        ylabels = foot_list

        self.fig = plt.figure(figsize=(12, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_yticks(self.yplaces)
        self.ax.set_yticklabels(ylabels)
        self.ax.set_ylim((0, yspan))
        self.ax.set_xlim((0, length))
        #         ax.set_xlabel('Nogi')

        xmin, xmax = self.ax.get_xlim()
        self.ax.hlines(range(1, yspan), xmin, xmax)
        self.time_step = 0

    def draw_contacts(self, contacts):
        if self.time_step > self.length:
            return self.img
        for i in range(len(self.foot_list)):
            if contacts[i] < 0.5:
                continue
            start, end = self.time_step, self.time_step + 1
            pos = self.yplaces[i]
            self.ax.add_patch(patches.Rectangle((start, pos - self.delta / 2.0), end - start, self.delta, color=self.foot_colors[i]))
        self.time_step += 1
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        self.fig.canvas.draw()
        self.img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return self.img


def generate_points(d, T):
    feet_data = np.zeros((T,6))
    d["phi1"] = 0
    for i in range(6):
        start_contact = int(T * d["phi"+str(i+1)])
        end_contact = int(T * ((d['beta'] + d["phi"+str(i+1)]) % 1))
        if start_contact > end_contact:
#             print("revert")
            for j in range(0, end_contact, 1):
                feet_data[j,i] = 1
            for j in range(end_contact, start_contact, 1):
                feet_data[j,i] = 0
            for j in range(start_contact, T, 1):
                feet_data[j,i] = 1
        else:
            for j in range(start_contact, end_contact, 1):
                feet_data[j,i] = 1
#         print(start_contact, end_contact)
    return feet_data