from roboschool.gym_mujoco_walkers import *
import pandas as pd
from copy import copy
import csv


class RoboschoolMutant(RoboschoolForwardWalkerMujocoXML):
    foot_list = ['front_left_foot', 'front_right_foot', 'mid_left_foot', 'mid_right_foot', 'back_left_foot',
                 'back_right_foot']
    foot_colors = ['#000000', '#cc7f4d', '#00667f', '#ff667f', '#7f7f7f', '#00ff7f']
    # Defaults:
    gaits_config_path = './walk_analyse/'
    out_path = './walk_analyse/'
    gait_name = None
    gait_cycle_len = 30
    contact_reward = 0.5
    use_reward = [True for i in range(7)]
    log_rewards = False;
    render_mode = 0;
    correct_step_call = False

    gl_step = 0

    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "mutant.xml", "torso", action_dim=12, obs_dim=39, power=2.5)
        self.feet_graph = FeetGraph(self.foot_list, self.foot_colors, 500)

    def set_params(self, gaits_config_path='./walk_analyse/', gait_name=None, gait_cycle_len=30,
                   out_path='./walk_analyse/', log_rewards=False, render_mode=0, reward_mask=63, contact_reward=0.5):
        self.gaits_config_path = gaits_config_path
        self.gait_name = gait_name
        self.gait_cycle_len = gait_cycle_len
        self.out_path = out_path
        self.log_rewards = log_rewards
        self.render_mode = render_mode
        # self.use_reward = [reward_mask & 1, reward_mask & 2, reward_mask & 4, reward_mask & 8, reward_mask & 16, reward_mask & 32]
        self.use_reward = [((reward_mask & (2 ** i)) != 0) for i in range(7)]
        self.contact_reward = contact_reward

        if self.gait_name is not None:
            gdf = pd.read_csv(os.path.join(self.gaits_config_path, 'gaits.csv'))
            self.gaits = gdf.set_index('gait_name').T.to_dict()
            self.desired_contacts = generate_points(self.gaits[self.gait_name], self.gait_cycle_len)
            self.phase_map = generate_phase_map(self.desired_contacts)
            # self.ground_rewards = make_smooth_reward(self.desired_contacts)
            # self.air_rewards = make_smooth_reward(invert_gait(self.desired_contacts))
            self.main_leg_last_contact = False
            self.gait_step = 0
            self.last_phase = 0
            self.phase_time = 0
            self.phase_time_limit = 250

        if self.log_rewards:
            self.f = open(os.path.join(self.out_path, 'reward_log.csv'), 'w')
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
        if mode == "human":
            self.scene.human_render_detected = True
            return self.scene.cpp_world.test_window()
        elif mode == "rgb_array":
            if self.render_mode == 0:
                return None

            self.camera_follow_side()
            rgb, _, _, _, _ = self.camera.render(False, False, False)  # render_depth, render_labeling, print_timing)
            rendered_rgb1 = np.fromstring(rgb, dtype=np.uint8).reshape((self.VIDEO_H, self.VIDEO_W, 3))

            if self.render_mode == 2:
                self.camera_follow_top()
                rgb, _, _, _, _ = self.camera.render(False, False, False)
                rendered_rgb2 = np.fromstring(rgb, dtype=np.uint8).reshape((self.VIDEO_H, self.VIDEO_W, 3))
                rendered_rgb3 = self.feet_graph.draw_contacts(self.feet_contact)
                rendered_rgb = np.concatenate((rendered_rgb2, rendered_rgb1), axis=1)
                rendered_rgb = np.concatenate((rendered_rgb, rendered_rgb3), axis=0)
                return rendered_rgb
            return rendered_rgb1

        else:
            assert (0)

    def HUD(self, s, a, done):
        active = self.scene.actor_is_active(self)
        if active and self.done <= 2:
            self.scene.cpp_world.test_window_history_advance()
            self.scene.cpp_world.test_window_observations(s.tolist())
            self.scene.cpp_world.test_window_actions(a.tolist())
            self.scene.cpp_world.test_window_rewards(self.rewards)
        if self.done <= 1:  # Only post score on first time done flag is seen, keep this score if user continues to use env
            info_str = "step: %04i  reward: %07.1f %s" % (self.frame, self.reward, "DONE" if self.done else "")
            # info_str = "%02.2f %02.2f %02.2f %02.2f %02.2f %02.2f s: %04i" % (a[1], a[3], a[5], a[7], a[9], a[11], self.frame)
            if active:
                self.scene.cpp_world.test_window_score(info_str)
            self.camera.test_window_score(
                info_str)  # will appear on video ("rgb_array"), but not on cameras istalled on the robot (because that would be different camera)

    def camera_follow_top(self):
        x, y, z = self.body_xyz
        self.camera.move_and_look_at(x, y, 4, x, y, 1.0)

    def camera_follow_side(self):
        x, y, z = self.body_xyz
        self.camera.move_and_look_at(x, y - 3.0, z, x, y, z)

    def get_test_action(self):
        a = [0.0 for _ in range(12)]
        for i in range(1, 12, 2):
            a[i] = np.sin(self.gl_step / 10)

        # sec = int(self.gl_step / 60)
        # lid = (2 * sec) + 1
        # if lid < 12:
        #     a[lid] = -2 #((-1) ** (sec % 2))
        self.gl_step += 1
        return a


    ## wtf, is needed for google.colab
    def _seed(self, seed=None):
        return super(RoboschoolMutant, self)._seed(seed=None)

    def _reset(self):
        s = super(RoboschoolMutant, self)._reset()
        self.phase_time = 0
        self.gait_step = 0
        return np.append(s, 0)


    def _step(self, a):
        # a = np.array(self.get_test_action())
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive = 0
        progress = 0
        electricity_cost = 0
        joints_at_limit_cost = 0
        feet_collision_cost = 0.0
        gait_reward = 0
        self.correct_step_call = True

        alive = float(self.alive_bonus(state[0] + self.initial_z,
                                       self.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True


        if not self.use_reward[0]:
            alive = 0

        if self.use_reward[1]:
            potential_old = self.potential
            self.potential = self.calc_potential()
            progress = float(self.potential - potential_old)
            # if progress > 1.4:
            #     progress = 1.4

        # feet_collision_cost = 0.0
        if self.use_reward[2]:
            for i, f in enumerate(self.feet):
                contact_names = set(x.name for x in f.contact_list())
                # print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
                self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
                if contact_names - self.foot_ground_object_names:
                    feet_collision_cost += self.foot_collision_cost

        if self.use_reward[3]:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        if self.use_reward[4]:
            joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        ############### # complex speed-contact reward
        true_hits = 0
        if self.use_reward[5]:
            if self.gait_name is not None:
                contacts = state[32:38]
                # if (self.main_leg_last_contact is False and contacts[0] is True):
                #     self.gait_step = 0
                if self.gait_step >= self.gait_cycle_len:
                    self.gait_step = 0
                state = np.append(state, self.phase_map[self.gait_step])
                self.main_leg_last_contact = contacts[0]
                for i in range(len(contacts)):
                    joint_id = (2 * i) + 1
                    desired_contact = self.desired_contacts[self.gait_step][i]
                    if contacts[i] == desired_contact:
                        gait_reward += self.contact_reward * 1.1
                        true_hits += 1
                    else:
                        # legs 4 and 5 have inverted control signals (need to fix xml)
                        if i > 3:  # + is up, - is down:
                            if (desired_contact == 1 and a[joint_id] < 0) or (desired_contact == 0 and a[joint_id] > 0):
                                gait_reward += np.clip(np.abs(a[joint_id]), 0, 1) * self.contact_reward
                        else:
                            # - is up, + is down
                            if (desired_contact == 1 and a[joint_id] > 0) or (desired_contact == 0 and a[joint_id] < 0):
                                gait_reward += np.clip(np.abs(a[joint_id]), 0, 1) * self.contact_reward
                if true_hits == len(contacts):
                    self.gait_step += 1
                if self.last_phase == self.phase_map[self.gait_step]:
                    self.phase_time += 1
                else:
                    self.last_phase == self.phase_map[self.gait_step]
                    self.phase_time = 0
                if self.phase_time > self.phase_time_limit:
                    done = True
        # progress = 0
        ###############

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost,
            gait_reward
        ]

        self.frame += 1
        if (done and not self.done) or self.frame == self.spec.timestep_limit:
            self.episode_over(self.frame)
        self.done += done  # 2 == 1+True
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
        if self.log_rewards is True:
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
            self.ax.add_patch(
                patches.Rectangle((start, pos - self.delta / 2.0), end - start, self.delta, color=self.foot_colors[i]))
        self.time_step += 1
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        self.fig.canvas.draw()
        self.img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return self.img


def generate_points(d, T):
    feet_data = np.zeros((T, 6))
    d["phi1"] = 0
    for i in range(6):
        start_contact = int(T * d["phi" + str(i + 1)])
        end_contact = int(T * ((d['beta'] + d["phi" + str(i + 1)]) % 1))
        if start_contact > end_contact:
            for j in range(0, end_contact, 1):
                feet_data[j, i] = 1
            for j in range(end_contact, start_contact, 1):
                feet_data[j, i] = 0
            for j in range(start_contact, T, 1):
                feet_data[j, i] = 1
        else:
            for j in range(start_contact, end_contact, 1):
                feet_data[j, i] = 1
    return feet_data


def generate_phase_map(gait_points):
    phase = 0
    ph_map = [phase]
    pos = gait_points[0]
    for i in range(1, len(gait_points), 1):
        same = True
        for j in range(len(gait_points[i])):
            same = same and (gait_points[i][j] == gait_points[i-1][j])
        if not same:
            phase += 1
        ph_map.append(phase)
    return ph_map

# rendering in script params


def invert_gait(gait_points):
    inverted_gait = copy(gait_points)
    for i in range(len(inverted_gait)):
        for j in range(len(inverted_gait[i])):
            inverted_gait[i][j] = 1 - inverted_gait[i][j]
    return inverted_gait


def make_smooth_reward(gait_points):
    reward_matrix = copy(gait_points)
    length = len(reward_matrix[:, 0])

    for i in range(len(reward_matrix[0])):
        z_len = length - int(sum(reward_matrix[:, i]))
        d = 1.0 / (z_len / 2)
        border = 0

        if reward_matrix[:, i][0] == 0:
            while reward_matrix[:, i][border] != 1:
                border += 1
        else:
            while reward_matrix[:, i][border] != 0:
                border += 1
            border -= (length - z_len)
            # if border < 0:
            #     border += length

        z_pos = border - int(z_len / 2) - 1
        for j in range(int(length / 2) + 1):
            if abs(z_pos + j) < length:
                reward_matrix[:, i][z_pos + j] = d * j # round(d * j, 2)
            if abs(z_pos - j) < length:
                reward_matrix[:, i][z_pos - j] = d * j # round(d * j, 2)
    return reward_matrix
