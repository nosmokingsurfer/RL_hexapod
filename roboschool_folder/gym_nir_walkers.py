from roboschool.gym_mujoco_walkers import *

class RoboschoolMutant(RoboschoolForwardWalkerMujocoXML):
    # todo: fix legs in xml
    # foot_list = ['front_left_foot', 'front_right_foot', 'mid_right_foot', 'mid_left_foot', 'back_left_foot', 'back_right_foot']

    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot', 'right_mid_foot', 'left_mid_foot']
    foot_colors = ['#000000', '#7f7f7f', '#00ff7f', '#cc7f4d', '#ff667f', '#00667f']
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "mutant.xml", "torso", action_dim=12, obs_dim=38, power=2.5)
        self.feet_graph = FeetGraph(self.foot_list, self.foot_colors, 500)
    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def init_cameras(self):
        self.cam1 = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "top_camera")
        self.cam2 = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "side_camera")

    def _render(self, mode, close):
        if close:
            return
        if mode=="human":
            self.scene.human_render_detected = True
            return self.scene.cpp_world.test_window()
        elif mode=="rgb_array":
            # change foot color if contact       ??? material tex name ???
            # for part_id, contact in enumerate(self.feet_contact):
            #     if contact > 0.5:
            #         self.parts[self.foot_list[part_id]].set_multiply_color(self.foot_list[part_id],123)
                # else:
                #     self.parts[self.foot_list[part_id]].set_multiply_color(self.foot_list[part_id], 124)


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

