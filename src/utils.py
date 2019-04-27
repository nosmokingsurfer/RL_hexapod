"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import os
import shutil
import glob
import csv


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means

    def save(self, path):
        with open(os.path.join(path, 'scaler.csv'), 'w') as f:
            fieldnames = ['m'] + [('vars_' + str(i)) for i in range(len(self.vars))] + [('means_' + str(i)) for i
                                                                                                   in range(len(self.means))]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            row = {}
            for j, vs in enumerate(self.vars):
                row['vars_' + str(j)] = vs
            for j, ms in enumerate(self.means):
                row['means_' + str(j)] = ms
            row['m'] = self.m
            writer.writerow(row)
            f.flush()

    def load(self, path, obs_dim):
        with open(os.path.join(path, 'scaler.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.m = int(row["m"])
                self.vars = np.array([float(vs) for vs in [row["vars_" + str(i)] for i in range(obs_dim)]])
                self.means = np.array([float(ms) for ms in [row["means_" + str(i)] for i in range(obs_dim)]])
                self.first_pass = False



class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, now, out_path):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        if out_path is None:
            path = os.path.join('log-files', logname, now)
        else:
            path = os.path.join(out_path, logname, now)
        os.makedirs(path)
        # filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        # for filename in filenames:     # for reference
        #     shutil.copy(filename, path)

        self.path = path
        self.write_header = True
        self.write_header_trj = True
        self.log_entry = {}
        self.f = open(os.path.join(path, 'log.csv'), 'w')
        self.f_trj = open(os.path.join(path, 'log_trajectories.csv'), 'w')
        self.writer = None  # DictWriter created with first call to write() method
        self.writer_trj = None  # DictWriter created with first call to write() method

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        toplist = ['_Episode', 'Steps', '_full_train_time', '_MeanReward', '_time_simulation', '_time_policy_train', '_time_value_train']

        if display:
            self.disp(self.log_entry)
        if self.write_header:
            self.fieldnames = [x for x in self.log_entry.keys() if x not in toplist]
            self.fieldnames = toplist + self. fieldnames
            self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}
        self.reopen_log()

    def reopen_log(self):
        self.f.flush()
        self.f.close()
        self.f = open(os.path.join(self.path, 'log.csv'), 'a')
        self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
                                                               log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('Mean alive:  {:5.2f}'.format(log['_m_alive']))
        print('Mean progress:  {:5.2f}'.format(log['_m_progress']))
        print('Mean electricity_cost:  {:5.2f}'.format(log['_m_electricity_cost']))
        print('Mean joints_at_limit_cost:  {:5.2f}'.format(log['_m_joints_at_limit_cost']))
        print('Mean feet_collision_cost:  {:5.2f}'.format(log['_m_feet_collision_cost']))
        print('Mean gait_reward:  {:5.2f}'.format(log['_m_gait_reward']))

        print('Batch Simulation time : {}'.format(log['_time_simulation']))
        print('Batch Policy train time: {}'.format(log['_time_policy_train']))
        print('Batch Value train time: {}'.format(log['_time_value_train']))
        print('Total time spent: {}'.format(log['_full_train_time']))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def log_trajectory(self, trajectory):
        obs_dim = len(trajectory["observes"][0])
        act_dim = len(trajectory["actions"][0])
        if self.write_header_trj:
            fieldnames = ['episode'] + ['reward'] + [('act_' + str(i)) for i in range(act_dim)] + [('obs_' + str(i)) for i in range(obs_dim)]
            self.writer_trj = csv.DictWriter(self.f_trj , fieldnames=fieldnames)
            self.writer_trj.writeheader()
            self.write_header_trj = False

        for i in range(len(trajectory["observes"])):
            row = {}
            for j, obs in enumerate(trajectory["observes"][i]):
                row['obs_' + str(j)] = obs
            for j, act in enumerate(trajectory["actions"][i]):
                row['act_' + str(j)] = act
            row["reward"] = trajectory["rewards"][i]
            row['episode'] = i
            self.writer_trj.writerow(row)
            self.f_trj.flush()

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()
        self.f_trj.close()
