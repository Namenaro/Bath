# -*- coding: utf-8 -*
import os
import cv2
import random
import numpy as np

from bath.sess_hist import SessionHistory
from bath.preservation_low_reward import PreservationReward

class World:
    def __init__(self, fovea_side, retina_padding, session_time_len):
        self.PADDING = 100 # при выборе начала новой сессии, точка выбирается случайно на такой отступе от края картинки
        self.fovea_side = fovea_side
        self.retina_padding = retina_padding
        self.max_session_time = session_time_len

        self.session_history = None
        self.curr_picture = None
        self.current_coord = None
        self.current_time_in_session = 0


    def set_picture(self, path):
        assert os.path.isfile(path)
        self.curr_picture = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.curr_picture = self.curr_picture* float(1) / float(255)

    def reset(self):
        "устанавливает взгляд в случайную точку на картинке, возвращает ретину, стартует новую сессию "
        self._set_random_left_top()
        self.session_history = SessionHistory(initial_coord=self.current_coord)
        observation = self._make_observation(current_retina_as_matrix=self._get_current_retina_matrix(),
                                             last_action_as_vector=self.session_history.last_action())
        self.session_history.add(fovea_matrix=self._get_currnet_fovea_matrix(),
                                 context_matrix=self._get_current_retina_matrix(),
                                 action=np.array([0., 0.], dtype='float32'))
        self.current_time_in_session = 0
        return observation


    def step(self, action):
        done = False
        reward = 0
        self.current_time_in_session +=1
        self.current_coord += action
        self.session_history.add(fovea_matrix=self._get_currnet_fovea_matrix(),
                                 context_matrix=self._get_current_retina_matrix(),
                                 action=action)
        if self.current_time_in_session >= self.max_session_time:
            done = True
            reward = self._evaluate_revard_in_session(self.session_history)
        observation = self._make_observation(current_retina_as_matrix=self._get_current_retina_matrix(),
                                             last_action_as_vector=self.session_history.last_action())
        return observation, reward, done

    #---------------------------------------------
    def _evaluate_revard_in_session(self, session_history):
        reward_generator = PreservationReward()
        reward = reward_generator.evaluate_session(session_history=session_history)
        return reward

    def _get_currnet_fovea_matrix(self):
        return self._get_subframe(side=self.fovea_side,
                                  x=self.top_left_x,
                                  y=self.top_left_y)

    def _get_current_retina_matrix(self):
        return self._get_subframe(side=self.fovea_side+self.retina_padding,
                                  x=self.top_left_x-self.retina_padding,
                                  y=self.top_left_y-self.retina_padding)

    def _make_observation(self, current_retina_as_matrix, last_action_as_vector):
        """
        Наблюдение для сети это вектор, составленный их текущей ретины и прошлого действия
        """
        current_retina_as_vector = current_retina_as_matrix.flatten()
        observation = np.concatenate(current_retina_as_vector, last_action_as_vector)
        return observation

    def _get_subframe(self, side, x, y):
        X1 = x
        X2 = x + side
        Y1 = y
        Y2 = y + side
        return self.curr_picture[X1:X2, Y1:Y2]

    def _set_random_left_top(self):
        min_x = self.PADDING
        min_y = self.PADDING

        max_x = self.curr_picture.shape[0] - self.PADDING
        max_y = self.curr_picture.shape[1] - self.PADDING

        side = self.fovea_side + self.retina_padding
        assert max_x - min_x >= side, 'picture is too small'
        assert max_y - min_y >= side, 'picture is too small'

        top_left_x = random.randint(min_x, max_x)
        top_left_y = random.randint(min_y, max_y)
        self.current_coord = np.array([top_left_x, top_left_y], dtype='float32')

    def _is_out_of_bounds(self):
        top_left_x = self.current_coord[0]
        top_left_y = self.current_coord[1]
        pic_min_x = top_left_x - self.retina_padding
        pic_max_x = top_left_x + self.retina_padding + self.fovea_side
        pic_min_y = top_left_y - self.retina_padding
        pic_max_y = top_left_y + self.retina_padding + self.fovea_side
        if pic_min_x < 0 or pic_min_y < 0:
            return True
        shape = self.curr_picture.shape
        if pic_max_x > shape[0] or pic_max_y > shape[1]:
            return True
        return False



if __name__ == '__main__':
    picture_name = 'C:/forest2.jpg'
    seq_len = 5
    commands = []
    for i in range(seq_len):
        command = np.array([2., 1.], dtype='float32')
        commands.append(command)

    world = World(fovea_side=4, retina_padding=4, session_time_len=seq_len)
    world.set_picture(picture_name)
    world.reset()
    for command in commands:
        observation, reward, done = world.step(command)
        print('action= ' + str(command))
        if done is True:
            print('DONE! reward=' + str(reward))
