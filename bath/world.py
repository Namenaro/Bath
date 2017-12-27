# -*- coding: utf-8 -*
import os
import cv2
import random
import numpy as np

class SessionHistory:
    def __init__(self, initial_coord):
        self.foveals = []  # матрица  [0,1] значений
        self.contexts = [] # матрица  [0,1] значений
        self.initial_coord =  initial_coord # коордиината начала траектории на картинке
        self.actions = []

    def add(self, fovea_matrix, context_matrix, action):
        self.foveals.append(fovea_matrix)
        self.contexts.append(context_matrix)
        self.actions.append(action)

    def size(self):
        return len(self.foveals)

    def get_foveal_dataset(self):
        dataset = np.array(self.foveals, dtype='float32')
        flatten_len = np.prod(self.foveals[0].shape[1:]) # если матрица 3 на 3, то будет 9
        dataset = dataset.reshape((len(dataset), flatten_len))
        return dataset

    def last_action(self):
        if self.size() > 0:
            return self.actions[0]
        else:
            return np.array([0., 0.], dtype='float32')

    def visualise_all_on_picture(self, pictue_name, side):
        curr_picture = cv2.imread(pictue_name, cv2.IMREAD_GRAYSCALE)
        curr_picture = curr_picture * float(1) / float(255)
        self.visualise_on_01_pic(curr_picture)

    def visualise_on_01_pic(self, pictue_01, side):
        top_left_x = self.initial_coord[0]
        top_left_y = self.initial_coord[1]
        for i in range(self.size()):
            cv2.rectangle(pictue_01,
                          pt1=(top_left_x, top_left_y),
                          pt2=(top_left_x + side, top_left_y + side),
                          color=(255, 0, 0))
            top_left_x += self.actions[i][0]
            top_left_y += self.actions[i][1]

class World:
    """
    Наблюдение для сети это вектор, составленный их текущей ретины и прошлого действия
    """
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
        command = [2, 1]
        commands.append(command)

    world = World(fovea_side=4, retina_padding=4)
    world.set_picture(picture_name)
    world.run_session()
    for command in commands:
        world.action(command)
    world.print_history()
    world.visualise_history()
    world.save_history(filename='pr1.pkl')