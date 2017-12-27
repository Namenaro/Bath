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
        self.actions = []  # вектор флоат значений

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