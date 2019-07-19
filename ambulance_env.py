
'''
Shenggong Ji, Yu Zheng, Zhaoyuan Wang, Tianrui Li
A Deep Reinforcement Learning-Enabled Dynamic Redeployment System for Mobile Ambulances
IMWUT/UbiComp 2019
---------------------------------------------------------------------------------------
'''

from Utility import Utility
import numpy as np


class gameEnv:

    def __init__(self):

        # simulation settings
        self.num_ambulances = 60  # number of ambulances used in the simulation
        # speed of an ambulance at a road segment = ratio_of_speed_limit * speed limit of the road segment
        self.ratio_of_speed_limit = 0.6
        # the probability of a patient being transported to a hospital
        self.pro_to_hospital = 0.8
        # the length of time a patient is treated in scene
        # if not being transported to a hospital
        self.in_site_treat_time = 10 * 60
        self.time_threshold = 10 * 60  # the pickup time threshold for patients

        # data
        self.requests = []  # EMS request
        self.hospitals = []  # hospital
        self.vertex_travel_time = {}  # vertex-to-vertex travel time
        self.stations = []  # stations
        self.ambulances = []  # ambulances

        # variables needed to run simulations
        self.time_list = []  # event-trigger time list
        self.ambu_list = []  # the corresponding ambulance id for an event
        self.cur_time_slot = 0  # the current time slot
        self.cur_request_id = 0  # the current EMS request id
        self.end_request_id = 0  # the last EMS request id
        self.first_initialization = True  # if true, load in data (just once)
        self.done = False  # denotes whether one simulation ends or not
        self.cur_reward = 0.0  # reward of the current step
        self.cur_ambu_for_action = -1  # current ambulance id for redeployment
        # initial allocation of ambulances to stations
        self.station_initial_ambulance_prob = []

        # record performance
        self.pickup_times = []  # pickup time of each EMS request
        self.pickup_ratio = []  # ratio of EMS requests picked up within self.time_threshold

        # state setting
        self.num_periods = 1  # parameter m in the paper
        self.num_occupied = 1  # parameter k in the paper
        # whether to consider each factor (4 factors in total)
        self.factors_selected = [0, 0, 0, 0]
        # the length of each factor
        self.factors_len = [self.num_periods, 1, 1, self.num_occupied]

        self.data_folder = 'data'

    def reset(self, _cur_req_id, _end_req_id):

        if self.first_initialization:

            self.first_initialization = False

            self.load_in_hospitals()
            self.load_in_requests()
            self.load_in_vertex_travel_time()
            self.load_in_stations()

            self.initial_ambulances()

        else:

            for j in range(len(self.stations)):

                self.stations[j].ambu = []
                self.stations[j].num_ambu = 0

            self.ambulances = []

            self.initial_ambulances()

        self.done = False

        self.pickup_times = []
        self.pickup_ratio = []

        self.cur_request_id = _cur_req_id
        self.end_request_id = _end_req_id

        self.time_list = []
        self.ambu_list = []
        self.time_list.append(self.requests[self.cur_request_id][0])
        self.ambu_list.append(-1)

        self.transition_to_next_state()

        s = self.return_current_state()

        return s

    def step(self, action):

        self.stations[action].num_ambu += 1

        sta_loc = self.stations[action].vertex
        ambu_id = self.cur_ambu_for_action

        ambu_cur_arr_id = self.ambulances[ambu_id].arr_id
        ambu_cur_loc = self.ambulances[ambu_id].arr[ambu_cur_arr_id][0]

        _travel_time = self.vertex_travel_time[str(
            ambu_cur_loc)+','+str(sta_loc)]
        self.ambulances[ambu_id].arr.append(
            [sta_loc, self.cur_time_slot+_travel_time, 'station', action])

        self.time_list.append(self.cur_time_slot+_travel_time)
        self.ambu_list.append(ambu_id)

        self.transition_to_next_state()

        if not self.done:
            s = self.return_current_state()
            return s, self.cur_reward, self.done
        else:
            return [], self.cur_reward, self.done

    def transition_to_next_state(self):

        ready_to_redeploy = False
        self.cur_reward = 0.0

        while not ready_to_redeploy:

            if len(self.time_list) == 0:
                self.done = True
                break

            self.cur_time_slot = min(self.time_list)

            _id = self.time_list.index(self.cur_time_slot)
            ambu_id = self.ambu_list[_id]

            del self.time_list[_id]
            del self.ambu_list[_id]

            if ambu_id == -1:  # a request comes

                s_id = self.find_nearest_station_with_available_ambulances()

                if s_id != -1:

                    a_id = self.stations[s_id].ambu[0]
                    del self.stations[s_id].ambu[0]
                    self.stations[s_id].num_ambu -= 1

                    vertex_2 = self.stations[s_id].vertex
                    req_ver = self.requests[self.cur_request_id][1]
                    _travel_time = self.vertex_travel_time[str(
                        vertex_2)+','+str(req_ver)]

                    self.pickup_times.append(_travel_time)

                    if _travel_time < self.time_threshold:
                        self.pickup_ratio.append(1.0)
                        self.cur_reward += 1.0
                    else:
                        self.pickup_ratio.append(0.0)

                    self.ambulances[a_id].arr.append(
                        [req_ver, self.cur_time_slot+_travel_time, 'scene', self.cur_time_slot])

                    self.time_list.append(self.cur_time_slot+_travel_time)
                    self.ambu_list.append(a_id)

                # append a new request
                self.cur_request_id += 1
                if self.cur_request_id < self.end_request_id:
                    self.time_list.append(
                        self.requests[self.cur_request_id][0])
                    self.ambu_list.append(-1)

            else:  # update state of an ambulance

                self.ambulances[ambu_id].arr_id += 1

                ambu_cur_arr_id = self.ambulances[ambu_id].arr_id

                ambu_cur_loc = self.ambulances[ambu_id].arr[ambu_cur_arr_id][0]

                if self.ambulances[ambu_id].arr[ambu_cur_arr_id][2] == 'scene':

                    if np.random.random() < self.pro_to_hospital:  # go to hospital

                        hos_id = self.select_a_hospital_3(ambu_cur_loc)
                        hos_loc = self.hospitals[hos_id]

                        _travel_time = self.vertex_travel_time[str(
                            ambu_cur_loc)+','+str(hos_loc)]
                        self.ambulances[ambu_id].arr.append(
                            [hos_loc, self.cur_time_slot+_travel_time, 'redeploy'])

                        self.time_list.append(self.cur_time_slot+_travel_time)
                        self.ambu_list.append(ambu_id)

                    else:  # in-site treatment

                        self.ambulances[ambu_id].arr.append(
                            [ambu_cur_loc, self.cur_time_slot+self.in_site_treat_time, 'redeploy'])

                        self.time_list.append(
                            self.cur_time_slot+self.in_site_treat_time)
                        self.ambu_list.append(ambu_id)

                if self.ambulances[ambu_id].arr[ambu_cur_arr_id][2] == 'redeploy':

                    self.cur_ambu_for_action = ambu_id
                    ready_to_redeploy = True

                if self.ambulances[ambu_id].arr[ambu_cur_arr_id][2] == 'station':

                    cur_sta_id = self.ambulances[ambu_id].arr[ambu_cur_arr_id][3]
                    self.stations[cur_sta_id].ambu.append(ambu_id)

    def load_in_hospitals(self):

        f = open(self.data_folder + '/hospitals.txt', 'r')
        for line in f:
            self.hospitals.append(int(line))
        f.close()

    def load_in_requests(self):

        ref_time = '2014-10-01 00:00:00'

        f = open(self.data_folder + '/ems_requests.txt', 'r')
        for line in f:
            items = line.split(',')
            _time = int(Utility.get_elapsed_seconds(
                Utility.string2time(ref_time), Utility.string2time(items[0])))
            self.requests.append([_time, int(items[1])])
        f.close()

    def load_in_vertex_travel_time(self):

        f = open(self.data_folder + '/travel_time_between_road_vertex.txt', 'r')
        for line in f:
            s = line.split(',')
            self.vertex_travel_time[s[0]+','+s[1]
                                    ] = float(s[2]) / self.ratio_of_speed_limit
        f.close()

    def load_in_stations(self):

        f = open(self.data_folder + '/stations.txt', 'r')
        for line in f:
            _station = station()
            _station.vertex = int(line)
            self.stations.append(_station)
        f.close()

        f = open(self.data_folder + '/each-station-lambda-31days.csv', 'r')
        i = -1

        for line in f:

            i += 1
            items = line.split(',')
            for j in items:
                self.stations[i].request_rate.append(float(j))

            self.station_initial_ambulance_prob.append(
                sum(self.stations[i].request_rate))

        f.close()

        _sum = sum(self.station_initial_ambulance_prob)
        for i in range(len(self.station_initial_ambulance_prob)):
            self.station_initial_ambulance_prob[i] /= _sum

    def initial_ambulances(self):

        for i in range(self.num_ambulances):  # for each ambulance

            # select an ambulance based on self.station_initial_ambulance_prob
            _id = np.random.choice(
                range(len(self.stations)), p=self.station_initial_ambulance_prob)

            self.stations[_id].ambu.append(i)
            self.stations[_id].num_ambu += 1

            _ambu = ambulance()
            _ambu.arr.append([self.stations[_id].vertex, 0, 'station', _id])

            self.ambulances.append(_ambu)

    def find_nearest_station_with_available_ambulances(self):

        min_travel_time = float("inf")
        selected_station_id = -1

        request_vertex = self.requests[self.cur_request_id][1]

        for i in range(len(self.stations)):

            if len(self.stations[i].ambu) == 0:
                continue

            i_vertex = self.stations[i].vertex
            i_time = self.vertex_travel_time[str(
                i_vertex)+','+str(request_vertex)]

            if min_travel_time > i_time:
                min_travel_time = i_time
                selected_station_id = i

        return selected_station_id

    def select_a_hospital_3(self, ambu_cur_loc):

        prob = []
        _max = -1

        for i in range(len(self.hospitals)):

            i_time = self.vertex_travel_time[str(
                ambu_cur_loc)+','+str(self.hospitals[i])]
            prob.append(i_time)

            if _max < i_time:
                _max = i_time

        _sum = 0.0
        for i in range(len(self.hospitals)):
            if prob[i] != 0:
                prob[i] = _max - prob[i]
            _sum += prob[i]

        for i in range(len(self.hospitals)):
            prob[i] /= _sum

        return np.random.choice(range(len(self.hospitals)), p=prob)

    def return_current_state(self):

        s = []

        t_id_0 = self.cur_time_slot // 3600
        alpha = (self.cur_time_slot - t_id_0 * 3600) / 3600.0

        ambu_id = self.cur_ambu_for_action
        ambu_arr_id = self.ambulances[ambu_id].arr_id
        ambu_loc = self.ambulances[ambu_id].arr[ambu_arr_id][0]

        for j in range(len(self.stations)):

            if self.factors_selected[0] == 1:
                for _t in range(self.num_periods):

                    t_id_1 = int(t_id_0 + _t) % (31 * 24)
                    t_id_2 = int(t_id_0 + _t + 1) % (31 * 24)

                    j_lambda = self.stations[j].request_rate[t_id_1] + alpha*(self.stations[j].request_rate[t_id_2] -
                                                                              self.stations[j].request_rate[t_id_1])

                    s.append(j_lambda)

            if self.factors_selected[1] == 1:
                j_n = self.stations[j].num_ambu
                s.append(j_n)

            if self.factors_selected[2] == 1:
                j_t = self.vertex_travel_time[str(
                    ambu_loc)+','+str(self.stations[j].vertex)]
                # j_t = self.add_tra_time_est_err(j_t)
                s.append(j_t / 3600.0)

            if self.factors_selected[3] == 1:
                tmp_time = np.ones((self.num_occupied)) * 7200.0
                for i in range(len(self.ambulances)):
                    if i == self.cur_ambu_for_action:
                        continue
                    if self.ambulances[i].arr[-1][2] == 'redeploy':
                        i_loc = self.ambulances[i].arr[-1][0]
                        i_t1 = self.ambulances[i].arr[-1][1] - \
                            self.cur_time_slot
                        i_tij = i_t1 + \
                            self.vertex_travel_time[str(
                                i_loc)+','+str(self.stations[j].vertex)]
                        # i_tij = self.add_tra_time_est_err(i_tij)

                        _max_id = np.argmax(tmp_time)
                        if i_tij < tmp_time[_max_id]:
                            tmp_time[_max_id] = i_tij

                for __t in tmp_time:
                    s.append(__t / 3600.0)

        return s

    def return_total_factor_length(self):

        self.factors_len = [self.num_periods, 1, 1, self.num_occupied]
        _len = 0
        for i in range(len(self.factors_selected)):
            _len += self.factors_selected[i] * self.factors_len[i]

        return _len


class ambulance:

    def __init__(self):

        self.arr = []  # arrangement
        self.arr_id = 0


class station:

    def __init__(self):

        self.vertex = -1
        self.request_rate = []
        self.ambu = []
        self.num_ambu = 0


if __name__ == "__main__":
    env = gameEnv()
    env.reset(0, 14011)
