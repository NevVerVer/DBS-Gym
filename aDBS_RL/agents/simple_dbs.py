import numpy as np


# class HFDBS():
#     def __init__(self, action: float):
#         """
#         A naive agent that always returns the same action.
#         """
#         self.action = action

#     def predict(self, observation, state=None, episode_start=None, deterministic=True):
#         batch_size = observation.shape[0]
#         actions = np.full((batch_size,), self.action, dtype=np.float32)
#         return [actions], None 

class HFDBS():
    def __init__(self, action: float):
        """
        A naive agent that always returns the same action.
        """
        self.action = action
    def predict(self, observation, state=None,
                episode_start=None, deterministic=True):
        return [[self.action]], None
    

class RandomDBS():
    def __init__(self, action_magnitude: float):
        """
        A naive agent that always returns a random action.
        params:
            action_magnitude: float - must be a strictly positive number.
        """
        self.action_magnitude = action_magnitude
        assert self.action_magnitude > 0

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        batch_size = observation.shape[0]
        actions = np.random.uniform(-self.action_magnitude, self.action_magnitude, size=(batch_size,)).astype(np.float32)
        return [actions], None 
    

class PIDController:
    def __init__(self, 
                 Kp_init, 
                 Ki_init, 
                 Kd_init, 
                 dt, 
                 env,
                 u_max = 1.,
                 u_min = -1.,
                 reward = 'bbpow',
                 ):
        """
        reward : str - type of reward from {'bbpow', 'temp', 'thr'}
        """
        self.Kp = Kp_init  
        self.Ki = Ki_init  
        self.Kd = Kd_init
        self.dt = dt
        self.u_max = u_max
        self.u_min = u_min
        self.action = 0
        self.integral = 0
        self.prev_error = 1
        self.reward = reward
        self.env = env


    def compute(self, error):
        # Compute PID terms
        self.integral = self.integral + error * self.dt #if self.integral >=0 else self.integral - error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.dt != 0 else 0.0

        # PID control output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return np.clip(output, self.u_min, self.u_max)
    
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        x_state = observation.ravel()
        if self.reward == 'bbpow':
            e = -self.env.reward_bbpow_action(x_state, [self.action]) 
        elif self.reward == 'temp':
            e = -self.env.reward_temp_const_lfp_betafilt_action(x_state, [self.action])
        elif self.reward == 'thr':
            e = -self.env.reward_bbpow_threth_action(x_state, [self.action])
        else:
            raise NotImplementedError()
        self.action = self.compute(e)
        # print(e, self.action, observation.shape)
        batch_size = observation.shape[0]
        actions = np.full((batch_size,), self.action, dtype=np.float32)
        return [actions], None     
