import os
import pickle

class Memory:
    def __init__(self):
        self.buffer = {"actions": [], "states": [], "logprobs": [], "rewards": [],
                        "dones": [], "next_states": []}

    def save_eps(self, state, reward, action, done, next_state):
        self.buffer["rewards"].append(reward)
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["dones"].append(float(done))
        self.buffer["next_states"].append(next_state)

    def clearMemory(self):
        self.buffer = {}

    def saveBuffer(self):
        with open('saved_buffer.pkl', 'wb') as f:
            pickle.dump(self.buffer, f)

    def deleteBuffer(self):
        os.remove('saved_buffer.pkl')

    def loadBuffer(self):
        with open('saved_buffer.pkl', 'rb') as f:
            self.buffer = pickle.load(f)
