import random
import numpy as np
from collections import defaultdict
from .constants import *
import os
import pickle
from pathlib import Path

class QLearningAgent:
    def __init__(self):
        def default_q_values():
            return np.zeros(4)
        
        self.q_table = defaultdict(default_q_values)  # 4 ações possíveis
        self.alpha = ALPHA  # Taxa de aprendizado
        self.gamma = GAMMA  # Fator de desconto
        self.epsilon = EPSILON_START  # Taxa de exploração
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.actions = [0, 1, 2, 3]  # 0: cima, 1: direita, 2: baixo, 3: esquerda
        
    def save_model(self, filename="q_learning_model.pkl"):
        """Salva o modelo Q-table em um arquivo"""
        save_dir = Path("save")
        save_dir.mkdir(exist_ok=True)
        
        # Converte o defaultdict para um dict normal antes de salvar
        q_table_dict = dict(self.q_table)
        filepath = save_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_table_dict,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            }, f)
        print(f"Modelo salvo em {filepath}")
    
    def load_model(self, filename="q_learning_model.pkl"):
        """Carrega o modelo Q-table de um arquivo"""
        filepath = Path("save") / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                def default_q_values():
                    return np.zeros(4)
                
                self.q_table = defaultdict(default_q_values, data['q_table'])
                self.alpha = data.get('alpha', ALPHA)
                self.gamma = data.get('gamma', GAMMA)
                self.epsilon = data.get('epsilon', EPSILON_START)
                self.epsilon_min = data.get('epsilon_min', EPSILON_MIN)
                self.epsilon_decay = data.get('epsilon_decay', EPSILON_DECAY)
                
            print(f"Modelo carregado de {filepath}")
            return True
        print(f"Arquivo {filepath} não encontrado")
        return False
    
    def get_state_key(self, agent_pos, checkpoint_pos, roadmap):
        # Simplifica o estado para (dx, dy) em relação ao checkpoint
        dx = checkpoint_pos[1] - agent_pos[1]
        dy = checkpoint_pos[0] - agent_pos[0]
        
        # Discretiza a diferença (reduz o espaço de estados)
        dx = max(min(dx, 5), -5)
        dy = max(min(dy, 5), -5)
        
        return (dx, dy)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Exploração
        return np.argmax(self.q_table[state])  # Exploração
    
    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)