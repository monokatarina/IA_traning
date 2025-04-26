import pygame
import time
from .roadmap import RoadMap
from .agent import QLearningAgent
from .constants import *
import os
import pickle
from pathlib import Path
from .metrics_window import MetricsWindow

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("IA Treinando para Navegar em Estradas")
        
        self.roadmap = RoadMap()
        self.agent_pos = self.roadmap.start_pos
        self.agent = QLearningAgent()
        self.episodes = 0
        self.steps = 0
        self.total_rewards = 0
        self.running = True
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.start_time = time.time()
        self.rewards_history = []
        self.avg_rewards_history = []
        self.metrics_window = MetricsWindow() 

    def save_game(self, filename="game_state.pkl"):
        """Salva o estado completo do jogo"""
        save_dir = Path("save")
        save_dir.mkdir(exist_ok=True)
        
        state = {
            'episodes': self.episodes,
            'total_rewards': self.total_rewards,
            'agent': {
                'q_table': dict(self.agent.q_table),
                'alpha': self.agent.alpha,
                'gamma': self.agent.gamma,
                'epsilon': self.agent.epsilon,
                'epsilon_min': self.agent.epsilon_min,
                'epsilon_decay': self.agent.epsilon_decay
            },
            'roadmap': self.roadmap,
            'agent_pos': self.agent_pos,
            'start_time': self.start_time,
            'rewards_history': self.rewards_history,
            'avg_rewards_history': self.avg_rewards_history
        }
        
        filepath = save_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Jogo salvo em {filepath}")

    def load_game(self, filename="game_state.pkl"):
        """Carrega o estado completo do jogo"""
        filepath = Path("save") / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
                self.episodes = state['episodes']
                self.total_rewards = state['total_rewards']
                
                # Recria o agente
                self.agent = QLearningAgent()
                def default_q_values():
                    return np.zeros(4)
                self.agent.q_table = defaultdict(default_q_values, state['agent']['q_table'])
                self.agent.alpha = state['agent'].get('alpha', ALPHA)
                self.agent.gamma = state['agent'].get('gamma', GAMMA)
                self.agent.epsilon = state['agent'].get('epsilon', EPSILON_START)
                self.agent.epsilon_min = state['agent'].get('epsilon_min', EPSILON_MIN)
                self.agent.epsilon_decay = state['agent'].get('epsilon_decay', EPSILON_DECAY)
                
                self.roadmap = state['roadmap']
                self.agent_pos = state['agent_pos']
                self.start_time = state['start_time']
                self.rewards_history = state.get('rewards_history', [])
                self.avg_rewards_history = state.get('avg_rewards_history', [])
                
                # Reinicializa componentes do pygame
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont(None, 36)
                self.running = True
                
            print(f"Jogo carregado de {filepath}")
            return True
        print(f"Arquivo {filepath} não encontrado")
        return False
    
    def reset(self):
        self.roadmap = RoadMap()
        self.agent_pos = self.roadmap.start_pos
        self.roadmap.current_checkpoint = 0
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        # Verificação adicional de segurança
        if len(self.roadmap.checkpoints) == 0:
            return (0, 0)  # Estado padrão se não houver checkpoints
        
        # Garante que current_checkpoint está dentro dos limites
        self.roadmap.current_checkpoint = min(self.roadmap.current_checkpoint, len(self.roadmap.checkpoints)-1)
        checkpoint_pos = self.roadmap.checkpoints[self.roadmap.current_checkpoint]
        return self.agent.get_state_key(self.agent_pos, checkpoint_pos, self.roadmap)
    
    def move_agent(self, action):
        # 0: cima, 1: direita, 2: baixo, 3: esquerda
        dr = [-1, 0, 1, 0]
        dc = [0, 1, 0, -1]
        
        new_row = self.agent_pos[0] + dr[action]
        new_col = self.agent_pos[1] + dc[action]
        
        reward = 0
        done = False
        
        # Verificar se o movimento é válido
        if not self.roadmap.is_road(new_row, new_col):
            reward = -10  # Penalidade por sair da estrada
            done = True
        else:
            self.agent_pos = (new_row, new_col)
            self.steps += 1
            reward = -0.1  # Pequena penalidade por passo para incentivar eficiência
            
            # Verificar se alcançou o checkpoint (se houver checkpoints)
            if len(self.roadmap.checkpoints) > 0 and self.agent_pos == self.roadmap.checkpoints[self.roadmap.current_checkpoint]:
                reward = 10  # Grande recompensa por alcançar o checkpoint
                self.roadmap.current_checkpoint += 1
                
                if self.roadmap.current_checkpoint >= len(self.roadmap.checkpoints):
                    reward = 20  # Recompensa máxima por completar todos os checkpoints
                    done = True
        
        return reward, done
    
    def run_episode(self, render=False):
        state = self.reset()
        total_reward = 0
        done = False
        
        while not done and self.steps < 1000: # Limitar a 1000 passos por episódio
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()  # Fecha o jogo corretamente
                    return total_reward  # Retorna early se o jogo foi fechado
            action = self.agent.choose_action(state)
            reward, done = self.move_agent(action)
            next_state = self.get_state()
            
            self.agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            if render:
                self.render()
                self.clock.tick(120)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return total_reward
        
        self.agent.decay_epsilon()
        self.episodes += 1
        self.total_rewards += total_reward
        self.rewards_history.append(total_reward)
        
        # Calcular média móvel das recompensas
        if len(self.rewards_history) >= 100:
            avg = sum(self.rewards_history[-100:]) / 100
        else:
            avg = sum(self.rewards_history) / len(self.rewards_history) if self.rewards_history else 0
        self.avg_rewards_history.append(avg)
        
        try:
            self.metrics_window.update(self.episodes, total_reward, avg, self.agent.epsilon)
        except:
            pass
        
        return total_reward
    
    def render(self):
        self.screen.fill(BLACK)
        self.roadmap.draw(self.screen)
        
        # Desenhar agente
        pygame.draw.rect(self.screen, RED, (self.agent_pos[1] * CELL_SIZE, self.agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Mostrar informações
        info_text = [
            f"Episódio: {self.episodes}",
            f"Epsilon: {self.agent.epsilon:.2f}",
            f"Tempo: {int(time.time() - self.start_time)}s"
        ]
        
        for i, text in enumerate(info_text):
            rendered_text = self.font.render(text, True, WHITE)
            self.screen.blit(rendered_text, (10, 10 + i * 30))
        
        
        pygame.display.flip()
    def close(self):
        try:
            self.metrics_window.close()
        except:
            pass
        pygame.quit()

    def train(self, episodes=1000):
        # Limpar histórico ao iniciar novo treinamento
        self.rewards_history = []
        self.avg_rewards_history = []
        
        for _ in range(episodes):
            self.run_episode(render=False)
            
            if self.episodes % 10 == 0:
                avg_reward = self.total_rewards / 100
                print(f"Episódio: {self.episodes}, Recompensa Média: {avg_reward:.1f}, Epsilon: {self.agent.epsilon:.2f}")
                self.total_rewards = 0
                
                # Renderizar um episódio para visualização
                self.run_episode(render=True)
        
        self.run_episode(render=True)
        print("Treinamento concluído!")