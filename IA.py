import pygame
import random
import numpy as np
from collections import defaultdict
import time

# Configurações
pygame.init()
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 20
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("IA Treinando para Navegar em Estradas")

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

class RoadMap:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)
        self.generate_maze()
        self.checkpoints = self.generate_checkpoints(3)  # Garante pelo menos 3 checkpoints
        self.current_checkpoint = 0
        self.start_pos = self.find_random_road_position()
        self.validate_checkpoints()  # Nova validação
    
    def validate_checkpoints(self):
        """Garante que temos checkpoints válidos"""
        if len(self.checkpoints) == 0:
            # Se não houver checkpoints, cria pelo menos um
            road_positions = [(r, c) for r in range(ROWS) for c in range(COLS) if self.grid[r][c] == 1]
            if road_positions:
                self.checkpoints.append(random.choice(road_positions))
            else:
                # Se não houver estradas, cria um checkpoint no meio
                self.grid[ROWS//2][COLS//2] = 1
                self.checkpoints.append((ROWS//2, COLS//2))
    
    def generate_maze(self):
        # Geração do labirinto usando Depth-First Search
        stack = [(random.randint(0, ROWS-1), random.randint(0, COLS-1))]
        visited = set()
        
        while stack:
            row, col = stack.pop()
            self.grid[row][col] = 1
            visited.add((row, col))
            
            neighbors = []
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = row + dr, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS and (r, c) not in visited:
                    neighbors.append((r, c))
            
            random.shuffle(neighbors)
            stack.extend(neighbors)
        
        # Conectar áreas desconectadas
        self.connect_isolated_areas()
    
    def connect_isolated_areas(self):
        # Identificar áreas isoladas e conectá-las
        visited = np.zeros((ROWS, COLS), dtype=bool)
        regions = []
        
        for row in range(ROWS):
            for col in range(COLS):
                if self.grid[row][col] == 1 and not visited[row][col]:
                    region = []
                    stack = [(row, col)]
                    visited[row][col] = True
                    
                    while stack:
                        r, c = stack.pop()
                        region.append((r, c))
                        
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < ROWS and 0 <= nc < COLS and 
                                self.grid[nr][nc] == 1 and not visited[nr][nc]):
                                visited[nr][nc] = True
                                stack.append((nr, nc))
                    
                    regions.append(region)
        
        # Conectar regiões se houver mais de uma
        if len(regions) > 1:
            for i in range(len(regions)-1):
                r1, c1 = random.choice(regions[i])
                r2, c2 = random.choice(regions[i+1])
                
                # Criar caminho entre as regiões
                while r1 != r2 or c1 != c2:
                    if r1 < r2:
                        r1 += 1
                    elif r1 > r2:
                        r1 -= 1
                    elif c1 < c2:
                        c1 += 1
                    elif c1 > c2:
                        c1 -= 1
                    
                    self.grid[r1][c1] = 1
    
    def generate_checkpoints(self, num):
        checkpoints = []
        road_positions = [(r, c) for r in range(ROWS) for c in range(COLS) if self.grid[r][c] == 1]
        
        # Garante que não tentaremos mais checkpoints que estradas disponíveis
        num = min(num, len(road_positions))
        
        for _ in range(num):
            if road_positions:
                pos = random.choice(road_positions)
                checkpoints.append(pos)
                road_positions.remove(pos)
        
        return checkpoints
    
    def find_random_road_position(self):
        road_positions = [(r, c) for r in range(ROWS) for c in range(COLS) if self.grid[r][c] == 1]
        return random.choice(road_positions) if road_positions else (ROWS//2, COLS//2)
    
    def is_road(self, row, col):
        return 0 <= row < ROWS and 0 <= col < COLS and self.grid[row][col] == 1
    
    def draw(self, screen):
        for row in range(ROWS):
            for col in range(COLS):
                if self.grid[row][col] == 1:
                    pygame.draw.rect(screen, GRAY, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Desenhar checkpoints
        for i, (row, col) in enumerate(self.checkpoints):
            if i < len(self.checkpoints):  # Verificação adicional de segurança
                color = GREEN if i == self.current_checkpoint else BLUE
                pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                font = pygame.font.SysFont(None, 20)
                text = font.render(str(i+1), True, BLACK)
                screen.blit(text, (col * CELL_SIZE + CELL_SIZE//3, row * CELL_SIZE + CELL_SIZE//4))

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 ações possíveis
        self.alpha = 0.1  # Taxa de aprendizado
        self.gamma = 0.9  # Fator de desconto
        self.epsilon = 1.0  # Taxa de exploração
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.actions = [0, 1, 2, 3]  # 0: cima, 1: direita, 2: baixo, 3: esquerda
    
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

class Game:
    def __init__(self):
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
        
        while not done and self.steps < 1000:  # Limite de passos por episódio
            action = self.agent.choose_action(state)
            reward, done = self.move_agent(action)
            next_state = self.get_state()
            
            self.agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            if render:
                self.render()
                self.clock.tick(30)  # Velocidade de renderização
        
        self.agent.decay_epsilon()
        self.episodes += 1
        self.total_rewards += total_reward
        
        return total_reward
    
    def render(self):
        screen.fill(BLACK)
        self.roadmap.draw(screen)
        
        # Desenhar agente
        pygame.draw.rect(screen, RED, 
                         (self.agent_pos[1] * CELL_SIZE, self.agent_pos[0] * CELL_SIZE, 
                          CELL_SIZE, CELL_SIZE))
        
        # Mostrar informações
        info_text = [
            f"Episódio: {self.episodes}",
            f"Epsilon: {self.agent.epsilon:.2f}",
            f"Checkpoint: {min(self.roadmap.current_checkpoint+1, len(self.roadmap.checkpoints))}/{len(self.roadmap.checkpoints)}",
            f"Recompensa Média: {self.total_rewards/max(1, self.episodes):.1f}",
            f"Tempo: {int(time.time() - self.start_time)}s"
        ]
        
        for i, text in enumerate(info_text):
            rendered_text = self.font.render(text, True, WHITE)
            screen.blit(rendered_text, (10, 10 + i * 30))
        
        pygame.display.flip()
    
    def train(self, episodes=1000):
        for _ in range(episodes):
            self.run_episode(render=False)
            
            # Mostrar progresso a cada 100 episódios
            if self.episodes % 10 == 0:
                avg_reward = self.total_rewards / 100
                print(f"Episódio: {self.episodes}, Recompensa Média: {avg_reward:.1f}, Epsilon: {self.agent.epsilon:.2f}")
                self.total_rewards = 0
                
                # Renderizar um episódio para visualização
                self.run_episode(render=True)
        
        # Após o treinamento, mostrar o desempenho final
        self.run_episode(render=True)
        print("Treinamento concluído!")

def main():
    game = Game()
    
    # Treinar a IA (pode levar alguns minutos)
    game.train(episodes=1000)
    
    # Manter a janela aberta após o treinamento
    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
        
        game.run_episode(render=True)
        game.clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()