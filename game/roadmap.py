import random
import numpy as np
from collections import defaultdict
from .constants import *



class RoadMap:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)
        self.generate_maze()
        self.checkpoints = self.generate_checkpoints(3)
        self.current_checkpoint = 0
        self.start_pos = self.find_random_road_position()
        self.validate_checkpoints()
    
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
        import pygame
        # Desenhar fundo
        screen.fill(BLACK)
        
        # Textura para as paredes (opcional)
        wall_texture = pygame.Surface((CELL_SIZE, CELL_SIZE))
        wall_texture.fill((50, 50, 50))  # Cor base
        # Adiciona padrão de tijolos
        for i in range(0, CELL_SIZE, 2):
            pygame.draw.line(wall_texture, (70, 70, 70), (0, i), (CELL_SIZE, i), 1)
        
        # Desenhar caminhos (estradas)
        road_surface = pygame.Surface((CELL_SIZE-2, CELL_SIZE-2))  # Um pouco menor que a célula
        road_surface.fill((100, 100, 100))  # Cor cinza para a estrada
        # Adiciona marcações de estrada
        pygame.draw.line(road_surface, (150, 150, 150), 
                    (CELL_SIZE//4, CELL_SIZE//2), 
                    (3*CELL_SIZE//4, CELL_SIZE//2), 2)
        
        # Desenhar o grid
        for row in range(ROWS):
            for col in range(COLS):
                if self.grid[row][col] == 1:  # Caminho
                    screen.blit(road_surface, (col * CELL_SIZE + 1, row * CELL_SIZE + 1))
                else:  # Parede
                    screen.blit(wall_texture, (col * CELL_SIZE, row * CELL_SIZE))
        
        # Desenhar checkpoints com efeitos visuais
        checkpoint_colors = [
            (0, 255, 0),   # Verde - próximo checkpoint
            (0, 200, 200),  # Ciano
            (200, 0, 200),  # Magenta
            (255, 255, 0),  # Amarelo
        ]
        
        for i, (row, col) in enumerate(self.checkpoints):
            if i < len(checkpoint_colors):
                color = checkpoint_colors[i]
                if i == self.current_checkpoint:
                    # Efeito pulsante para o checkpoint atual
                    size = CELL_SIZE + int(2 * abs(pygame.time.get_ticks() % 1000 - 500) / 500)
                    pygame.draw.circle(screen, color, 
                                    (col * CELL_SIZE + CELL_SIZE//2, 
                                    row * CELL_SIZE + CELL_SIZE//2), 
                                    size//2 - 2)
                else:
                    # Checkpoints completos
                    pygame.draw.rect(screen, color, 
                                (col * CELL_SIZE + 2, row * CELL_SIZE + 2, 
                                    CELL_SIZE - 4, CELL_SIZE - 4), 
                                    border_radius=5)
                
                # Número do checkpoint
                font = pygame.font.SysFont('Arial', 16, bold=True)
                text = font.render(str(i+1), True, BLACK)
                text_rect = text.get_rect(center=(col * CELL_SIZE + CELL_SIZE//2, 
                                                row * CELL_SIZE + CELL_SIZE//2))
                screen.blit(text, text_rect)
        
        # Adicionar bordas arredondadas nas paredes
        for row in range(ROWS):
            for col in range(COLS):
                if self.grid[row][col] == 0:  # Parede
                    # Verifica vizinhos para suavizar cantos
                    neighbors = 0
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        r, c = row + dr, col + dc
                        if 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == 0:
                            neighbors += 1
                    
                    # Desenha cantos arredondados para paredes isoladas
                    if neighbors < 4:
                        pygame.draw.rect(screen, (70, 70, 70), 
                                    (col * CELL_SIZE, row * CELL_SIZE, 
                                        CELL_SIZE, CELL_SIZE), 
                                        border_radius=3)