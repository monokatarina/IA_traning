# Configurações do jogo
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 20
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Configurações do agente
ALPHA = 0.1  # Taxa de aprendizado
GAMMA = 0.9  # Fator de desconto
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995