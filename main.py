import sys
import pygame
import signal
from game.game import Game

def handle_exit(signum, frame):
    """Handler para encerramento limpo"""
    print("\nSalvando progresso antes de sair...")
    game.save_game()
    game.agent.save_model()
    game.close()
    pygame.quit()
    sys.exit(0)

def main():
    global game  # Para acessar no handler
    
    print("Pressione Ctrl+C para sair a qualquer momento.")
    print("Pressione 'S' para salvar o jogo ou 'L' para carregar um jogo salvo.")
    
    # Configurar handler para Ctrl+C
    signal.signal(signal.SIGINT, handle_exit)
    
    pygame.init()
    game = Game()
    
    try:
        # Tenta carregar um jogo existente
        if not game.load_game():
            print("Iniciando novo jogo...")
            game.train(episodes=1000)
    except Exception as e:
        print(f"Erro ao carregar jogo: {e}, iniciando novo jogo...")
        game.train(episodes=1000)
    
    try:
        running = True
        while running:
            # Processar eventos primeiro
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # Tecla S para salvar
                        game.save_game()
                        game.agent.save_model()
                        print("Jogo salvo manualmente!")
                    elif event.key == pygame.K_l:  # Tecla L para carregar
                        if game.load_game():
                            game.agent.load_model()
                        
            if not running:
                break
                
            # Executar episódio
            game.run_episode(render=True)
            game.clock.tick(120)
            
    except KeyboardInterrupt:
        print("\nInterrupção recebida, salvando...")
    except Exception as e:
        print(f"Erro durante execução: {e}")
    finally:
        # Garante que o salvamento ocorra mesmo com erros
        print("Salvando progresso antes de sair...")
        game.save_game()
        game.agent.save_model()
        game.close()
        pygame.quit()

if __name__ == "__main__":
    main()