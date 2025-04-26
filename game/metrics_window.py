import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe
import numpy as np
import signal

def metrics_process(conn):
    # Configuração para evitar que o matplotlib trave
    plt.switch_backend('QtAgg')  # Ou 'TkAgg' se não funcionar
    
    # Configurar handler para ignorar Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        # Estilo do gráfico
        plt.style.use('ggplot')
        
        # Criar figura e eixos
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plt.subplots_adjust(hspace=0.5)
        fig.canvas.manager.set_window_title('Métricas de Treinamento')
        
        rewards_history = []
        avg_history = []
        epsilon_history = []
        
        while True:
            if conn.poll(0.1):
                data = conn.recv()
                if data == 'close':
                    plt.close('all')
                    break
                
                episode, reward, avg, epsilon = data
                rewards_history.append(reward)
                avg_history.append(avg)
                epsilon_history.append(epsilon)
                
                # Atualizar gráfico de recompensas
                ax1.clear()
                ax1.plot(rewards_history, 'b-', label='Recompensa')
                ax1.plot(avg_history, 'r-', label='Média (100 eps)')
                ax1.set_title(f'Episódio: {episode} | Recompensa: {reward:.1f} | Média: {avg:.1f}')
                ax1.set_xlabel('Episódio')
                ax1.set_ylabel('Recompensa')
                ax1.legend()
                ax1.grid(True)
                
                # Atualizar gráfico de epsilon
                ax2.clear()
                ax2.plot(epsilon_history, 'g-', label='Epsilon')
                ax2.set_title(f'Exploração: {epsilon:.3f}')
                ax2.set_xlabel('Episódio')
                ax2.set_ylabel('Taxa')
                ax2.legend()
                ax2.grid(True)
                
                plt.pause(0.01)
                
    except Exception as e:
        print(f"Erro no processo de métricas: {e}")
    finally:
        conn.close()

class MetricsWindow:
    def __init__(self):
        self.parent_conn, child_conn = Pipe()
        self.process = Process(
            target=metrics_process,
            args=(child_conn,),
            daemon=True
        )
        self.process.start()
    
    def update(self, episode, reward, avg, epsilon):
        try:
            self.parent_conn.send((episode, reward, avg, epsilon))
        except:
            pass
    
    def close(self):
        try:
            self.parent_conn.send('close')
            self.process.join(timeout=0.5)
        except:
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()