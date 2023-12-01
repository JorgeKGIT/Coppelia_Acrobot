# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:06:05 2023

@author: Jorge
"""

#Se instalan librerias necesarias
#!pip install neat-python
#!pip install coppeliasim-zmqremoteapi-client
#!pip install cbor

import numpy as np
import time
import sim
import pandas as pd

from python.zmqRemoteApi import RemoteAPIClient
class Coppelia():

    def __init__(self):
        print('*** connecting to coppeliasim')
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')

    def start_simulation(self):
        # print('*** saving environment')
        self.default_idle_fps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.startSimulation()

    def stop_simulation(self):
        # print('*** stopping simulation')
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        # print('*** restoring environment')
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.default_idle_fps)
        print('*** done')

    def is_running(self):
        return self.sim.getSimulationState() != self.sim.simulation_stopped


class Acrobot():
    
    num_sonar = 16
    ang_max = np.pi
    joint_pos=0
    joint2_handle=-1
    dummy_handle=-1

    def __init__(self, sim, robot_id, use_camera=False, use_lidar=False):
        self.sim = sim
        print('*** getting handles', robot_id)
        #self.joint1 = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1')
        #self.joint2 = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1/Link1/Joint2')
        self.joint2_handle = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1/Link1/Joint2')
        self.dummy_handle = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1/Link1/Joint2/Link2/Dummy')


    def set_speed(self, speed):
        self.sim.setJointTargetVelocity(self.joint2_handle, speed)
    def set_pos_joint(self,pos):
        pos=pos*np.pi/180
        self.sim.setJointTargetPosition(self, pos,  [])
    def set_joint_torque(self,torque):
        self.sim.setJointTargetForce( self.joint2_handle, torque, True)
    def get_handle(self):
        return self.joint2_handle

    def get_pos_joint(self):
        self.joint_pos=self.sim.getJointTargetPosition(self.joint2_handle)*180/np.pi
        return self.joint_pos
    def get_pos_dummy(self):
        return self.sim.getObjectPosition(self.dummy_handle,self.sim.handle_world)
         
    def get_velocity_joint(self):
        return  self.sim.getJointTargetVelocity(self.joint2_handle)
    def get_torque_joint(self):
        return self.sim.getJointForce(self.joint2_handle)
    def start_motor(self):
        self.sim.setObjectInt32Param(self.joint2_handle,sim.sim_jointintparam_motor_enabled,1)
        #self.sim.setObjectInt32Param(self.joint2_handle,sim.sim_jointintparam_ctrl_enabled,1)
        
        
'''def main(args=None):
    coppelia = Coppelia()
    robot = Acrobot(coppelia.sim, 'Cuboid')
    #robot.set_speed(+1.2)
    coppelia.start_simulation()
    robot.start_motor()
    #robot.set_speed(+3.14)
    robot.set_joint_torque(-0.3)
    t1=0
    print(f'Position Dummy:',robot.get_pos_dummy()[2] )
    while (t := coppelia.sim.getSimulationTime()) < 5:
        print(f'Simulation time: {t:.3f} [s]')
        #print(f'Velocity joint: ', robot.get_velocity_joint())
        print(f'Torque joint: ',robot.get_torque_joint())
        #if t1%2==0:
        #   robot.set_speed(-3.14)
        #else:
        #    robot.set_speed(3.14)
        print(f'Position Dummy:',robot.get_pos_dummy()[2] )
        #if t1%2==0:
        #    robot.set_joint_torque(-0.25)
        #else:
         #   robot.set_joint_torque(0.25)
        print(f' grados: {robot.get_pos_joint():.2f}')
        t1+=1

    coppelia.stop_simulation()'''

#+2.2275 altura máxima del dummy
#con torque 0.2 no llega a subir
#fitness será el que más tiempo esté el dummy en la posición más alta
#Exito: dejar x segundos el dummy en la posición más alta
#ctrl+w cierra escena en copelia
#main()

def calcular_recompensa(distance, time_near_target, steps):
    return (distance * time_near_target) / steps


import neat

# Función para conectar con CoppeliaSim y evaluar la red neuronal
def eval_genomes(genomes, config):
    # BORRAR client = b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient', 'b0RemoteApi') 
    coppelia = Coppelia() # Conectar a CoppeliaSim
    robot = Acrobot(coppelia.sim, 'Cuboid')
    target_pos = 2.2275  
    umbral_cercania=0.2
    for genome_id, genome in genomes:
        # Crear una red neuronal para el genoma
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        total_reward = 0.0
        max_steps = 1000  # Número máximo de pasos por episodio
        
        for _ in range(5):  # Ejecutar 5 episodios para evaluar
            # Iniciar la simulación en CoppeliaSim
            coppelia.start_simulation()
            robot.start_motor()
            
            
            steps = 0
            time_near_target = 0  # Contador de tiempo cerca del objetivo
            
            while True:
                # Obtener la posición del extremo del acrobot y del objetivo en CoppeliaSim
                acrobot_position = robot.get_pos_dummy()[2]
                
                # Calcular la distancia entre la posición actual y el objetivo
                distance = abs(target_pos-acrobot_position)  # Implementa esta función
                
                # Calcular la recompensa en base a la cercanía al objetivo y el tiempo
                # Aquí, puedes diseñar una función que premie el tiempo cerca del objetivo
                # Podría ser algo como (distancia * tiempo_cerca) / steps o alguna variante
                reward = calcular_recompensa(distance, time_near_target, coppelia.sim.getSimulationTime())  # Implementa esta función
                
                # Sumar la recompensa total y actualizar el tiempo cerca del objetivo
                total_reward += reward
                if distance < umbral_cercania:  # Define un umbral para "cerca del objetivo"
                    time_near_target += 1
                
                # Ejecutar la red neuronal para obtener la acción
                action = net.activate([distance, time_near_target])  # Los inputs son la distancia y el tiempo cerca del objetivo
                
                # Enviar la acción a CoppeliaSim y avanzar la simulación un paso
                # Implementa la lógica para enviar la acción y avanzar la simulación
                
                steps += 1
                if steps >= max_steps:
                    break
            
            # Detener la simulación después de cada episodio
            coppelia.stop_simulation()
        
        # Calcular la aptitud del genoma basada en la recompensa total
        genome.fitness = total_reward / 5  # Promedio de las recompensas de los 5 episodios
    
    # Cerrar la conexión con CoppeliaSim después de evaluar todos los genomas
    coppelia.stop_simulation()
#C:/Users/Jorge/Desktop/Uni/Master/Tercer Semestre/Robótica/proyecto_acrobot/
# Configuración para NEAT (igual que en el ejemplo anterior)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'C:/Users/Jorge/Desktop/Uni/Master/Tercer Semestre/Robótica/proyecto_acrobot/config.txt')  # Reemplaza 'config-feedforward' con tu archivo de configuración

# Crear la población inicial
p = neat.Population(config)

# Añadir un reportero para mostrar el progreso
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Ejecutar NEAT (reemplaza 100 con el número de generaciones que desees)
winner = p.run(eval_genomes, 100)