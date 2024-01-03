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
import math
from python.zmqRemoteApi import RemoteAPIClient
import os
import pickle
def calcular_pos(robot):
    _,_,dimensions1=robot.get_dimensiones_link1()
    _,_,dimensions2=robot.get_dimensiones_link2()
    theta1=robot.get_pos_joint1()
    theta2=robot.get_pos_joint2()
    theta1_r=math.radians(theta1)
    theta2_r=math.radians(theta2)
    L1=dimensions1[0]
    L2=dimensions2[0]
    #print("Informacion: ", L1," ",L2," ",theta1," ",theta2)
    '''    if theta2<0:
        theta2+=360
    if theta1<0:
        theta1+=360'''
    x1,y1=L1*math.sin(theta1_r),-L1*math.cos(theta1_r)
    x2,y2=x1+L2*math.sin(theta2_r+theta1_r),y1-L2*math.cos(theta2_r+theta1_r)
    return y2
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
    joint1_handle=-1
    link1_handle=-1
    joint2_handle=-1
    link2_handle=-1
    def __init__(self, sim, robot_id, use_camera=False, use_lidar=False):
        self.sim = sim
        print('*** getting handles', robot_id)
        #self.joint1 = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1')
        #self.joint2 = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1/Link1/Joint2')
        
        self.dummy_handle = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1/Link1/Joint2/Link2/Dummy')
        self.joint1_handle = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1')
        self.joint2_handle = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1/Link1/Joint2')
        self.link1_handle = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1/Link1')
        self.link2_handle = self.sim.getObject(f'/{robot_id}/Acrobot/Joint1/Link1/Joint2/Link2')


    def set_speed(self, speed):
        self.sim.setJointTargetVelocity(self.joint2_handle, speed)
    def set_pos_joint(self,pos):
        pos=pos*np.pi/180
        self.sim.setJointTargetPosition(self, pos,  [])
    def set_joint_torque(self,torque):
        self.sim.setJointTargetForce( self.joint2_handle, torque, True)

    def get_joint_torque(self):
        return self.sim.getJointForce( self.joint2_handle)
    
    def get_handle(self):
        return self.joint2_handle

    def get_pos_joint2(self):
        self.joint_pos=self.sim.getJointPosition(self.joint2_handle)*180/np.pi
        return self.joint_pos

    def get_pos_joint1(self):
         return self.sim.getJointPosition(self.joint1_handle) * 180 / np.pi

    def get_pos_dummy(self):
        return self.sim.getObjectPosition(self.dummy_handle,self.sim.handle_world)
         
    def get_velocity_joint(self):
        return  self.sim.getJointTargetVelocity(self.joint2_handle)
    def get_torque_joint(self):
        return self.sim.getJointForce(self.joint2_handle)
    
    def get_dimensiones_link1(self):
        return self.sim.getShapeGeomInfo(self.link1_handle)
    
    def get_dimensiones_link2(self):
        return self.sim.getShapeGeomInfo(self.link2_handle)
    
    def start_motor(self):
        self.sim.setObjectInt32Param(self.joint2_handle,sim.sim_jointintparam_motor_enabled,1)
        #self.sim.setObjectInt32Param(self.joint2_handle,sim.sim_jointintparam_ctrl_enabled,1)

#+2.2275 altura máxima del dummy
#con torque 0.2 no llega a subir
#fitness será el que más tiempo esté el dummy en la posición más alta
#Exito: dejar x segundos el dummy en la posición más alta
#ctrl+w cierra escena en copelia
#main()

def calcular_recompensa(distance):
   
    return (2-distance) 


import neat

# Función para conectar con CoppeliaSim y evaluar la red neuronal
def eval_genomes(genomes, config):
    # BORRAR client = b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient', 'b0RemoteApi') 
    coppelia = Coppelia() # Conectar a CoppeliaSim
    robot = Acrobot(coppelia.sim, 'Cuboid')
    for genome_id, genome in genomes:
        # Crear una red neuronal para el genoma
        genome.fitness=0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        total_reward = 0.0
        reward= []
        for _ in range(3):  # Ejecutar 5 episodios para evaluar
            # Iniciar la simulación en CoppeliaSim
            coppelia = Coppelia() # Conectar a CoppeliaSim
            robot = Acrobot(coppelia.sim, 'Cuboid')
            coppelia.start_simulation()
            robot.start_motor()
            robot.set_joint_torque(0)  
            
            steps = 0
            time_near_target = 0  # Contador de tiempo cerca del objetivo
            estuvo_cerca=False
            tiempo_sim=5
            while (t := coppelia.sim.getSimulationTime()) < tiempo_sim:
                # Calcular la distancia entre la posición actual y el objetivo
                distance = 1-calcular_pos(robot)  # Implementa esta función
                
                # Calcular la recompensa en base a la cercanía al objetivo y el tiempo
                # Aquí, puedes diseñar una función que premie el tiempo cerca del objetivo
                '''if  (1-calcular_pos(robot)) < umbral_cercania and not estuvo_cerca:
                    cercania_start=coppelia.sim.getSimulationTime()
                    estuvo_cerca=True
                    # Podría ser algo como (distancia * tiempo_cerca) / steps o alguna variante
                    
                else:
                    reward=0
                if estuvo_cerca and (1-calcular_pos(robot)) > umbral_cercania:
                    cercania_end=coppelia.sim.getSimulationTime()
                    time_near_target+=cercania_end-cercania_start
                    estuvo_cerca=False
                    reward += calcular_recompensa(distance, coppelia.sim.getSimulationTime() )  # Implementa esta función
                '''
                 # Implementa esta función
                #reward.append( calcular_recompensa(distance ) )
                if distance < 0.1:  # Si está muy cerca del objetivo
                    recompensa = 100  # Recompensa alta por alcanzar el objetivo
                else:
                    recompensa = max(0, 2 - distance * 10)  # Penalización gradual mientras se aleja del objetivo
                reward.append( recompensa )
                # Sumar la recompensa total y actualizar el tiempo cerca del objetivo
                
                
                # Ejecutar la red neuronal para obtener la acción
                # Posicion Joint 1
                # Posicion Joint 2
                # Posicion Robot
                # Distancia
                # Fuerza
                # Tiempo
                tiempo_Act=coppelia.sim.getSimulationTime()
                j1=robot.get_pos_joint1()
                j2=robot.get_pos_joint2()
                pos=calcular_pos(robot)
                torque=robot.get_torque_joint()
                if tiempo_Act is None:
                    print("ES NONE: tiempo_Act",tiempo_Act )
                if j1 is None:
                    print("ES NONE: j1",j1 )
                if j2 is None:
                    print("ES NONE: j2",j2 )
                if pos is None:
                    print("ES NONE: pos",pos )
                if torque is None:
                    torque=robot.get_torque_joint()
                    if torque is None:
                        torque=torque_guardado
                        print("ES NONEEEE: torque",torque )
                
                action = net.activate((j1,j2,pos,distance,torque, tiempo_Act))  # Los inputs son la distancia y el tiempo cerca del objetivo
                #print("La accion es:", action, " con recompensa: ",calcular_recompensa(distance ))
                #print(robot.get_pos_joint1()," ",robot.get_pos_joint2()," ",calcular_pos(robot)," ",distance," ",robot.get_torque_joint()," ", coppelia.sim.getSimulationTime())
                robot.set_joint_torque(0.2*action[0])
                torque_guardado=0.2*action[0]
                # Enviar la acción a CoppeliaSim y avanzar la simulación un paso
                # Implementa la lógica para enviar la acción y avanzar la simulación
            
            total_reward = sum(reward) / len(reward)
            print("Longitud de recompensas: ",len(reward))
            # Detener la simulación después de cada episodio
            coppelia.stop_simulation()
            print("Fin simulacion episodica")
        
        # Calcular la aptitud del genoma basada en la recompensa total
        genome.fitness = total_reward / tiempo_sim  # Promedio de las recompensas de los 5 episodios
    
    # Cerrar la conexión con CoppeliaSim después de evaluar todos los genomas
    print("Fin simulacion")
    coppelia.stop_simulation()
    return genome.fitness
def test_neat(genome,config):
    net=neat.nn.FeedForwardNetwork.create(genome,config)
    coppelia = Coppelia() # Conectar a CoppeliaSim
    robot = Acrobot(coppelia.sim, 'Cuboid')
    while (t := coppelia.sim.getSimulationTime()) < 20:
        distance=1-calcular_pos(robot)
        action = net.activate((robot.get_pos_joint1(),robot.get_pos_joint2(),calcular_pos(robot),distance,robot.get_torque_joint(), coppelia.sim.getSimulationTime()))  # Los inputs son la distancia y el tiempo cerca del objetivo
        robot.set_joint_torque(0.2*action[0])        

def run_neat(config):
    # Crear la población inicial
    p = neat.Population(config)
    # Añadir un report para mostrar el progreso
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #Para guardar progreso del entrenamiento, cada x iteraciones
    p.add_reporter(neat.Checkpointer(1))
    #Para cargar desde entrenamiento:
    #p= neat.Checkpointer.restore_checkpoint('neat_checkpoint-27')
    # Ejecutar NEAT (reemplaza 50 con el número de generaciones que desees)
    winner = p.run(eval_genomes, 50)
    #Best genomes:
    with open("best.pickle","wb") as f:
        pickle.dump(winner,f)

def test_ai(config):
    with open("best.picke","rb") as f:
        winner= pickle.load(f)
    test_neat(winner,config)
#C:/Users/Jorge/Desktop/Uni/Master/Tercer Semestre/Robótica/proyecto_acrobot/
# Configuración para NEAT (igual que en el ejemplo anterior)
if __name__ == "__main__":
    local_dir= os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)  # Reemplaza 'config-feedforward' con tu archivo de configuración
    run_neat(config)
    test_ai(config)

