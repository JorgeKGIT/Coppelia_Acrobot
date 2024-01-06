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

    def get_pos_joint2_rad(self):
        return self.sim.getJointPosition(self.joint2_handle)

    def get_pos_joint1_rad(self):
         return self.sim.getJointPosition(self.joint1_handle) 
    
    def get_pos_dummy(self):
        return self.sim.getObjectPosition(self.dummy_handle,self.sim.handle_world)
         
    def get_velocity_joint2(self):
        return  self.sim.getJointVelocity(self.joint2_handle)
    def get_velocity_joint1(self):
        return  self.sim.getJointVelocity(self.joint1_handle)
    
    def get_velocity_dummy(self):
        return  self.sim.getObjectVelocity(self.dummy_handle)
    
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
   
    if distance < 0.1:  # Si está muy cerca del objetivo
        recompensa = 100  # Recompensa alta por alcanzar el objetivo
    else:
        recompensa = max(0, (2 - distance) * 10)
    
    return recompensa


import neat

# Función para conectar con CoppeliaSim y evaluar la red neuronal
def eval_genomes(genomes, config):
    # BORRAR client = b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient', 'b0RemoteApi') 
    coppelia = Coppelia() # Conectar a CoppeliaSim
    robot = Acrobot(coppelia.sim, 'Cuboid')
    contador_genomas = 0
    
    for genome_id, genome in genomes:
        print("Numero de genoma: ",contador_genomas , " genome_id",genome_id)
        # Crear una red neuronal para el genoma
        
        activation_default = config.genome_config.activation_default
        #print("Función de activación predeterminada:", activation_default)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        total_reward = 0.0
        reward= []
        #for _ in range(3):  # Ejecutar 5 episodios para evaluar
        # Iniciar la simulación en CoppeliaSim
        coppelia = Coppelia() # Conectar a CoppeliaSim
        robot = Acrobot(coppelia.sim, 'Cuboid')
        coppelia.start_simulation()
        robot.start_motor()
        robot.set_joint_torque(0)  
        
        steps = 0
        time_near_target = 0  # Contador de tiempo cerca del objetivo
        estuvo_cerca=False
        tiempo_sim=20
        vuelta=False
        while (t := coppelia.sim.getSimulationTime()) < tiempo_sim:
            # Calcular la distancia entre la posición actual y el objetivo
            distance = 1-calcular_pos(robot)  
              # Penalización gradual mientras se aleja del objetivo
            reward.append( calcular_recompensa(distance) )
            if( robot.get_pos_joint2_rad()>np.pi*2 or robot.get_pos_joint2_rad()<-np.pi*2):
                #print("Ha dado vuelta entera")
                vuelta=True
                break
            # Ejecutar la red neuronal para obtener la acción
            # Posicion
            # Posicion Joint 1
            # Posicion Joint 2
            # Velocidad Joint 1 
            # Velocidad Joint 2
            # Tiempo
            pos=calcular_pos(robot)
            j1=robot.get_pos_joint1_rad()
            j2=robot.get_pos_joint2_rad()
            j1_dot=robot.get_velocity_joint1()
            j2_dot=robot.get_velocity_joint2()  
            pos_dot=robot.get_velocity_dummy()[1][2]
            action = net.activate((pos,j1,j2,j1_dot,j2_dot, pos_dot))  # Los inputs son la distancia y el tiempo cerca del objetivo
            #print("action= ",action)
            #print(pos,j1,j2,j1_dot,j2_dot,robot.get_velocity_dummy()[1][2])
            #print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(pos, j1, j2, j1_dot, j2_dot, pos_dot))
            robot.set_joint_torque(0.2*action[0])
            #torque_guardado=0.2*action[0]
            # Enviar la acción a CoppeliaSim y avanzar la simulación un paso
            # Implementa la lógica para enviar la acción y avanzar la simulación
        if vuelta :
            total_reward=-100
        elif sum(reward)<0.05:
            total_reward=-1000
        else: 
            total_reward = sum(reward) / len(reward)

        print("Longitud de recompensas: ",len(reward))
        # Detener la simulación después de cada episodio
        coppelia.stop_simulation()
        print("Fin simulacion episodica")
        contador_genomas+=1
        # Calcular la aptitud del genoma basada en la recompensa total
        genome.fitness = total_reward   # Promedio de las recompensas de los 5 episodios
        print("Fitness: ",genome.fitness," total_reward", total_reward , " sum(reward)", sum(reward))
    # Cerrar la conexión con CoppeliaSim después de evaluar todos los genomas
    print("Fin simulacion")
    coppelia.stop_simulation()


def test_neat(genome,config):
    net=neat.nn.FeedForwardNetwork.create(genome,config)
    coppelia = Coppelia() # Conectar a CoppeliaSim
    robot = Acrobot(coppelia.sim, 'Cuboid')
    coppelia.start_simulation()
    robot.start_motor()
    robot.set_joint_torque(0)  
    while (t := coppelia.sim.getSimulationTime()) < 20:
        distance=1-calcular_pos(robot)
        pos=calcular_pos(robot)
        j1=robot.get_pos_joint1_rad()
        j2=robot.get_pos_joint2_rad()
        j1_dot=robot.get_velocity_joint1()
        j2_dot=robot.get_velocity_joint2()
        pos_dot=robot.get_velocity_dummy()[1][2]
        action = net.activate((pos,j1,j2,j1_dot,j2_dot, pos_dot))  # Los inputs son la distancia y el tiempo cerca del objetivo    
        robot.set_joint_torque(0.2*action[0])        
    coppelia.stop_simulation()

def run_neat(config):
    # Crear la población inicial
    p = neat.Population(config)
    #p= neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    #neat-checkpoint-11
    # Añadir un report para mostrar el progreso
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #Para guardar progreso del entrenamiento, cada x iteraciones
    p.add_reporter(neat.Checkpointer(1))

    activation_functions = dir(neat.activations)
    print("Funciones de activación disponibles:")
    for activation_function in activation_functions:
        print(activation_function)
    #Para cargar desde entrenamiento:
    #p= neat.Checkpointer.restore_checkpoint('neat_checkpoint-27')
    # Ejecutar NEAT (reemplaza 50 con el número de generaciones que desees)
    winner = p.run(eval_genomes, 300)
    print("Ganador: ",winner)
    #Best genomes:
    with open("best.pickle","wb") as f:
        pickle.dump(winner,f)
    test_ai(config)

def test_ai(config):
    with open("best.pickle","rb") as f:
        winner= pickle.load(f)
    test_neat(winner,config)
#C:/Users/Jorge/Desktop/Uni/Master/Tercer Semestre/Robótica/proyecto_acrobot/
# Configuración para NEAT (igual que en el ejemplo anterior)
if __name__ == "__main__":
    local_dir= os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)  
    #run_neat(config)
    test_ai(config)

