import pygame
import numpy as np
import random
from scipy.integrate import odeint
import pygame
import math
import numpy as np 
import sympy as sp
from sympy.utilities import lambdify
from sympy.solvers import solve
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Pendulum:
    def __init__(self, screen_width, screen_height,length, time, initial_angle, initial_velocity):

        #screen Var
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        #Pend Var
        self.length = length
        self.time = time
        self.initial_conditions = [initial_angle, initial_velocity]
        
        self.starting_angle = initial_angle
        self.starting_velocity = initial_velocity
        self.angle_values = self.solve_pendulum(0.10)

        self.angle_index = 0


    def solve_pendulum(self, damping_coefficient):
        mass, gravity, length, time = sp.symbols(("mass", "gravity", "length", "time"))
        angle = sp.Function("angle")(time)
        angle_derivative = angle.diff(time)
        angle_second_derivative = angle_derivative.diff(time)

        x_coordinate = length * sp.sin(angle) 
        y_coordinate = -length * sp.cos(angle)
        kinetic_energy = sp.Rational(1, 2) * mass * (x_coordinate.diff(time)**2 + y_coordinate.diff(time)**2)
        potential_energy = mass * gravity * y_coordinate
        lagrangian = kinetic_energy - potential_energy
        
        lagrangian_left_hand_side = lagrangian.diff(angle)
        lagrangian_right_hand_side = sp.diff(lagrangian.diff(angle_derivative), time)

        equation_of_motion = lagrangian_right_hand_side - lagrangian_left_hand_side
        equation_of_motion = sp.solve(equation_of_motion, angle_second_derivative)[0]

        # Define the function for angle derivative and acceleration
        angle_derivative_function = sp.lambdify(angle_derivative, angle_derivative)
        acceleration_function = sp.lambdify((gravity, length, angle), equation_of_motion)

        # Define the function for the derivative of state
        def derivative_of_state(state, time, gravity, length):
            current_angle, current_velocity = state
            damping_force = -damping_coefficient * current_velocity  # Damping force
            return [angle_derivative_function(current_velocity),
                    acceleration_function(gravity, length, current_angle) + damping_force]

        # Set up initial conditions and solve the differential equation
        gravity = 980 
        length = self.length
        time = self.time
        initial_conditions = self.initial_conditions
        solution = odeint(derivative_of_state, initial_conditions, time, args=(gravity, length))

        angle_values = solution.T[0]

        return angle_values

    def draw(self, screen):
        # Draw the pendulum on the screen
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        angle = self.angle_values[self.angle_index]
        x = center_x + self.length * np.sin(angle)
        y = center_y + self.length * np.cos(angle)

        pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), (x, y), 5)
        pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 10)

    def update(self):
        self.angle_index = (self.angle_index + 1) % len(self.angle_values)



    
class PendulumGame:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pendulum Simulation")

        self.clock = pygame.time.Clock()
        simulation_time = np.linspace(0, 30, 200)
        self.pendulum = Pendulum(self.screen_width, self.screen_height, 200, simulation_time, np.pi/4, 0)

    def run_game(self):
        running = True
        time_at_zero = 0
        close_to_zero_threshold = np.deg2rad(5)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((0, 0, 0))

            self.pendulum.draw(self.screen)
            self.pendulum.update()

            current_angle = self.pendulum.angle_values[self.pendulum.angle_index]
            if abs(current_angle) < close_to_zero_threshold:
                time_at_zero += 1 / 30  # Increment time spent close to 0 degrees (assuming 30 FPS)

            # If the pendulum is close to 0 degrees and has stayed there for at least 2 seconds, stop the simulation
            if time_at_zero >= 2:
                print("game_ended")
                running = False

            pygame.display.flip()
            self.clock.tick(30)
            



        pygame.quit()

if __name__ == "__main__":
    game = PendulumGame()
    game.run_game()

