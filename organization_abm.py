# organization_abm.py

import random
import networkx as nx
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==========================
# 1. Define the Generative AI Agent
# ==========================

class GenerativeAI(Agent):
    """
    Generative AI agent that evolves over time by increasing its knowledge contribution.
    """
    def __init__(self, unique_id, model, initial_contribution=2.0, evolution_rate=0.05):
        super().__init__(unique_id, model)
        self.knowledge_contribution = initial_contribution
        self.evolution_rate = evolution_rate  # Rate at which AI evolves per step

    def step(self):
        # Evolve AI's knowledge contribution
        self.knowledge_contribution += self.evolution_rate

    def provide_information(self):
        return self.knowledge_contribution

# ==========================
# 2. Define the Employee Agent
# ==========================

class EmployeeAgent(Agent):
    """
    Employee agent with diverse roles, knowledge, AI attitudes, and behavioral dynamics.
    """
    def __init__(self, unique_id, model, role, expertise, ai_attitude):
        super().__init__(unique_id, model)
        self.role = role  # e.g., Manager, Staff, etc.
        self.expertise = expertise  # Expertise level (1-10)
        self.ai_attitude = ai_attitude  # 'positive', 'neutral', 'negative'
        self.knowledge = expertise  # Initialize knowledge base
        self.decay_rate = 0.01  # Knowledge decay rate per step
        self.ai_usefulness_threshold = 5.0  # Threshold to change attitude

    def step(self):
        # Knowledge decay
        self.knowledge = max(self.knowledge - self.decay_rate, 0)

        # Interact with social network
        social_neighbors = self.model.social_network.neighbors(self.unique_id)
        social_neighbors = list(social_neighbors)
        if social_neighbors:
            partner_id = random.choice(social_neighbors)
            partner = self.model.schedule.agents[partner_id]
            self.exchange_information(partner)

        # Interact with organizational network if the agent exists in the network
        if self.unique_id in self.model.org_network:
            org_neighbors = self.model.org_network.neighbors(self.unique_id)
            org_neighbors = list(org_neighbors)
            if org_neighbors:
                org_partner_id = random.choice(org_neighbors)
                org_partner = self.model.schedule.agents[org_partner_id]
                self.exchange_information(org_partner)

        # Interact with Generative AI
        self.interact_with_ai()

        # Possibly evolve social network
        self.evolve_social_network()

        # Possibly change AI attitude based on knowledge
        self.update_ai_attitude()


    def exchange_information(self, partner):
        """
        Exchange information with another agent by averaging knowledge.
        """
        if isinstance(partner, EmployeeAgent):
            avg_knowledge = (self.knowledge + partner.knowledge) / 2
            self.knowledge = avg_knowledge
            partner.knowledge = avg_knowledge

    def interact_with_ai(self):
        """
        Interact with the AI agent based on AI attitude.
        """
        ai_agent = self.model.ai_agent
        if self.ai_attitude == 'positive':
            # AI augments knowledge
            self.knowledge += ai_agent.provide_information()
            self.model.ai_usage_count +=1
        elif self.ai_attitude == 'neutral':
            # AI provides minimal information
            self.knowledge += ai_agent.provide_information() * 0.5
            self.model.ai_usage_count +=0.5
        elif self.ai_attitude == 'negative':
            # Reluctant to use AI; minimal or no information
            self.knowledge += ai_agent.provide_information() * 0.2
            self.model.ai_usage_count +=0.2

    def evolve_social_network(self):
        """
        Dynamically evolve the social network by adding/removing connections.
        """
        add_prob = 0.01  # Probability to form a new connection
        remove_prob = 0.005  # Probability to remove an existing connection

        # Attempt to add a new social connection
        if random.random() < add_prob:
            possible_agents = set([agent.unique_id for agent in self.model.schedule.agents if isinstance(agent, EmployeeAgent)]) - set([self.unique_id]) - set(self.model.social_network.neighbors(self.unique_id))
            if possible_agents:
                new_neighbor = random.choice(list(possible_agents))
                self.model.social_network.add_edge(self.unique_id, new_neighbor)

        # Attempt to remove an existing social connection
        if random.random() < remove_prob and self.model.social_network.degree(self.unique_id) > 1:
            current_neighbors = list(self.model.social_network.neighbors(self.unique_id))
            if current_neighbors:
                remove_neighbor = random.choice(current_neighbors)
                self.model.social_network.remove_edge(self.unique_id, remove_neighbor)


    def update_ai_attitude(self):
        """
        Update AI attitude based on knowledge level.
        If knowledge exceeds a threshold, become more positive towards AI.
        If below, become more negative.
        """
        if self.knowledge > self.ai_usefulness_threshold and self.ai_attitude != 'positive':
            self.ai_attitude = 'positive'
        elif self.knowledge < self.ai_usefulness_threshold and self.ai_attitude != 'negative':
            self.ai_attitude = 'negative'
        # Else, remain neutral or current attitude

# ==========================
# 3. Define Network Creation Functions
# ==========================

def create_hierarchical_network(num_levels, span_of_control, num_employees):
    """
    Create a hierarchical organizational network, ensuring that the number of nodes
    matches or exceeds the number of employees.
    """
    G = nx.DiGraph()
    current_id = 0
    G.add_node(current_id)
    previous_level = [current_id]
    current_id += 1

    for level in range(1, num_levels):
        current_level = []
        for manager in previous_level:
            for _ in range(span_of_control):
                if current_id < num_employees:
                    G.add_node(current_id)
                    G.add_edge(manager, current_id)
                    current_level.append(current_id)
                    current_id += 1
        previous_level = current_level

    # If fewer nodes were created than num_employees, add extra nodes
    while len(G.nodes) < num_employees:
        G.add_node(current_id)
        current_id += 1

    return G


def create_onion_network(num_levels, span_of_control, num_employees):
    """
    Create an onion-like organizational network with inter-layer connections,
    ensuring that the number of nodes matches or exceeds the number of employees.
    """
    G = nx.DiGraph()
    current_id = 0
    G.add_node(current_id)
    previous_level = [current_id]
    current_id += 1

    for level in range(1, num_levels):
        current_level = []
        for manager in previous_level:
            for _ in range(span_of_control):
                if current_id < num_employees:
                    G.add_node(current_id)
                    G.add_edge(manager, current_id)
                    current_level.append(current_id)
                    current_id += 1
        # Add inter-layer connections (e.g., lateral connections)
        for node in current_level:
            if random.random() < 0.3 and previous_level:
                lateral = random.choice(previous_level)
                if lateral != node:
                    G.add_edge(node, lateral)
        previous_level = current_level

    # If fewer nodes were created than num_employees, add extra nodes
    while len(G.nodes) < num_employees:
        G.add_node(current_id)
        current_id += 1

    return G


def create_small_world_network(num_agents, k, p):
    """
    Create a small-world social network with enough nodes for all agents.
    """
    return nx.watts_strogatz_graph(num_agents, k, p)


def create_scale_free_network(num_agents, m):
    """
    Create a scale-free social network with enough nodes for all agents.
    """
    return nx.barabasi_albert_graph(num_agents, m)

# ==========================
# 4. Define the ABM Model
# ==========================

class OrganizationModel(Model):
    """
    Organization Model that simulates employees interacting within organizational and social networks,
    interacting with a generative AI agent, with dynamic networks, knowledge decay, AI evolution,
    diverse roles, and behavioral dynamics.
    """
    def __init__(self, 
                 num_employees=100, 
                 num_levels=4, 
                 span_of_control=4,
                 org_network_type='hierarchical',  # 'hierarchical' or 'onion'
                 social_network_type='small_world',  # 'small_world' or 'scale_free'
                 social_k=4, 
                 social_p=0.1, 
                 social_m=2,
                 ai_contribution=2.0,
                 ai_evolution_rate=0.05,
                 knowledge_decay_rate=0.01,
                 max_steps=100,
                 output_dir='output'):
        """
        Parameters:
            num_employees: Number of human agents
            num_levels: Number of hierarchical levels (for organizational networks)
            span_of_control: Number of direct reports per manager
            org_network_type: 'hierarchical' or 'onion'
            social_network_type: 'small_world' or 'scale_free'
            social_k: Parameter for social network (e.g., neighbors in small-world)
            social_p: Rewiring probability for small-world
            social_m: Parameter for scale-free network (number of edges to attach)
            ai_contribution: Initial AI's knowledge contribution per interaction
            ai_evolution_rate: Rate at which AI's knowledge contribution evolves per step
            knowledge_decay_rate: Rate at which agents' knowledge decays per step
            max_steps: Number of steps to run the simulation
            output_dir: Directory to save output data
        """
        self.num_employees = num_employees
        self.num_agents = num_employees
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.max_steps = max_steps
        self.current_step = 0

        # Create Organizational Network
        if org_network_type == 'hierarchical':
            self.org_network = create_hierarchical_network(num_levels, span_of_control, num_employees)
        elif org_network_type == 'onion':
            self.org_network = create_onion_network(num_levels, span_of_control, num_employees)
        else:
            raise ValueError("Unsupported organizational network type. Choose 'hierarchical' or 'onion'.")

        # Create Social Interaction Network
        total_agents = num_employees +1  # +1 for AI agent
        if social_network_type == 'small_world':
            self.social_network = create_small_world_network(total_agents, social_k, social_p)
        elif social_network_type == 'scale_free':
            self.social_network = create_scale_free_network(total_agents, social_m)
        else:
            raise ValueError("Unsupported social network type. Choose 'small_world' or 'scale_free'.")

        # Initialize Generative AI Agent
        ai_unique_id = num_employees  # Assign last ID to AI
        self.ai_agent = GenerativeAI(ai_unique_id, self, initial_contribution=ai_contribution, evolution_rate=ai_evolution_rate)
        self.schedule.add(self.ai_agent)

        # Initialize NetworkGrid with Social Network
        self.grid = NetworkGrid(self.social_network)

        # Initialize Agents
        for i in range(num_employees):
            role = self.assign_role(i)
            expertise = random.uniform(1, 10)
            ai_attitude = self.assign_ai_attitude()
            agent = EmployeeAgent(i, self, role, expertise, ai_attitude)
            agent.decay_rate = knowledge_decay_rate  # Set knowledge decay rate
            self.schedule.add(agent)
            self.grid.place_agent(agent, i)  # Place based on social network node

        # Place AI agent in the last node
        self.grid.place_agent(self.ai_agent, ai_unique_id)

        # Initialize AI usage count for utilization metrics
        self.ai_usage_count = 0.0

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "Average Knowledge": self.compute_average_knowledge,
                "Knowledge Std Dev": self.compute_knowledge_std_dev,
                "AI Utilization": self.compute_ai_utilization,
                "Positive Attitudes": lambda m: m.count_ai_attitudes('positive'),
                "Neutral Attitudes": lambda m: m.count_ai_attitudes('neutral'),
                "Negative Attitudes": lambda m: m.count_ai_attitudes('negative'),
                "AI Contribution": self.get_ai_contribution
            }
        )

        # Prepare output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def assign_role(self, agent_id):
        """
        Assign role based on hierarchical level in organizational network.
        Managers have higher hierarchical levels, Staff have lower.
        """
        # Determine the hierarchical level of the agent
        level = self.get_agent_level(agent_id)
        if level == 0:
            return "CEO"
        elif level == 1:
            return "Manager_Level_1"
        elif level ==2:
            return "Manager_Level_2"
        else:
            return "Staff"

    def get_agent_level(self, agent_id):
        """
        Determine the hierarchical level of an agent in the organizational network.
        """
        if agent_id not in self.org_network:
            return -1  # Safeguard: Return -1 if the agent is not in the network
        level = 0
        current_id = agent_id
        while True:
            predecessors = list(self.org_network.predecessors(current_id))
            if not predecessors:
                break
            current_id = predecessors[0]
            level += 1
        return level


    def assign_ai_attitude(self):
        """
        Assign initial AI attitude based on probabilities.
        """
        r = random.random()
        if r <0.5:
            return 'positive'
        elif r <0.8:
            return 'neutral'
        else:
            return 'negative'

    def compute_average_knowledge(self):
        """
        Compute the average knowledge across all employee agents.
        """
        total_knowledge = sum(agent.knowledge for agent in self.schedule.agents if isinstance(agent, EmployeeAgent))
        return total_knowledge / self.num_employees

    def compute_knowledge_std_dev(self):
        """
        Compute the standard deviation of knowledge across all employee agents.
        """
        knowledge_values = [agent.knowledge for agent in self.schedule.agents if isinstance(agent, EmployeeAgent)]
        return pd.Series(knowledge_values).std()

    def compute_ai_utilization(self):
        """
        Compute AI utilization metrics.
        """
        # AI utilization was tracked as a float count in ai_usage_count
        return self.ai_usage_count

    def count_ai_attitudes(self, attitude_type):
        """
        Count the number of agents with the specified AI attitude.
    
        Parameters:
        attitude_type (str): The attitude to count ('positive', 'neutral', 'negative').
    
        Returns:
        int: The count of agents with the specified attitude.
        """
        return sum(1 for agent in self.schedule.agents if isinstance(agent, EmployeeAgent) and agent.ai_attitude == attitude_type)


    def get_ai_contribution(self):
        """
        Get the current AI knowledge contribution.
        """
        return self.ai_agent.knowledge_contribution

    #in the future, maybe this can be used to evolve the network
    '''
    def evolve_organizational_network(self):
        """
        Evolve the organizational network by reassigning a manager for a random agent.
        """
        # Probability to reassign a manager per step
        reassign_prob = 0.02
        if random.random() < reassign_prob:
            # Select a random employee (not CEO)
            employee_agents = [agent for agent in self.schedule.agents if isinstance(agent, EmployeeAgent) and self.get_agent_level(agent.unique_id) >0]
            if not employee_agents:
                return
            agent = random.choice(employee_agents)
            current_level = self.get_agent_level(agent.unique_id)
            # Possible new managers are from higher levels
            possible_managers = [a.unique_id for a in self.schedule.agents if isinstance(a, EmployeeAgent) and self.get_agent_level(a.unique_id) < current_level]
            if possible_managers:
                new_manager = random.choice(possible_managers)
                # Remove current manager edge
                current_managers = list(self.org_network.predecessors(agent.unique_id))
                if current_managers:
                    current_manager = current_managers[0]
                    self.org_network.remove_edge(current_manager, agent.unique_id)
                # Add new manager edge
                self.org_network.add_edge(new_manager, agent.unique_id)
                print(f"Step {self.current_step}: Reassigned Agent {agent.unique_id} to Manager {new_manager}")
        '''
    def step(self):
        """
        Execute one step of the model.
        """
        self.datacollector.collect(self)
        self.ai_usage_count =0.0  # Reset AI usage count each step
        self.schedule.step()
        self.current_step +=1
        if self.current_step >= self.max_steps:
            self.running = False
            #self.save_data()


        # Evolve social network is handled by agents
        # Evolve organizational network
        #self.evolve_organizational_network()
