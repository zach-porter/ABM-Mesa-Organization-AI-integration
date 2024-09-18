# Organization Agent-Based Model (ABM)

This repository contains a simpler version of the [ABM found here](https://github.com/zach-porter/ABM_OOP_ai_organizations). This particluar simulation model uses the mesa package for exploring the interactions between employees and generative AI within an organizational and social network context. The model is built using the [Mesa](https://mesa.readthedocs.io/en/stable/) framework for agent-based modeling and [NetworkX](https://networkx.org/) for network management. 

## Overview

The `Organization ABM` model simulates a workplace environment where employees interact with a generative AI system. Employees have different roles, expertise, and attitudes towards AI. Over time, they exchange knowledge with their peers and the AI system while their knowledge decays. The model also explores how attitudes towards AI evolve based on employees' experiences and knowledge.

### Key Features

- **Generative AI Agent**: A dynamic agent whose knowledge contribution evolves over time and influences employee agents.
- **Employee Agents**: Simulate diverse employees with roles, varying expertise, and attitudes towards AI (positive, neutral, or negative).
- **Organizational and Social Networks**: The model includes both a hierarchical (or onion) organizational structure and a small-world (or scale-free) social network.
- **Knowledge Dynamics**: Employees' knowledge decays over time but can be augmented through interactions with peers and AI.
- **Dynamic Social Networks**: Agents can form or remove social connections dynamically during the simulation.



## Running the Simulation

To run the simulation, instantiate the `OrganizationModel` class in `organization_abm.py` with your desired parameters and execute the model for a given number of steps.

Example:
```python
from organization_abm import OrganizationModel

# Create an instance of the model
model = OrganizationModel(num_employees=100, max_steps=100)

# Run the model
while model.running:
    model.step()
```

### Parameters

- `num_employees`: Number of employee agents.
- `num_levels`: Number of hierarchical levels in the organization.
- `span_of_control`: Number of direct reports per manager.
- `org_network_type`: Type of organizational network (`hierarchical` or `onion`).
- `social_network_type`: Type of social network (`small_world` or `scale_free`).
- `ai_contribution`: Initial knowledge contribution of the generative AI.
- `ai_evolution_rate`: Rate at which the AI evolves over time.
- `knowledge_decay_rate`: Rate at which employee knowledge decays.
- `max_steps`: Maximum number of simulation steps.

## Data Collection

The model collects the following data during each step:
- **Average Knowledge**: The mean knowledge level of all employee agents.
- **Knowledge Standard Deviation**: The variation of knowledge levels among agents.
- **AI Utilization**: Tracks how frequently employees interact with the AI agent.
- **Attitude Distribution**: Counts the number of employees with positive, neutral, or negative attitudes towards AI.

This data can be used to analyze how AI adoption impacts knowledge flow and employee dynamics in the workplace.

## Visualization and Output

The simulation can generate network visualizations and outputs data to the `output/` directory. You can also use the visualizations notebook to visually see the model step by step. You can modify the parameters and output behavior in the `OrganizationModel` class.


## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or suggest new features. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a Pull Request.

## Contact

If you have any questions, feel free to reach out!

- **GitHub**: [zach-porter](https://zach-porter.github.io/contact.html)
