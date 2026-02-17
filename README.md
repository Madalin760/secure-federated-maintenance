# Federated Learning Demo for Predictive Maintenance

This is a personal project I created to understand how **Federated Learning** works in an industrial context.

I wanted to see if I could train an AI model to detect engine failures without actually seeing the raw sensor data from the factories (to respect data privacy).

## What I implemented
* **Simulation:** I created a script that generates synthetic data (vibration & temperature) for two imaginary factories.
* **Federated Learning:** Instead of sending data to a central server, I trained the models locally at each "factory".
* **Privacy:** I experimented with adding random noise to the model's parameters (a basic form of Differential Privacy) to see if the model can still learn.

## How to run it
1. Clone the repository.
2. Install dependencies: `pip install tensorflow numpy matplotlib`
3. Run the simulation: `python main.py`

## Results
The simulation shows that the global model improves over 5 rounds of communication, reaching high accuracy even with the added noise.

---
*Built with Python & TensorFlow.*