# Routing Agent

A reinforcement-learning project for topology-aware network routing.

## Train the agent

```bash
python routing_agent.py routing_env --train
```

Stop training with `Ctrl+C`. 
Training outputs are saved inside the `runs/` directory.

## Test the saved model

```bash
python routing_agent_new.py routing_env --evaluate "path/to/model.pt" --output "path/to/save/output.csv"
```
