# CS781 Project 
## Shield Synthesis with Blackouts
### Github Link https://github.com/Anish-08/CS781Project.git

## Steps to run the code:
To activate the docker environment, in the cloned tempest directory, run ```run_using_docker.sh```. It gives the docker environment named ```tempest_development```. 
Attach to this docker image.

Inside the docker image for tempest : ```tempest_development``` which is activated , replace notebooks folder by our notebooks folder and inside it run the code using 

```python3 FaultyActions.py```

## Changing inputs
To change the parameters go into ```sb3utils.py``` and change values of ```n , k``` and ```method```  (1 means no shield while 0 means with shield).

To change the arena input go into ```FaultyActions.py```, change ```line 45: env = "MiniGrid-WindyCity-Adv-v0"``` to your preferred environment from ```opt/tempest/examples/Minigrid/minigrid/__init__.py```

## Interpreting Results
When running the command ```python3 FaultyActions.py```, the output provides detailed information for each iteration. It includes the frequency of events such as the agent walking into lava and colliding with adversaries. Additionally, it reports the average reward within each episode. These outputs are analyzed to evaluate the agent's performance and interpret the results of the experiments effectively. The final outputs of our experiments are stored in the ```final_res``` folder.
