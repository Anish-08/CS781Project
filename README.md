# CS781 Project 
## Shield Synthesis with Blackouts

## Steps to run the code:
To activate the docker environment, in the cloned tempest directory, run ```run_using_docker.sh```. It gives the docker environment named ```tempest_development```. 
Attach to this docker image.

Inside the docker image for tempest : ```tempest_development``` which is activated , replace notebooks folder by our notebooks folder and inside it run the code using 

```python3 FaultyActions.py```

To change the parameters go into ```sb3utils.py``` and change values of ```n , k``` and ```method```  (1 means no shield while 0 means with shield).
To change the arena input go into ```FaultyActions.py```, change line 45: env = "MiniGrid-WindyCity-Adv-v0" to your preferred environment from ```opt/tempest/examples/Minigrid/minigrid/__init__.py```
