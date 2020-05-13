## Under the hood
*ZazuML* is built up from 4 main services, 
![](../images/zazuml_diagram.png)

1. The *Predictor* is in charge of selecting the optimal model based on the 
priorities of the user.

2. The *Timer* searches for hyper-parameters, manages and keeps track of trials

3. The *Zazu* is in charge launching local or remote trials and distribution of gpu resources amongst trials

4. The *Trial* 

The [Zazu Model Zoo](https://github.com/dataloop-ai/zoo), was once known as
the *ZaZOO* (my little joke), but to avoid confusion we renamed to just plane old *ZOO*.

![model_space](../images/tetra4.jpeg)

The tetrahedron in the image above represents a vector space where each model occupies a unique 
position with it's own advantages and short comings. *ZazuML* computes the minimal euclidean distance 
between your priorities and model architecture. 
