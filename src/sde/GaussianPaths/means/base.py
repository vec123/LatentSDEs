import jax

class BaseMeanModel:
    def __call__(self, t):
        raise NotImplementedError
    
    def get_dot_mu(self, t):
        return jax.jacobian(self.__call__)(t)